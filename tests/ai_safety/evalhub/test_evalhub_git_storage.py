"""Git repository as a storage source for evaluation provider test data.

Covers the EvalHub ``test_data_ref.git`` integration. For Kubernetes Job runs the
operator schedules an init container (named ``init``) on the evaluation pod that clones
the configured repository at ``ref`` into the shared ``/test_data`` volume before the
adapter runs; the resolved commit SHA is written to ``.git-metadata`` and surfaced as
``resolved_sha`` on the job's ``test_data_ref`` for reproducibility.

Schema (confirmed against eval-hub docs/src/components/schemas/TestDataRefGit.yaml)::

    test_data_ref:
      git:
        url:        https://github.com/org/repo.git   # required, http(s) only (no ssh)
        ref:        main | v1.2.0 | <full-or-abbrev SHA>   # required
        sub_path:   datasets/lm-eval                   # optional; mounted at /test_data
        secret_ref: my-git-credentials                 # optional; kubernetes.io/basic-auth

Private repos reference a ``kubernetes.io/basic-auth`` Secret (username/password) in the
job namespace via ``secret_ref``; credentials are never placed in the request body.

Fixture repo: the happy-path test clones the public EleutherAI/lm-evaluation-harness repo
pinned to the immutable release tag ``v0.4.12``. A tag (or branch) gets a shallow
``depth=1`` clone, so this stays fast even though the repo has a large history; a raw
commit SHA would force a full clone (eval-hub cmd/eval_runtime_init/git.go). arc_easy pulls
its dataset and the ``google/flan-t5-small`` tokenizer from HuggingFace, so the cloned tree
content is incidental — the test verifies clone-then-eval wiring and that the exact resolved
commit is recorded. The run is capped to ~10 samples via the benchmark ``num_examples``
default, so it is a quick smoke eval, not a full benchmark.

Constants live in constants.py, the payload builders and log/status helpers in utils.py, and
the ``git_basic_auth_secret``/``submit_git_job`` fixtures in conftest.py (next to the PVC
equivalents).
"""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.ai_safety.evalhub.constants import (
    ENV_GIT_REF,
    ENV_GIT_SUBPATH,
    ENV_GIT_URL,
    GIT_DATASET_COMMIT,
    GIT_DATASET_REF,
    GIT_DATASET_REPO_URL,
    GIT_INIT_CONTAINER_NAME,
    GIT_INVALID_REF,
    GIT_MISSING_SUBPATH,
    GIT_SENTINEL_PASSWORD,
    GIT_UNREACHABLE_REPO_URL,
)
from tests.ai_safety.evalhub.utils import (
    capture_git_init_container_logs,
    find_resolved_sha,
    get_evalhub_job_logs_http,
    get_git_init_container,
    get_job_status_message,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

GIT_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-git-storage"})

# The happy-path test needs a real fixture repo/tag (see docstring). If GIT_DATASET_COMMIT is
# ever reset to the all-zeros placeholder, that test skips rather than reporting a product failure.
_PLACEHOLDER_COMMIT: str = "0" * 40
_GIT_FIXTURE_READY: bool = GIT_DATASET_COMMIT != _PLACEHOLDER_COMMIT


@pytest.mark.parametrize("model_namespace", [GIT_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubGitStorage:
    """Git-backed test data source for evaluation jobs."""

    def test_valid_repo_fixed_commit(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a valid repository pinned to an immutable tag,
        when an evaluation job is submitted with test_data_ref.git,
        then the init container clones successfully, the eval runs (a ~10-sample smoke),
        and the run records the exact resolved commit.
        """
        if not _GIT_FIXTURE_READY:
            pytest.skip(
                "Fixture repo/commit not configured: set GIT_DATASET_REPO_URL/GIT_DATASET_REF "
                "(a real public repo + immutable tag) and GIT_DATASET_COMMIT (the commit that tag "
                "resolves to). See module docstring."
            )

        # ref is the tag (fast shallow clone); the default google/flan-t5-small tokenizer and the
        # arc_easy dataset come from HuggingFace, so no tokenizer_path override is needed.
        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-valid-commit",
        )

        # The clone runs in an init container that stages the checkout into /test_data,
        # wired with the git URL and ref via the runtime env ABI.
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec
        init_container = get_git_init_container(spec=spec)
        assert init_container is not None, (
            f"Expected a git clone init container named {GIT_INIT_CONTAINER_NAME!r}, "
            f"got: {[c.name for c in (spec.initContainers or [])]}"
        )
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_URL) == GIT_DATASET_REPO_URL, (
            f"Init container {ENV_GIT_URL} mismatch: {init_env.get(ENV_GIT_URL)!r}"
        )
        assert init_env.get(ENV_GIT_REF) == GIT_DATASET_REF, (
            f"Init container {ENV_GIT_REF} mismatch: {init_env.get(ENV_GIT_REF)!r}"
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        # Clone succeeded and the eval ran to completion with benchmark metrics.
        validate_evalhub_job_completed(job_data=job_data)

        # The run records the exact commit the tag resolves to, for reproducibility.
        resolved_sha = find_resolved_sha(obj=job_data)
        assert resolved_sha == GIT_DATASET_COMMIT, (
            f"Expected recorded resolved_sha {GIT_DATASET_COMMIT!r} (tag {GIT_DATASET_REF!r}), got {resolved_sha!r}"
        )

    def test_invalid_git_reference_fails_cleanly(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a nonexistent branch, tag, or commit,
        when the job is submitted,
        then the job fails cleanly before the evaluation starts.
        """
        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_INVALID_REF,
            job_name="git-invalid-ref",
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        state = job_data.get("status", {}).get("state")
        assert state == "failed", f"Job with an invalid git ref should fail, got state '{state}'"

        # The configured benchmark is still echoed back on a failed job, but because the clone
        # fails before evaluation, none of them carry metrics -> the eval never produced results.
        results = job_data.get("results", {}) or {}
        benchmarks = results.get("benchmarks", []) or []
        benchmarks_with_metrics = [b for b in benchmarks if b.get("metrics")]
        assert not benchmarks_with_metrics, (
            f"Evaluation must not produce metrics when the git ref is invalid, got: {benchmarks_with_metrics}"
        )

        # No resolved commit is recorded for a clone that never succeeded.
        assert find_resolved_sha(obj=job_data) is None, "resolved_sha must not be recorded for a failed clone"

    def test_repository_access_failure_no_secret_leak(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        git_basic_auth_secret: Secret,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a private/unreachable repo with invalid credentials,
        when the job is submitted,
        then the clone fails and no secrets leak into logs or status.
        """
        job_id = submit_git_job(
            url=GIT_UNREACHABLE_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-access-failure",
            secret_ref=git_basic_auth_secret.name,
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        state = job_data.get("status", {}).get("state")
        assert state == "failed", f"Job against an unreachable/private repo should fail, got '{state}'"

        # The credential value must never appear in the status payload...
        assert GIT_SENTINEL_PASSWORD not in str(job_data), "Credential value leaked into job status"

        # ...nor in the aggregated job logs.
        logs_response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert GIT_SENTINEL_PASSWORD not in logs_response.text, "Credential value leaked into job logs"

    def test_missing_test_data_path_fails_with_data_error(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a repo that clones successfully but a sub_path that is absent,
        when the job runs,
        then the evaluation fails with a precise data-related error.
        """
        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-missing-data-path",
            sub_path=GIT_MISSING_SUBPATH,
        )

        # The clone init container is still injected; only the sub_path staging fails.
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        batch_job = batch_jobs[0]
        init_container = get_git_init_container(spec=batch_job.instance.spec.template.spec)
        assert init_container is not None, "Expected a git clone init container even when the sub_path is bad"
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_SUBPATH) == GIT_MISSING_SUBPATH, (
            f"Init container {ENV_GIT_SUBPATH} mismatch: {init_env.get(ENV_GIT_SUBPATH)!r}"
        )

        # The clone succeeds; only sub_path staging fails. The precise data/path error is written
        # to the init container's stdout (not the generic API status message nor the adapter-only
        # logs endpoint), and eval-hub removes the pod once the job is terminal - so capture the
        # init logs live, as soon as the init container terminates.
        init_logs = capture_git_init_container_logs(admin_client=admin_client, batch_job=batch_job)

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        state = job_data.get("status", {}).get("state")
        assert state == "failed", f"Job with a missing sub_path should fail, got '{state}'"

        # The evaluation never produced results - the failure is a data/staging error, not an eval error.
        benchmarks_with_metrics = [
            b for b in (job_data.get("results", {}) or {}).get("benchmarks", []) if b.get("metrics")
        ]
        assert not benchmarks_with_metrics, (
            f"Evaluation must not produce metrics on a bad sub_path: {benchmarks_with_metrics}"
        )

        # The init container's own error names the missing sub_path precisely.
        status_message = get_job_status_message(job_data=job_data)
        haystack = f"{init_logs}\n{status_message}".lower()
        assert "sub_path" in haystack and "not found" in haystack, (
            "Expected a precise data-related error naming the missing sub_path.\n"
            f"init logs: {init_logs!r}\nstatus message: {status_message!r}"
        )
