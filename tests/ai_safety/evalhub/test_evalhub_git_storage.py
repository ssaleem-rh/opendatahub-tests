"""Git repository as a storage source for evaluation provider test data.

Covers the EvalHub ``test_data_ref.git`` integration: an init container clones the repo at
``ref`` into the shared ``/test_data`` volume before the adapter runs. The happy path clones
the public EleutherAI/lm-evaluation-harness repo pinned to the immutable tag ``v0.4.12``.
"""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    ENV_GIT_REF,
    ENV_GIT_SUBPATH,
    ENV_GIT_URL,
    EVALHUB_LOG_ADAPTER_CONTAINER,
    GIT_CONFLICT_S3_BUCKET,
    GIT_CONFLICT_S3_KEY,
    GIT_DATASET_COMMIT,
    GIT_DATASET_REF,
    GIT_DATASET_REPO_URL,
    GIT_FIXTURE_READY,
    GIT_INIT_CONTAINER_NAME,
    GIT_INVALID_REF,
    GIT_MISSING_SUBPATH,
    GIT_NON_HTTPS_URLS,
    GIT_SENTINEL_PASSWORD,
    GIT_UNREACHABLE_REPO_URL,
    GIT_VALID_SUBPATH,
    TEST_DATA_MOUNT_PATH,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_git_test_data_ref,
    build_s3_test_data_ref,
    capture_git_init_container_logs,
    find_resolved_sha,
    get_evalhub_job_logs_http,
    get_git_init_container,
    get_job_status_message,
    post_evalhub_job_raw,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

GIT_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-git-storage"})

FIXTURE_SKIP_REASON = (
    "Fixture repo/commit not configured: set GIT_DATASET_REPO_URL/GIT_DATASET_REF (a real public "
    "repo + immutable tag) and GIT_DATASET_COMMIT (the commit that tag resolves to)."
)


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
        then the init container clones into the shared volume, the eval runs, and the exact
        resolved commit is recorded."""
        if not GIT_FIXTURE_READY:
            pytest.skip(FIXTURE_SKIP_REASON)

        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-valid-commit",
        )

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
            f"got: {[container.name for container in (spec.initContainers or [])]}"
        )
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_URL) == GIT_DATASET_REPO_URL, (
            f"Init container {ENV_GIT_URL} mismatch: {init_env.get(ENV_GIT_URL)!r}"
        )
        assert init_env.get(ENV_GIT_REF) == GIT_DATASET_REF, (
            f"Init container {ENV_GIT_REF} mismatch: {init_env.get(ENV_GIT_REF)!r}"
        )

        init_mounts = {mount.mountPath: mount.name for mount in (init_container.volumeMounts or [])}
        assert TEST_DATA_MOUNT_PATH in init_mounts, (
            f"init container must mount the shared test-data volume at {TEST_DATA_MOUNT_PATH}, got {init_mounts}"
        )
        shared_volume = init_mounts[TEST_DATA_MOUNT_PATH]
        adapter_container = next(
            (container for container in spec.containers if container.name == EVALHUB_LOG_ADAPTER_CONTAINER), None
        )
        assert adapter_container is not None, (
            f"Expected an {EVALHUB_LOG_ADAPTER_CONTAINER!r} container, "
            f"got: {[container.name for container in (spec.containers or [])]}"
        )
        adapter_mounts = {mount.name for mount in (adapter_container.volumeMounts or [])}
        assert shared_volume in adapter_mounts, (
            f"adapter must consume the same {shared_volume!r} volume the init container populated"
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_data)

        resolved_sha = find_resolved_sha(obj=job_data)
        assert resolved_sha == GIT_DATASET_COMMIT, (
            f"Expected recorded resolved_sha {GIT_DATASET_COMMIT!r} (tag {GIT_DATASET_REF!r}), got {resolved_sha!r}"
        )

    def test_valid_sub_path_stages_and_runs(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a valid repository and an existing optional sub_path,
        when the job is submitted,
        then the sub_path subtree is staged and the evaluation completes."""
        if not GIT_FIXTURE_READY:
            pytest.skip(FIXTURE_SKIP_REASON)

        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-valid-subpath",
            sub_path=GIT_VALID_SUBPATH,
        )

        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        init_container = get_git_init_container(spec=batch_jobs[0].instance.spec.template.spec)
        assert init_container is not None, "Expected a git clone init container for a sub_path job"
        init_env = {env.name: env.value for env in (init_container.env or [])}
        assert init_env.get(ENV_GIT_SUBPATH) == GIT_VALID_SUBPATH, (
            f"Init container {ENV_GIT_SUBPATH} mismatch: {init_env.get(ENV_GIT_SUBPATH)!r}"
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_data)

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
        then it fails cleanly before the evaluation starts."""
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

        results = job_data.get("results", {}) or {}
        benchmarks_with_metrics = [
            benchmark for benchmark in (results.get("benchmarks") or []) if benchmark.get("metrics")
        ]
        assert not benchmarks_with_metrics, (
            f"Evaluation must not produce metrics when the git ref is invalid, got: {benchmarks_with_metrics}"
        )
        assert find_resolved_sha(obj=job_data) is None, "resolved_sha must not be recorded for a failed clone"

    def test_repository_access_failure_no_secret_leak(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        git_basic_auth_secret: Secret,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given a private/unreachable repo with invalid credentials,
        when the job is submitted,
        then the credential Secret is isolated to the init container and no secret leaks into
        status or logs."""
        job_id = submit_git_job(
            url=GIT_UNREACHABLE_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-access-failure",
            secret_ref=git_basic_auth_secret.name,
        )

        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        spec = batch_jobs[0].instance.spec.template.spec
        creds_volumes = [
            volume.name
            for volume in (spec.volumes or [])
            if volume.secret and volume.secret.secretName == git_basic_auth_secret.name
        ]
        assert creds_volumes, (
            f"Expected a volume backed by Secret {git_basic_auth_secret.name!r}, "
            f"got volumes: {[volume.name for volume in (spec.volumes or [])]}"
        )
        creds_volume = creds_volumes[0]
        init_container = get_git_init_container(spec=spec)
        assert init_container is not None, "Expected a git clone init container to mount the credential Secret"
        init_mounted = {mount.name for mount in (init_container.volumeMounts or [])}
        assert creds_volume in init_mounted, f"credential Secret {creds_volume!r} must be mounted on the init container"
        adapter_container = next(
            (container for container in spec.containers if container.name == EVALHUB_LOG_ADAPTER_CONTAINER), None
        )
        assert adapter_container is not None, f"Expected an {EVALHUB_LOG_ADAPTER_CONTAINER!r} container"
        adapter_mounted = {mount.name for mount in (adapter_container.volumeMounts or [])}
        assert creds_volume not in adapter_mounted, (
            f"credential Secret {creds_volume!r} must NOT be mounted on the {EVALHUB_LOG_ADAPTER_CONTAINER!r} container"
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

        assert GIT_SENTINEL_PASSWORD not in str(job_data), "Credential value leaked into job status"

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
        then the evaluation fails with a precise data-related error."""
        job_id = submit_git_job(
            url=GIT_DATASET_REPO_URL,
            ref=GIT_DATASET_REF,
            job_name="git-missing-data-path",
            sub_path=GIT_MISSING_SUBPATH,
        )

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

        # The precise staging error lands in the init container's stdout, and the pod is GC'd once
        # the job is terminal, so capture the init logs live before waiting for completion.
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

        benchmarks_with_metrics = [
            benchmark
            for benchmark in ((job_data.get("results", {}) or {}).get("benchmarks") or [])
            if benchmark.get("metrics")
        ]
        assert not benchmarks_with_metrics, (
            f"Evaluation must not produce metrics on a bad sub_path: {benchmarks_with_metrics}"
        )

        status_message = get_job_status_message(job_data=job_data)
        haystack = f"{init_logs}\n{status_message}".lower()
        assert "sub_path" in haystack and "not found" in haystack, (
            "Expected a precise data-related error naming the missing sub_path.\n"
            f"init logs: {init_logs!r}\nstatus message: {status_message!r}"
        )

    def test_conflicting_git_and_s3_refs_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Given a job whose test_data_ref sets both git and s3,
        when it is submitted,
        then the API rejects it (exactly one source is allowed) and no job is created."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="git-s3-conflict",
        )
        conflicting_ref = {
            **build_git_test_data_ref(url=GIT_DATASET_REPO_URL, ref=GIT_DATASET_REF),
            **build_s3_test_data_ref(bucket=GIT_CONFLICT_S3_BUCKET, key=GIT_CONFLICT_S3_KEY),
        }
        for benchmark in payload["benchmarks"]:
            benchmark["test_data_ref"] = conflicting_ref

        response = post_evalhub_job_raw(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert 400 <= response.status_code < 500, (
            f"A job specifying both git and s3 test_data_ref must be rejected with a 4xx, "
            f"got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize(
        "non_https_url",
        [pytest.param(url, id=scheme) for scheme, url in GIT_NON_HTTPS_URLS.items()],
    )
    def test_secret_ref_with_non_https_url_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        git_basic_auth_secret: Secret,
        non_https_url: str,
    ) -> None:
        """Given a git test_data_ref that pairs a secret_ref with a non-HTTPS URL
        (http://, git://, or git@ SSH),
        when the job is submitted,
        then the API rejects it (a secret_ref is only permitted with an HTTPS URL) and no job is
        created."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="git-secret-ref-non-https",
        )
        git_ref = build_git_test_data_ref(
            url=non_https_url,
            ref="main",
            secret_ref=git_basic_auth_secret.name,
        )
        for benchmark in payload["benchmarks"]:
            benchmark["test_data_ref"] = git_ref

        response = post_evalhub_job_raw(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert 400 <= response.status_code < 500, (
            f"A secret_ref with a non-HTTPS URL {non_https_url!r} must be rejected with a 4xx, "
            f"got {response.status_code}: {response.text}"
        )

    def test_non_https_url_without_secret_ref_accepted(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        submit_git_job: Callable[..., str],
    ) -> None:
        """Given the same non-HTTPS URL but no secret_ref,
        when the job is submitted,
        then the API accepts it (the non-HTTPS scheme alone is not rejected); the job may later fail
        at clone time, which is out of scope here."""
        job_id = submit_git_job(
            url=GIT_NON_HTTPS_URLS["http"],
            ref="main",
            job_name="git-non-https-no-secret",
        )
        assert job_id, "A non-HTTPS URL without a secret_ref must be accepted (a job id is returned)"
