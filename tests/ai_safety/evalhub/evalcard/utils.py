from __future__ import annotations

import json

from kubernetes.client.exceptions import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.exceptions import ExecOnPodError
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.ai_safety.evalhub.evalcard.constants import (
    EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS,
    EVALCARD_CARD_VERSION,
    EVALCARD_CONTEXT_REQUIRED_KEYS,
    EVALCARD_METADATA_REQUIRED_KEYS,
    EVALCARD_MLFLOW_ARTIFACT_NAME,
    EVALCARD_MLFLOW_RUN_TAG_KEY,
    EVALCARD_REQUIRED_TOP_LEVEL_KEYS,
    EVALCARD_SCHEMA_VERSION,
)
from tests.ai_safety.evalhub.utils import curl_from_evalhub_pod, get_evalhub_pod

MLFLOW_WORKSPACE_HEADER = "X-MLFLOW-WORKSPACE"
MLFLOW_SERVICE_PORT = 8443
SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

# Pod exec over the Kubernetes websocket API is occasionally flaky and can hang for
# minutes before failing with a raw ApiException (connection timeout) or ExecOnPodError;
# retry rather than aborting on the first transient failure.
_TRANSIENT_EXEC_EXCEPTIONS: dict[type, list] = {ExecOnPodError: [], ApiException: []}


def _mlflow_base_url() -> str:
    return f"https://mlflow.{py_config['applications_namespace']}.svc.cluster.local:{MLFLOW_SERVICE_PORT}/mlflow"


def _mlflow_auth_headers(admin_client: DynamicClient, namespace: str, tenant: str) -> dict[str, str]:
    """Build headers for calling MLflow as the EvalHub pod does.

    MLflow sits behind an auth proxy that requires a bearer token; eval-hub
    authenticates with its own projected ServiceAccount token
    (internal/eval_hub/mlflow/mlflow.go), so the test reads the same token
    from inside the EvalHub pod rather than minting a separate credential.
    """
    pod = get_evalhub_pod(admin_client=admin_client, namespace=namespace)
    token = pod.execute(command=["cat", SA_TOKEN_PATH], container="evalhub").strip()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        MLFLOW_WORKSPACE_HEADER: tenant,
    }


def _find_mlflow_run_id(
    admin_client: DynamicClient,
    namespace: str,
    tenant: str,
    experiment_id: str,
    job_id: str,
) -> str | None:
    headers = _mlflow_auth_headers(admin_client=admin_client, namespace=namespace, tenant=tenant)
    body = json.dumps({
        "experiment_ids": [experiment_id],
        "filter": f"tags.{EVALCARD_MLFLOW_RUN_TAG_KEY} = '{job_id}'",
    })
    raw = curl_from_evalhub_pod(
        admin_client=admin_client,
        namespace=namespace,
        url=f"{_mlflow_base_url()}/api/2.0/mlflow/runs/search",
        method="POST",
        headers=headers,
        data=body,
    )
    runs = json.loads(raw).get("runs", [])
    return runs[0]["info"]["run_id"] if runs else None


def _download_evalcard_artifact(
    admin_client: DynamicClient,
    namespace: str,
    tenant: str,
    experiment_id: str,
    run_id: str,
) -> dict:
    artifact_url = (
        f"{_mlflow_base_url()}/api/2.0/mlflow-artifacts/artifacts/"
        f"{experiment_id}/{run_id}/artifacts/{EVALCARD_MLFLOW_ARTIFACT_NAME}"
    )
    raw_card = curl_from_evalhub_pod(
        admin_client=admin_client,
        namespace=namespace,
        url=artifact_url,
        headers=_mlflow_auth_headers(admin_client=admin_client, namespace=namespace, tenant=tenant),
    )
    return json.loads(raw_card)


def get_evalcard_from_mlflow(
    admin_client: DynamicClient,
    namespace: str,
    tenant: str,
    experiment_id: str,
    job_id: str,
    timeout: int = 300,
) -> dict:
    """Retrieve the evaluation card from its MLflow artifact.

    EvalCard has no HTTP discovery endpoint; eval-hub persists it as an
    `evaluation-card.json` artifact on a dedicated MLflow run, tagged
    `evaluation_job_id=<job_id>`, in the job's experiment. Card persistence
    happens asynchronously after job completion, so this polls for the run
    before downloading the artifact. The generous timeout also absorbs
    occasional multi-minute hangs in the pod-exec websocket connection.
    """
    run_id = None
    for run_id in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        exceptions_dict=_TRANSIENT_EXEC_EXCEPTIONS,
        func=_find_mlflow_run_id,
        admin_client=admin_client,
        namespace=namespace,
        tenant=tenant,
        experiment_id=experiment_id,
        job_id=job_id,
    ):
        if run_id:
            break

    card = None
    for card in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        exceptions_dict=_TRANSIENT_EXEC_EXCEPTIONS,
        func=_download_evalcard_artifact,
        admin_client=admin_client,
        namespace=namespace,
        tenant=tenant,
        experiment_id=experiment_id,
        run_id=run_id,
    ):
        break
    return card


def validate_evalcard_schema(card: dict, job_id: str) -> None:
    """Assert required keys and version strings in an eval card dict."""
    for key in EVALCARD_REQUIRED_TOP_LEVEL_KEYS:
        assert key in card, f"EvalCard missing required top-level key: {key}"

    assert card["schema_version"] == EVALCARD_SCHEMA_VERSION, (
        f"Expected schema_version={EVALCARD_SCHEMA_VERSION}, got {card['schema_version']}"
    )
    assert card["card_version"] == EVALCARD_CARD_VERSION, (
        f"Expected card_version={EVALCARD_CARD_VERSION}, got {card['card_version']}"
    )

    metadata = card["metadata"]
    for key in EVALCARD_METADATA_REQUIRED_KEYS:
        assert key in metadata, f"EvalCard metadata missing key: {key}"
    assert metadata["evaluation_job_id"] == job_id, (
        f"Expected evaluation_job_id={job_id}, got {metadata['evaluation_job_id']}"
    )
    assert metadata["created_at"], "created_at must not be empty"

    context = card["context"]
    for key in EVALCARD_CONTEXT_REQUIRED_KEYS:
        assert key in context, f"EvalCard context missing key: {key}"
    assert context["model"], "model must not be empty"

    # Collection jobs only set context.collection_id; direct benchmark jobs set
    # context.benchmarks instead. The two are mutually exclusive.
    if "collection_id" not in context:
        assert isinstance(context.get("benchmarks"), list), "context.benchmarks must be a list"
        assert len(context["benchmarks"]) >= 1, "context.benchmarks must have at least one entry"

    results = card["results"]
    assert isinstance(results["benchmarks"], list), "results.benchmarks must be a list"
    assert len(results["benchmarks"]) >= 1, "results.benchmarks must have at least one entry"

    for bench in results["benchmarks"]:
        for key in EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS:
            assert key in bench, f"Benchmark result missing key: {key}"


def validate_evalcard_complete_mode(card: dict) -> None:
    """Assert that a complete-mode card has per-benchmark breakdowns and thresholds."""
    for bench in card["results"]["benchmarks"]:
        assert "test" in bench, "Complete card benchmark must have 'test' block"
        test_block = bench["test"]
        assert "primary_score" in test_block, "Complete card test must have primary_score"
        assert "threshold" in test_block, "Complete card test must have threshold"
        assert "pass" in test_block, "Complete card test must have pass boolean"


def validate_evalcard_collection_fields(card: dict, collection_id: str) -> None:
    """Assert collection-specific fields in an eval card."""
    context = card["context"]
    assert context.get("collection_id") == collection_id, (
        f"Expected collection_id={collection_id}, got {context.get('collection_id')}"
    )

    results = card["results"]
    assert "collection" in results, "Collection eval card must have 'collection' in results"
    collection_result = results["collection"]
    assert "test" in collection_result, "Collection results must have 'test' block"
    test_block = collection_result["test"]
    assert "score" in test_block, "Collection test must have score"
    assert "threshold" in test_block, "Collection test must have threshold"
    assert "pass" in test_block, "Collection test must have pass boolean"
