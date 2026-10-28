from __future__ import annotations

import requests
import structlog

from tests.ai_safety.evalhub.constants import EVALHUB_JOBS_PATH
from tests.ai_safety.evalhub.evalcard.constants import (
    EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS,
    EVALCARD_CARD_VERSION,
    EVALCARD_CONTEXT_REQUIRED_KEYS,
    EVALCARD_METADATA_REQUIRED_KEYS,
    EVALCARD_MLFLOW_ARTIFACT_NAME,
    EVALCARD_OCI_MANIFEST_TAG_TEMPLATE,
    EVALCARD_REQUIRED_TOP_LEVEL_KEYS,
    EVALCARD_RESULTS_REQUIRED_KEYS,
    EVALCARD_SCHEMA_VERSION,
    EVALHUB_EVALCARD_PATH_TEMPLATE,
)
from tests.ai_safety.evalhub.utils import build_headers
from utilities.registry_utils import pull_manifest_from_oci_registry

LOGGER = structlog.get_logger(name=__name__)


def get_evalcard_http(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> requests.Response:
    """GET the eval card for a job via the discovery API.

    Logs the requested URL and the response. A plain-text "404 page not found" means the EvalHub server
    has no such route registered (eval-hub exposes no evalcard endpoint; cards are written to MLflow as
    the ``evaluation-card.json`` artifact and, optionally, to an OCI registry), so the job is fetched to
    tell a missing route apart from a missing job.
    """
    path = EVALHUB_EVALCARD_PATH_TEMPLATE.format(job_id=job_id)
    url = f"https://{host}{path}"
    headers = build_headers(token=token, tenant=tenant)
    LOGGER.info(f"Requesting evalcard: GET {url} (tenant={tenant})")
    response = requests.get(url=url, headers=headers, verify=ca_bundle_file, timeout=30)
    LOGGER.info(
        f"Evalcard response: status={response.status_code} content-type={response.headers.get('Content-Type')} "
        f"body={response.text[:500]!r}"
    )
    if response.status_code == requests.codes.not_found and response.text.strip() == "404 page not found":
        job_response = requests.get(
            url=f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}",
            headers=headers,
            verify=ca_bundle_file,
            timeout=30,
        )
        LOGGER.error(
            f"EvalHub server has no route for {path} (Go mux default 404), while GET job {job_id} returned "
            f"{job_response.status_code}. Cards are stored as the MLflow artifact "
            f"{EVALCARD_MLFLOW_ARTIFACT_NAME!r} or exported to OCI, not served by this API."
        )
    return response


def pull_evalcard_from_oci_registry(registry_url: str, repo: str, oci_tag: str, job_id: str) -> dict:
    """Pull the eval card JSON that EvalHub exported to an OCI registry.

    EvalHub pushes the manifest under the tag ``<oci_tag>-<job_id>`` with the card as its first layer.

    Args:
        registry_url: Base URL of the registry, e.g. ``http://host``.
        repo: Repository the card was exported to.
        oci_tag: The ``oci_tag`` requested in the job's export configuration.
        job_id: Evaluation job id, appended to the tag by EvalHub.

    Returns:
        The eval card as a dict.
    """
    manifest_tag = EVALCARD_OCI_MANIFEST_TAG_TEMPLATE.format(oci_tag=oci_tag, job_id=job_id)
    manifest = pull_manifest_from_oci_registry(registry_url=registry_url, repo=repo, tag=manifest_tag)
    assert manifest.get("layers"), f"OCI manifest {repo}:{manifest_tag} must contain at least one layer: {manifest}"
    layer_digest = manifest["layers"][0]["digest"]
    blob_response = requests.get(url=f"{registry_url}/v2/{repo}/blobs/{layer_digest}", timeout=10)
    assert blob_response.status_code == requests.codes.ok, (
        f"Failed to pull eval card blob {layer_digest} from {repo}: {blob_response.status_code}"
    )
    return blob_response.json()


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
    assert isinstance(context["benchmarks"], list), "context.benchmarks must be a list"
    assert len(context["benchmarks"]) >= 1, "context.benchmarks must have at least one entry"

    results = card["results"]
    for key in EVALCARD_RESULTS_REQUIRED_KEYS:
        assert key in results, f"EvalCard results missing key: {key}"
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
