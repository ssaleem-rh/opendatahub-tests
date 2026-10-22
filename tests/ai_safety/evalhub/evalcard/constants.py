# EvalCard has no HTTP discovery endpoint; it is persisted as an MLflow run artifact
# (internal/eval_hub/mlflow/artifacts.go) and/or an OCI artifact (internal/eval_hub/evalcards).
EVALCARD_MLFLOW_RUN_TAG_KEY: str = "evaluation_job_id"
EVALCARD_MLFLOW_ARTIFACT_NAME: str = "evaluation-card.json"

# EvalCard version constants
EVALCARD_SCHEMA_VERSION: str = "1.0"
EVALCARD_CARD_VERSION: str = "1.0"

# OCI export constants
EVALCARD_OCI_REPO: str = "evalhub/evalcard"
EVALCARD_OCI_TAG: str = "v1"

# Structural validation: required keys at each level
EVALCARD_REQUIRED_TOP_LEVEL_KEYS: set[str] = {
    "card_version",
    "schema_version",
    "metadata",
    "context",
}

EVALCARD_METADATA_REQUIRED_KEYS: set[str] = {
    "evaluation_job_id",
    "created_at",
    "updated_at",
}

# "benchmarks" and "collection_id" are mutually exclusive on context (collection jobs
# only set collection_id), so they are validated conditionally, not listed here.
EVALCARD_CONTEXT_REQUIRED_KEYS: set[str] = {
    "model",
}

EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS: set[str] = {
    "status",
    "metrics",
}
