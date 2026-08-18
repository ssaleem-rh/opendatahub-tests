from tests.ai_safety.image_constants import AiSafetyImages

MINIO_MC_IMAGE: str = AiSafetyImages.MINIO_MC

EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_METRICS_PATH: str = "/metrics"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_JOBS_PATH: str = "/api/v1/evaluations/jobs"
EVALHUB_JOB_LOGS_PATH_TEMPLATE: str = "/api/v1/evaluations/jobs/{job_id}/logs"
EVALHUB_JOB_BENCHMARK_LOGS_PATH_TEMPLATE: str = "/api/v1/evaluations/jobs/{job_id}/benchmarks/{benchmark_index}/logs"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

# Job log API (RHAISTRAT-1437 / eval-hub HTTP API)
EVALHUB_LOG_CONTENT_TYPE: str = "text/plain"
EVALHUB_LOG_SECTION_PREFIX: str = "=== pod="
EVALHUB_LOG_ADAPTER_CONTAINER: str = "adapter"
EVALHUB_LOG_COMPLETED_MARKER: str = "Evaluation completed successfully"
EVALHUB_LOG_DEFAULT_TAIL_LINES: int = 1000
EVALHUB_LOG_MAX_TAIL_LINES: int = 10000

EVALHUB_APP_LABEL: str = "eval-hub"
EVALHUB_CONTAINER_NAME: str = "evalhub"
EVALHUB_KUBE_RBAC_PROXY_CONTAINER: str = "kube-rbac-proxy"
EVALHUB_COMPONENT_LABEL: str = "api"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION_V1: str = "v1"
EVALHUB_API_VERSION_V1ALPHA1: str = "v1alpha1"
EVALHUB_FULL_API_VERSION_V1: str = f"{EVALHUB_API_GROUP}/v1"
EVALHUB_FULL_API_VERSION_V1ALPHA1: str = f"{EVALHUB_API_GROUP}/v1alpha1"
EVALHUB_KIND: str = "EvalHub"
EVALHUB_PLURAL: str = "evalhubs"

# Multi-tenancy
EVALHUB_TENANT_LABEL_KEY: str = "evalhub.trustyai.opendatahub.io/tenant"
EVALHUB_TENANT_LABEL_VALUE: str = "true"
EVALHUB_COLLECTIONS_PATH: str = "/api/v1/evaluations/collections"
EVALHUB_PROVIDERS_ACCESS_CLUSTERROLE: str = "trustyai-service-operator-evalhub-providers-access"
EVALHUB_MT_CR_NAME: str = "evalhub-mt"
EVALHUB_VLLM_EMULATOR_PORT: int = 8000

# ClusterRole names (kustomize namePrefix applied by operator install)
EVALHUB_JOBS_WRITER_CLUSTERROLE: str = "trustyai-service-operator-evalhub-jobs-writer"
EVALHUB_JOB_CONFIG_CLUSTERROLE: str = "trustyai-service-operator-evalhub-job-config"

# EvalHub Kubernetes runtime (batch Job / ConfigMap) — mirrors eval-hub job_builders.go
EVALHUB_K8S_LABEL_APP: str = "app"
EVALHUB_K8S_LABEL_APP_VALUE: str = "evalhub"
EVALHUB_K8S_LABEL_COMPONENT: str = "component"
EVALHUB_K8S_LABEL_COMPONENT_VALUE: str = "evaluation-job"
EVALHUB_K8S_LABEL_JOB_ID: str = "job_id"
EVALHUB_K8S_ANNOTATION_JOB_ID: str = "eval-hub.github.io/job_id"
EVALHUB_K8S_ANNOTATION_PROVIDER_ID: str = "eval-hub.github.io/provider_id"
EVALHUB_K8S_ANNOTATION_BENCHMARK_ID: str = "eval-hub.github.io/benchmark_id"

# Shared RBAC rules for EvalHub user access
EVALHUB_USER_ROLE_RULES: list[dict[str, list[str]]] = [
    {
        "apiGroups": ["trustyai.opendatahub.io"],
        "resources": ["evaluations", "collections", "providers"],
        "verbs": ["get", "list", "create", "update", "delete"],
    },
    {
        "apiGroups": ["mlflow.kubeflow.org"],
        "resources": ["experiments"],
        "verbs": ["create", "get"],
    },
]

# Provider IDs for system providers
LM_EVALUATION_HARNESS_PROVIDER_ID: str = "lm_evaluation_harness"

# Garak provider
GARAK_SIMPLE_PROVIDER_ID: str = "garak"
GARAK_PROVIDER_ID: str = "garak-kfp"
GARAK_BENCHMARK_ID: str = "intents"
GARAK_QUICK_BENCHMARK_ID: str = "quick"
GARAK_JOB_TIMEOUT: int = 1800  # 30 minutes
GARAK_JOB_POLL_INTERVAL: int = 30  # seconds

# Job service account naming
EVALHUB_JOB_SA_PREFIX: str = "evalhub-"
EVALHUB_JOB_SA_SUFFIX: str = "-job"

# Garak intents CSV
GARAK_INTENTS_S3_KEY: str = "intents/misinformation_prompts.csv"
MINIO_UPLOADER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}

# Minimal MinIO for simple-mode intents (no DSPA needed)
SIMPLE_MINIO_ACCESS_KEY: str = "minioadmin"
SIMPLE_MINIO_SECRET_KEY: str = "minioadmin"
SIMPLE_MINIO_BUCKET: str = "evalhub-data"

# PVC storage test data
PVC_TEST_DATA_NAME: str = "evalhub-test-data"
PVC_TEST_DATA_SIZE: str = "2Gi"
PVC_TOKENIZER_PATH: str = "/test_data/tokenizer"

# Git storage test data (test_data_ref.git)
# Runtime ABI between the operator-injected clone init container and the pod
# (eval-hub: internal/eval_hub/runtimes/k8s/job_builders.go, cmd/eval_runtime_init).
GIT_INIT_CONTAINER_NAME: str = "init"
ENV_GIT_URL: str = "TEST_DATA_GIT_URL"
ENV_GIT_REF: str = "TEST_DATA_GIT_REF"
ENV_GIT_SUBPATH: str = "TEST_DATA_GIT_SUBPATH"

# Valid, reachable public repo pinned to an immutable tag. A tag (or branch) gets a fast
# shallow (depth=1) clone; GIT_DATASET_COMMIT is the exact commit that tag resolves to and is
# what the run must record as resolved_sha. arc_easy fetches its dataset and the default
# google/flan-t5-small tokenizer from HuggingFace, so no tokenizer needs to live in the repo.
GIT_DATASET_REPO_URL: str = "https://github.com/EleutherAI/lm-evaluation-harness.git"
GIT_DATASET_REF: str = "v0.4.12"  # immutable release tag; also used by the bad-sub_path test
GIT_DATASET_COMMIT: str = (
    "6d642546f4688648fced259eb3302efd36ece5af"  # commit v0.4.12 resolves to  # pragma: allowlist secret
)

# Negative cases.
# Non-hex ref that is neither a branch nor a tag -> clone fails during ls-remote, before any
# evaluation, with "not found as a branch or tag, and does not look like a commit SHA".
GIT_INVALID_REF: str = "no-such-branch-does-not-exist"
# Unreachable / nonexistent repo for the access-failure test (paired with bogus creds).
GIT_UNREACHABLE_REPO_URL: str = "https://github.com/opendatahub-io/eval-git-private-does-not-exist.git"
# Bad sub_path inside an otherwise-valid checkout: clone succeeds, staging fails with
# "sub_path %q not found in repository".
GIT_MISSING_SUBPATH: str = "definitely/not/a/real/path"

# basic-auth Secret for the access-failure test. The password is a recognizable sentinel so
# tests can assert it never leaks into job status or logs.
GIT_CREDS_SECRET_NAME: str = "git-basic-auth-creds"
GIT_SENTINEL_USERNAME: str = "sentinel-user"
GIT_SENTINEL_PASSWORD: str = "SENTINEL-GIT-PAT-do-not-log-3f9c12ab"  # pragma: allowlist secret

# Hardware profile
EVALHUB_DEFAULT_HARDWARE_PROFILE: str = "default-profile"

# ServiceMonitor and metrics Service
EVALHUB_METRICS_SERVICE_SUFFIX: str = "-metrics"
EVALHUB_METRICS_PORT: int = 8081
EVALHUB_METRICS_COMPONENT_LABEL: str = "metrics"
EVALHUB_SCRAPE_INTERVAL: str = "30s"

# OTEL Collector constants
OTEL_COLLECTOR_NAMESPACE: str = "otel-collector"
OTEL_COLLECTOR_GRPC_PORT: int = 4317
OTEL_COLLECTOR_HTTP_PORT: int = 4318
OTEL_COLLECTOR_PROMETHEUS_PORT: int = 8889

# OTEL error patterns that indicate initialization failure
OTEL_ERROR_PATTERNS: tuple[str, ...] = (
    "failed to initialize meter",
    "meter provider error",
    "panic",
    "OTEL initialization failed",
)

# OTLP export indicators in collector logs
OTLP_INDICATORS: tuple[str, ...] = (
    "ResourceMetrics",
    "ScopeMetrics",
    "http.server.request",
    "github.com/eval-hub",
)
