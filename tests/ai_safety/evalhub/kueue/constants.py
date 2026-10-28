from tests.ai_safety.image_constants import AiSafetyImages

# Sized to fit exactly one eval job pod: adapter (2 CPU / 4Gi) + sidecar (200m / 512Mi)
# = 2200m / 4.5Gi. A quota of 3 CPU / 5Gi admits one job and keeps a second pending.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"

# Distinct from EVALHUB_MT_CR_NAME: the operator names tenant RoleBindings
# '{instance.Name}-{tenant_ns}-...' and rewrites their subjects on every reconcile, so two EvalHub
# instances sharing a name keep overwriting each other's bindings in every tenant namespace.
KUEUE_EVALHUB_CR_NAME = "evalhub-kueue"

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
