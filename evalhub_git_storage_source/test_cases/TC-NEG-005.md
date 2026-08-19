---
test_case_id: TC-NEG-005
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-005: Job fails with non-existent secret_ref

**Objective**: Verify that the evaluation job fails when
`secret_ref` references a Kubernetes Secret that does not exist
in the evaluation job namespace.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- No Secret named `nonexistent-secret` exists in the evaluation
  job namespace

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a private repository and
   `secret_ref: "nonexistent-secret"` <!-- pragma: allowlist secret -->
2. Wait for the job pod to be created
3. Verify the pod fails to start or the init container cannot
   mount the Secret

**Expected Results**:

- The job pod enters an error state (e.g.,
  `CreateContainerConfigError` or init container fails to start)
- Pod events contain a message indicating the referenced Secret
  was not found

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-private.git",
        "ref": "main",
        "secret_ref": "nonexistent-secret" <!-- pragma: allowlist secret -->
      }
    }
  }'
```

**Validation**:

- `oc get events --field-selector involvedObject.name=<job-pod>`
  contains a Secret not found error

**Notes**: To be filled later in the process.
