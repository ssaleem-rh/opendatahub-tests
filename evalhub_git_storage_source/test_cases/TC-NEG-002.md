---
test_case_id: TC-NEG-002
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-002: Job fails with non-existent git ref

**Objective**: Verify that the git-clone init container fails
when the specified ref (branch, tag, or SHA) does not exist in
the target repository.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a ref that does not exist (e.g., `nonexistent-branch-xyz`)
2. Wait for the init container to attempt the clone
3. Verify the init container fails and the job enters an error
   state

**Expected Results**:

- The git-clone init container exits with a non-zero exit code
- The job status transitions to a failed or error state
- Pod events contain an error message related to the git clone
  failure

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "nonexistent-branch-xyz"
      }
    }
  }'
```

**Validation**:

- `oc get pod <job-pod> -o jsonpath='{.status.initContainerStatuses[*].state.terminated.exitCode}'`
  returns a non-zero value

**Notes**: To be filled later in the process.
