---
test_case_id: TC-SEC-004
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-SEC-004: Init container has dropped ALL capabilities

**Objective**: Verify that the git-clone init container drops all
Linux capabilities to enforce least-privilege execution.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a public repository
2. Wait for the job pod to be created
3. Inspect the pod spec for the git-clone init container's
   capabilities configuration

**Expected Results**:

- The git-clone init container's
  `securityContext.capabilities.drop` includes `ALL`
- No capabilities are added via `capabilities.add`

**Validation**:

- `oc get pod <job-pod> -o json | jq '.spec.initContainers[] | select(.name | contains("git-clone")) | .securityContext.capabilities'`
  returns `{"drop": ["ALL"]}` with no `add` field

**Notes**: To be filled later in the process.
