---
test_case_id: TC-SEC-003
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-SEC-003: Init container has SeccompProfile RuntimeDefault

**Objective**: Verify that the git-clone init container has
SeccompProfile set to `RuntimeDefault` to restrict available
system calls.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a public repository
2. Wait for the job pod to be created
3. Inspect the pod spec for the git-clone init container's
   seccomp profile configuration

**Expected Results**:

- The git-clone init container's `securityContext.seccompProfile`
  is set to `type: RuntimeDefault`

**Validation**:

- `oc get pod <job-pod> -o json | jq '.spec.initContainers[] | select(.name | contains("git-clone")) | .securityContext.seccompProfile'`
  returns `{"type": "RuntimeDefault"}`

**Notes**: To be filled later in the process.
