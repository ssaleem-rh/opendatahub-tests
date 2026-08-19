---
test_case_id: TC-SEC-002
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-SEC-002: Init container runs as non-root user

**Objective**: Verify that the git-clone init container is
configured to run as a non-root user, enforcing the security
posture required for credential handling.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a public repository
2. Wait for the job pod to be created
3. Inspect the pod spec for the git-clone init container's
   security context

**Expected Results**:

- The git-clone init container's `securityContext` has
  `runAsNonRoot: true`
- The init container's `runAsUser` is set to a non-zero UID
  (not 0)

**Validation**:

- `oc get pod <job-pod> -o json | jq '.spec.initContainers[] | select(.name | contains("git-clone")) | .securityContext'`
  shows `runAsNonRoot: true` and a non-zero `runAsUser`

**Notes**: To be filled later in the process.
