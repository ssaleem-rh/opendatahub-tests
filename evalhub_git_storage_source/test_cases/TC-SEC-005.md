---
test_case_id: TC-SEC-005
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-SEC-005: Credential Secret namespace-scoped to evaluation job namespace

**Objective**: Verify that the git credential Secret must reside
in the same namespace as the evaluation job, and that cross-
namespace Secret references are not permitted.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Kubernetes Secret `git-test-creds` exists in the evaluation
  job namespace (e.g., `evalhub-jobs`)
- No Secret named `git-test-creds` exists in a different
  namespace (e.g., `other-namespace`)

**Test Steps**:

1. Submit an evaluation job in namespace `evalhub-jobs` with
   `secret_ref: "git-test-creds"` — the Secret exists in the <!-- pragma: allowlist secret -->
   same namespace
2. Verify the job is accepted and the init container can mount
   the Secret
3. Submit an evaluation job referencing a Secret name that does
   not exist in the job's namespace
4. Verify the job fails due to the missing Secret

**Expected Results**:

- Job with a Secret in the same namespace succeeds — init
  container mounts the Secret and clones the repository
- Job referencing a non-existent Secret in the job namespace
  fails — the pod enters an error state or the init container
  fails to start with a Secret mount error

**Validation**:

- `oc get events --field-selector involvedObject.name=<failed-pod> -n <namespace>`
  contains a Secret mount failure event

**Notes**: To be filled later in the process.
