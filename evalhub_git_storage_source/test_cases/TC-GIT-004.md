---
test_case_id: TC-GIT-004
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-004: Init container clones private repository with credentials

**Objective**: Verify that the git-clone init container
successfully clones a private repository using HTTPS credentials
from a Kubernetes Secret referenced by `secret_ref`.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Private test repository `evalhub-test-data-private` exists and
  requires HTTPS authentication
- Kubernetes Secret `git-test-creds` exists in the evaluation
  job namespace with valid `username` and `password` (token)
  fields

**Test Steps**:

1. Verify the `git-test-creds` Secret exists in the target
   namespace:
   `oc get secret git-test-creds -n <namespace>`
2. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying the private repository URL, `ref: "main"`, and
   `secret_ref: "git-test-creds"` <!-- pragma: allowlist secret -->
3. Wait for the init container to complete
4. Verify that the cloned files are present at `/test_data/`

**Expected Results**:

- Init container completes with exit code 0
- Files from the private repository are present at `/test_data/`
  in the evaluation container
- No authentication error events in the pod's event log

**Validation**:

- `oc get events --field-selector involvedObject.name=<job-pod>`
  does not contain authentication failure events

**Notes**: To be filled later in the process.
