---
test_case_id: TC-NEG-003
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-003: Job fails with invalid credentials for private repo

**Objective**: Verify that the git-clone init container fails
when the Secret referenced by `secret_ref` contains invalid
credentials for a private repository.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Private test repository `evalhub-test-data-private` exists
- Kubernetes Secret `git-bad-creds` exists in the evaluation
  job namespace with an invalid token value

**Test Steps**:

1. Create a Secret with invalid credentials:

   ```bash
   oc create secret generic git-bad-creds \
     --from-literal=username=testuser \
     --from-literal=password=invalid-token-value \
     -n <evaluation-job-namespace>
   ```

2. Submit an evaluation job with `test_data_ref.git` specifying
   the private repository and `secret_ref: "git-bad-creds"` <!-- pragma: allowlist secret -->
3. Wait for the init container to attempt the clone
4. Verify the init container fails with an authentication error

**Expected Results**:

- The git-clone init container exits with a non-zero exit code
- The job enters a failed or error state
- Init container logs contain a git authentication failure
  message (e.g., HTTP 401 or 403)

**Validation**:

- `oc logs <job-pod> -c <git-clone-init-container>` contains an
  authentication error

**Notes**: To be filled later in the process.
