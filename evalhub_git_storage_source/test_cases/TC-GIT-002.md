---
test_case_id: TC-GIT-002
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-002: Init container clones public repository at tag ref

**Objective**: Verify that the git-clone init container
successfully clones a public repository at a specified tag ref.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  a tag `v1.0`

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying the public repository and `ref: "v1.0"`
2. Wait for the init container to complete
3. Inspect the evaluation pod to verify the init container
   exited with status 0
4. Verify that the cloned files correspond to the `v1.0` tag
   content at `/test_data/`

**Expected Results**:

- Init container completes with exit code 0
- Files at `/test_data/` match the content of the repository
  at the `v1.0` tag
- `git log -1 --format=%H` inside the cloned directory matches
  the commit SHA that the `v1.0` tag points to

**Validation**:

- `oc get pod <job-pod> -o jsonpath='{.status.initContainerStatuses[*].state.terminated.exitCode}'`
  returns `0`

**Notes**: To be filled later in the process.
