---
test_case_id: TC-GIT-001
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-001: Init container clones public repository at branch ref

**Objective**: Verify that the git-clone init container
successfully clones a public repository at a specified branch ref
and populates the shared volume at `/test_data/`.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  a `main` branch containing at least one evaluation dataset file

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying the public repository and `ref: "main"`
2. Wait for the init container to complete
3. Inspect the evaluation pod to verify the init container
   exited with status 0
4. Verify that the cloned repository files are present at the
   `/test_data/` mount path inside the evaluation container

**Expected Results**:

- Init container named with `git-clone` prefix completes with
  exit code 0
- Files from the `main` branch of the public test repository
  are present at `/test_data/` in the evaluation container
- The evaluation container can read files from `/test_data/`

**Validation**:

- `oc get pod <job-pod> -o jsonpath='{.status.initContainerStatuses[*].state.terminated.exitCode}'`
  returns `0`
- `oc exec <job-pod> -c eval -- ls /test_data/` lists
  expected repository files

**Notes**: To be filled later in the process.
