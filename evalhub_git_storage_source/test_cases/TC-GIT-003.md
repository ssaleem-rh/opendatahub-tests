---
test_case_id: TC-GIT-003
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-003: Init container clones public repository at commit SHA

**Objective**: Verify that the git-clone init container
successfully clones a public repository and checks out a specific
commit SHA.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  a known commit SHA

**Test Steps**:

1. Obtain a valid 40-character commit SHA from the public test
   repository
2. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying `ref` as the full commit SHA
3. Wait for the init container to complete
4. Verify that the cloned files at `/test_data/` correspond to
   the exact commit specified

**Expected Results**:

- Init container completes with exit code 0
- Files at `/test_data/` match the repository state at the
  specified commit SHA
- The HEAD of the cloned repository matches the requested
  commit SHA

**Notes**: To be filled later in the process.
