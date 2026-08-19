---
test_case_id: TC-META-003
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-META-003: Recorded commit SHA matches actual cloned commit

**Objective**: Verify that the `git_commit_sha` metadata recorded
by the git-clone init container matches the actual HEAD commit of
the cloned repository at `/test_data/`, confirming provenance
accuracy.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   the public repository and `ref: "main"`
2. Wait for the job pod init container to complete
3. Retrieve the `git_commit_sha` from the job metadata via
   `GET /api/v1/evaluations/jobs/{id}`
4. Exec into the evaluation container and run
   `git -C /test_data rev-parse HEAD` to get the actual HEAD
   of the cloned repository
5. Compare the two values

**Expected Results**:

- The `git_commit_sha` from the API response exactly matches
  the output of `git -C /test_data rev-parse HEAD` in the
  evaluation container
- Both values are 40-character hexadecimal strings

**Validation**:

- `oc exec <job-pod> -c eval -- git -C /test_data rev-parse HEAD`
  returns the same SHA as the API `git_commit_sha` field

**Notes**: To be filled later in the process.
