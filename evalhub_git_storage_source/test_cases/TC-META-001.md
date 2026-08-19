---
test_case_id: TC-META-001
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-META-001: Commit SHA recorded after job with branch ref

**Objective**: Verify that when an evaluation job uses a branch
ref (e.g., `main`), the resolved commit SHA is recorded as job
metadata after the git-clone init container completes.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  a `main` branch

**Test Steps**:

1. Note the current HEAD commit SHA of the `main` branch in the
   public test repository
2. Submit an evaluation job with `test_data_ref.git` specifying
   `ref: "main"`
3. Wait for the job to reach a running or completed state
4. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`
5. Verify the response contains a `git_commit_sha` field

**Expected Results**:

- The `git_commit_sha` field is present in the GET response
- The value is a 40-character hexadecimal string
- The value matches the HEAD commit SHA of the `main` branch
  at the time the clone was performed

**Notes**: To be filled later in the process.
