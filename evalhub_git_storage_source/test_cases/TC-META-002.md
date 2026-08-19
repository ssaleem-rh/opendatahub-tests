---
test_case_id: TC-META-002
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-META-002: Commit SHA recorded after job with tag ref

**Objective**: Verify that when an evaluation job uses a tag ref
(e.g., `v1.0`), the resolved commit SHA that the tag points to
is recorded as job metadata.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  a tag `v1.0` pointing to a known commit SHA

**Test Steps**:

1. Determine the commit SHA that the `v1.0` tag resolves to in
   the public test repository
2. Submit an evaluation job with `test_data_ref.git` specifying
   `ref: "v1.0"`
3. Wait for the job to reach a running or completed state
4. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`

**Expected Results**:

- The `git_commit_sha` field is present in the GET response
- The value matches the commit SHA that the `v1.0` tag points
  to in the test repository

**Notes**: To be filled later in the process.
