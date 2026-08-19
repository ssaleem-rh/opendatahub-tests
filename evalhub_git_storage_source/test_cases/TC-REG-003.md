---
test_case_id: TC-REG-003
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: both
---
# TC-REG-003: S3 job response does not include git-specific fields

**Objective**: Verify that evaluation jobs submitted with S3
storage sources do not return git-specific metadata fields
(`git_commit_sha`, `test_data_ref.git`) in the API response.

**Preconditions**:

- EvalHub API deployed with git storage source support
  (post-extension)
- S3-based evaluation job configuration is available

**Test Steps**:

1. Submit an evaluation job using S3 storage source
2. Wait for the job to complete
3. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`
4. Inspect the response body for git-specific fields

**Expected Results**:

- The GET response does not contain a `git_commit_sha` field
- The `test_data_ref` object does not contain a `git` key
- The response body structure matches the pre-extension S3 job
  format

**Notes**: To be filled later in the process.
