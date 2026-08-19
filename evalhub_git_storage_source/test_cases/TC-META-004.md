---
test_case_id: TC-META-004
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-META-004: Commit SHA retrievable via GET job endpoint

**Objective**: Verify that the resolved commit SHA is exposed
in the `GET /api/v1/evaluations/jobs/{id}` response for
reproducibility tracking.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- A completed evaluation job exists that used git storage source

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   a public repository
2. Wait for the job to complete
3. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`
4. Inspect the response body for the `git_commit_sha` field

**Expected Results**:

- The GET response includes a `git_commit_sha` field at the
  top level of the job object
- The value is a non-empty 40-character hexadecimal string
- The `test_data_ref.git` section in the response preserves the
  original `repository_url` and `ref` values from the
  submission

**Expected Response**:

```json
{
  "id": "job-a1b2c3d4",
  "status": "completed",
  "test_data_ref": {
    "git": {
      "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
      "ref": "main"
    }
  },
  "git_commit_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" <!-- pragma: allowlist secret -->
}
```

**Notes**: To be filled later in the process.
