---
test_case_id: TC-NEG-001
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-001: Reject job with invalid git repository URL

**Objective**: Verify that the API returns an error when an
evaluation job is submitted with an invalid or malformed git
repository URL.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref.git` containing an invalid `repository_url`
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message referencing the
  invalid `repository_url` field

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "not-a-valid-url",
        "ref": "main"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
