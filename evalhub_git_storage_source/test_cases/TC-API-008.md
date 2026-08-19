---
test_case_id: TC-API-008
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-API-008: Reject job with missing ref in git config

**Objective**: Verify that the API validates the presence of the
required `ref` field in `test_data_ref.git` and returns an
appropriate error when it is missing.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with a
   `test_data_ref.git` payload that omits `ref`
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating that `ref`
  is a required field

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
