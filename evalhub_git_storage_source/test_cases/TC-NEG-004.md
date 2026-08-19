---
test_case_id: TC-NEG-004
source_key: RHAISTRAT-2058
priority: P2
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-004: Reject job specifying both s3_test_data_ref and git

**Objective**: Verify that the API rejects evaluation job
submissions that specify both `s3_test_data_ref` and `git`
storage sources simultaneously, enforcing mutual exclusion.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref` containing both `s3_test_data_ref` and `git`
   fields
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating that
  `s3_test_data_ref` and `git` cannot be specified together

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "s3_test_data_ref": {
        "bucket": "evalhub-test-data",
        "key": "datasets/eval-data.csv"
      },
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
