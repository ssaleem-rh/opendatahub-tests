---
test_case_id: TC-NEG-010
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-08-19"
upgrade_phase: post
---
# TC-NEG-010: Reject job with empty test_data_ref (no storage source)

**Objective**: Verify that the API rejects an evaluation job whose
`test_data_ref` is present but specifies none of the supported
storage sources (`git`, `s3`, or `pvc`), enforcing that exactly one
source must be set.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with an empty
   `test_data_ref` object (no `git`, `s3_test_data_ref`, or `pvc`)
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating that a test
  data storage source is required (exactly one of `git`, `s3`, or
  `pvc` must be specified)
- No evaluation job is created

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {}
  }'
```

**Notes**: Confirmed on cluster A (2026-08-19) against eval-hub real API.
Result: **HTTP 400** — `"test_data_ref: one of s3, pvc, or git must be set"`.
No job created.
