---
test_case_id: TC-NEG-008
source_key: RHAISTRAT-2058
priority: P2
status: Draft
automation_status: Not Started
last_updated: "2026-08-19"
upgrade_phase: post
---
# TC-NEG-008: Reject job specifying git, s3, and pvc simultaneously

**Objective**: Verify that the API rejects an evaluation job whose
`test_data_ref` specifies all three storage sources (`git`,
`s3_test_data_ref`, and `pvc`) at once, enforcing that exactly one
source may be set.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref` containing `git`, `s3_test_data_ref`, and `pvc`
   fields together
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating the storage
  sources are mutually exclusive (exactly one of `git`, `s3`, or
  `pvc` is allowed)
- No evaluation job is created

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
      "pvc": {
        "claim_name": "eval-datasets-pvc"
      },
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main"
      }
    }
  }'
```

**Notes**: Confirmed on cluster A (2026-08-19) against eval-hub real API.
Result: **HTTP 400** — `"test_data_ref: exactly one of s3, pvc, or git must
be set"`. No job created.
