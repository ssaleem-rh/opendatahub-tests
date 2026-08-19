---
test_case_id: TC-NEG-007
source_key: RHAISTRAT-2058
priority: P2
status: Draft
automation_status: Not Started
last_updated: "2026-08-19"
upgrade_phase: post
---
# TC-NEG-007: Reject job specifying both git and pvc

**Objective**: Verify that the API rejects evaluation job
submissions that specify both `git` and `pvc` storage sources
simultaneously, enforcing mutual exclusion (exactly one storage
source is allowed).

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref` containing both `pvc` and `git` fields
2. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating that `git`
  and `pvc` cannot be specified together (mutual exclusion of
  test data sources)
- No evaluation job is created

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
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

**Notes**: Confirmed on cluster A (2026-08-19) against eval-hub real API
(`test_data_ref.git.url`/`ref` schema). Result: **HTTP 400** —
`"test_data_ref: exactly one of s3, pvc, or git must be set"`. No job created.
