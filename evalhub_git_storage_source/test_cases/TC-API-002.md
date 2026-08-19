---
test_case_id: TC-API-002
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-API-002: Submit evaluation job with git storage using tag ref

**Objective**: Verify that the POST job submission API accepts a
`test_data_ref.git` payload with a tag ref and returns a
successful response.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with a
  tag `v1.0`

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with a
   `test_data_ref.git` payload specifying a tag ref
2. Verify the API response status code and body

**Expected Results**:

- API returns HTTP 200 or 201 status code
- Response body contains a job ID
- Response body includes the submitted `test_data_ref.git`
  configuration with `ref` set to `v1.0`

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "v1.0"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
