---
test_case_id: TC-API-003
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-API-003: Submit evaluation job with git storage using commit SHA ref

**Objective**: Verify that the POST job submission API accepts a
`test_data_ref.git` payload with a full commit SHA as the ref.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with a
  known commit SHA

**Test Steps**:

1. Obtain a valid commit SHA from the public test repository
   (e.g., `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2`)
2. Send a POST request to `/api/v1/evaluations/jobs` with a
   `test_data_ref.git` payload specifying the commit SHA as `ref`
3. Verify the API response status code and body

**Expected Results**:

- API returns HTTP 200 or 201 status code
- Response body contains a job ID
- Response body includes the submitted `test_data_ref.git`
  configuration with `ref` set to the full commit SHA

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" <!-- pragma: allowlist secret -->
      }
    }
  }'
```

**Notes**: To be filled later in the process.
