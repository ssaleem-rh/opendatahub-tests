---
test_case_id: TC-API-004
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-API-004: Submit evaluation job with git storage and sub-path

**Objective**: Verify that the POST job submission API accepts a
`test_data_ref.git` payload with an optional `sub_path` field for
targeted dataset access within a repository.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` contains a
  sub-directory `datasets/subset-a/` with evaluation data

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with a
   `test_data_ref.git` payload including `sub_path`
2. Verify the API response status code and that the `sub_path`
   field is preserved in the response

**Expected Results**:

- API returns HTTP 200 or 201 status code
- Response body contains a job ID
- Response body includes the submitted `test_data_ref.git`
  configuration with `sub_path` set to `datasets/subset-a`

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main",
        "sub_path": "datasets/subset-a"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
