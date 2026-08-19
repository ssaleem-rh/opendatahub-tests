---
test_case_id: TC-API-005
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-API-005: Submit evaluation job with secret_ref for private repository

**Objective**: Verify that the POST job submission API accepts a
`test_data_ref.git` payload with a `secret_ref` referencing a
Kubernetes Secret containing git credentials for private
repository access.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Private test repository `evalhub-test-data-private` exists
- Kubernetes Secret `git-test-creds` exists in the evaluation job
  namespace with valid HTTPS basic auth credentials (username and
  password/token fields)

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with a
   `test_data_ref.git` payload including `secret_ref`
2. Verify the API response status code and that the `secret_ref`
   field is preserved in the response

**Expected Results**:

- API returns HTTP 200 or 201 status code
- Response body contains a job ID
- Response body includes the submitted `test_data_ref.git`
  configuration with `secret_ref` set to `git-test-creds`

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-private.git",
        "ref": "main",
        "secret_ref": "git-test-creds" <!-- pragma: allowlist secret -->
      }
    }
  }'
```

**Notes**: To be filled later in the process.
