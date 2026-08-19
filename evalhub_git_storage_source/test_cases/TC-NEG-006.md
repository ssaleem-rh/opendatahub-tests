---
test_case_id: TC-NEG-006
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-NEG-006: Reject job with non-HTTPS git URL

**Objective**: Verify that the API rejects evaluation job
submissions that specify a non-HTTPS git protocol URL (e.g.,
SSH or git://) since only HTTPS is supported.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref.git` containing an SSH-style repository URL
2. Verify the API returns an error response
3. Repeat with a `git://` protocol URL
4. Verify the API returns an error response

**Expected Results**:

- API returns HTTP 400 or 422 status code for SSH URL
  (`git@github.com:org/repo.git`)
- API returns HTTP 400 or 422 status code for git protocol URL
  (`git://github.com/org/repo.git`)
- Response body contains an error message indicating the URL
  protocol is not supported

**Test Data**:

```bash
# SSH URL
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "git@github.com:trustyai/evalhub-test-data-public.git",
        "ref": "main"
      }
    }
  }'

# git:// protocol
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "git://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
