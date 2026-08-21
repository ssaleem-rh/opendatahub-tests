---
test_case_id: TC-NEG-006
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Automated
last_updated: "2026-08-20"
upgrade_phase: post
---
# TC-NEG-006: Reject secret_ref with a non-HTTPS git URL

**Objective**: Verify that the API rejects an evaluation job that
supplies a `secret_ref` with a non-HTTPS git URL. A `secret_ref` may
only be provided with an HTTPS URL, so a `secret_ref` combined with
any non-HTTPS scheme (`http://`, `git://`, or `git@` SSH) is rejected.
A non-HTTPS URL without a `secret_ref` is accepted.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with
   `test_data_ref.git` containing a `secret_ref` and an `http://`
   repository URL
2. Repeat with a `git://` repository URL and a `secret_ref`
3. Repeat with a `git@` (SSH) repository URL and a `secret_ref`
4. Verify the API returns an error response for each request

**Expected Results**:

- A `secret_ref` is only permitted with an HTTPS URL; a `secret_ref`
  supplied with any non-HTTPS URL is rejected
- API returns HTTP 400 or 422 status code
- Response body contains an error message indicating a `secret_ref`
  requires an HTTPS URL
- No evaluation job is created

**Test Data**:

```bash
# Reject: secret_ref with a non-HTTPS URL. Repeat for each scheme by
# swapping repository_url: http://, git://, and git@ (SSH).
#   http://github.com/trustyai/evalhub-test-data-private.git
#   git://github.com/trustyai/evalhub-test-data-private.git
#   git@github.com:trustyai/evalhub-test-data-private.git
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "http://github.com/trustyai/evalhub-test-data-private.git",
        "ref": "main",
        "secret_ref": "git-test-creds"  # pragma: allowlist secret
      }
    }
  }'
```

**Notes**: Automated in
`tests/ai_safety/evalhub/test_evalhub_git_storage.py` as
`test_secret_ref_with_non_https_url_rejected` (parametrized over the
`http`, `git-protocol`, and `ssh` schemes) plus
`test_non_https_url_without_secret_ref_accepted` for the accepted
contrast. Verified against RHOAI 3.6.0-ea.1 on 2026-08-20: all three
`secret_ref` + non-HTTPS combinations are rejected with a 4xx and no
job is created, while the same non-HTTPS URL without a `secret_ref` is
accepted (202). This supersedes the earlier draft that assumed a
blanket non-HTTPS rejection.
