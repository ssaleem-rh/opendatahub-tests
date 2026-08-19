---
test_case_id: TC-NEG-009
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-08-19"
upgrade_phase: post
---
# TC-NEG-009: Client-supplied resolved commit SHA in git config is not honored

**Objective**: Verify that a client-supplied `resolved_sha` (or
equivalent resolved-commit field) in the submitted `test_data_ref.git`
payload is not honored. The resolved commit SHA is a server-populated,
read-only field recorded after the clone; a value provided by the
client must not be trusted.

**Test Steps**:

1. Send a POST request to `/api/v1/evaluations/jobs` with a valid
   `test_data_ref.git` (repository_url + ref) where the parent
   `test_data_ref` object also carries a bogus `resolved_sha` value
   (per the schema, `resolved_sha` sits on `test_data_ref`, alongside
   `git`, not inside the git config)
2. Inspect the API response and, if the job is accepted, the
   recorded commit SHA after the clone completes

**Expected Results**:

- API returns HTTP 400 rejecting the read-only field
- Response message indicates `resolved_sha` is read-only and must
  not be set on create
- No evaluation job is created, and no client-supplied SHA is ever
  recorded as the job's resolved commit metadata

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main"
      },
      "resolved_sha": "0000000000000000000000000000000000000000"
    }
  }'
```

**Notes**: Confirmed on cluster A (2026-08-19) against eval-hub real API.
Behavior is **reject** (not ignore-and-overwrite): **HTTP 400** —
`"The field 'resolved_sha' is read-only and must not be set on create."`
Per the OpenAPI schema, `resolved_sha` sits on the parent `test_data_ref`,
not inside `git`.
