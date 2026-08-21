# TC-NEG-006 — Passes with the updated spec

A `secret_ref` is only allowed with an HTTPS URL. It is rejected with any other
URL scheme (`http://`, `git://`, `git@` SSH). A non-HTTPS URL on its own — with
no `secret_ref` — is accepted. The tests pass under this updated spec, and I
have opened a merge request to correct the original spec.

## Behaviour

| Request | Result |
| --- | --- |
| `secret_ref` + non-HTTPS URL | Rejected (4xx) |
| non-HTTPS URL, no `secret_ref` | Accepted (202) |

## Reproduction — accepted case (non-HTTPS URL, no `secret_ref`)

```bash
curl -sk -X POST "https://<evalhub-route>/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -H "X-Tenant: evalhub-git-test" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "neg6-http-url",
    "model": {"url": "http://x.svc:8000/v1", "name": "emulatedModel"},
    "benchmarks": [{
      "id": "arc_easy",
      "provider_id": "lm_evaluation_harness",
      "test_data_ref": {"git": {"url": "http://github.com/eval-hub/eval-hub", "ref": "main"}}
    }]
  }'
```

## API response returned

HTTP 202

```json
{
  "resource": {
    "id": "c60c184b-f5b7-49db-be6b-5643ee5f6aa5",
    "tenant": "evalhub-git-test",
    "created_at": "2026-08-19T15:13:58.167354865Z",
    "owner": "shehan"
  },
  "status": {
    "state": "pending",
    "message": {
      "message": "Evaluation job created",
      "message_code": "evaluation_job_created",
      "message_origin": "server"
    }
  },
  "results": {},
  "name": "neg6-http-url",
  "model": {"url": "http://x.svc:8000/v1", "name": "emulatedModel"},
  "benchmarks": [{
    "id": "arc_easy",
    "provider_id": "lm_evaluation_harness",
    "test_data_ref": {"git": {"url": "http://github.com/eval-hub/eval-hub", "ref": "main"}}
  }]
}
```
