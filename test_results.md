# EvalHub Git Storage Source - Exploratory Test Results

**Feature:** `test_data_ref.git` (git-backed evaluation test data)
**Cluster:** shehan-saleems-cluster (RHOAI, TrustyAI operator Managed)
**Fixture repo:** `https://github.com/eval-hub/eval-hub` (public), path `tests/git-testdata`
**Date:** 2026-08-19
**Scope:** Public git repository storage source only.

**Out of scope / excluded from these results:**

- **S3 and PVC storage sources** - separate backends, owned by `test_evalhub_s3_storage.py` and the PVC suite. (mutual-exclusion negatives are still included below, since those exercise the git source.)
- **Private-repo** (secret_ref) and **operator-upgrade** cases - not executed.

**How tests were run:** API-observable cases via `curl` / Python `requests` against `POST /api/v1/evaluations/jobs`; init-container / pod-security cases via `oc`.

## Results

| Test | Scenario | Expected Result | Actual Result | Status |
| --- | --- | --- | --- | --- |
| TC-API-001 | Submit git job with **branch** ref | 202 accepted | 202 accepted | PASS |
| TC-API-002 | Submit git job with **tag** ref (`v1.0.1`) | 202 accepted | 202 accepted | PASS |
| TC-API-003 | Submit git job with **commit SHA** ref | 202 accepted | 202 accepted | PASS |
| TC-API-004 | Submit git job with **sub_path** | 202 accepted | 202 accepted | PASS |
| TC-API-006 | Public repo, **no secret_ref** | 202 accepted | 202 accepted | PASS |
| TC-API-007 | git config **missing `url`** | 400 rejected | 400 rejected | PASS |
| TC-API-008 | git config **missing `ref`** | 400 rejected | 400 rejected | PASS |
| TC-NEG-001 | **Malformed** git URL | 400 rejected | 400 rejected | PASS |
| TC-NEG-002 | **Nonexistent** git ref | 202 accepted, then clone fails cleanly, no metrics | 202, clone fails, no metrics | PASS |
| TC-NEG-004 | **git + s3** both provided | 400 (mutually exclusive) | 400 rejected | PASS |
| TC-NEG-006 | **Non-HTTPS** (`http://`) URL, no credentials | 4xx rejected | **202 accepted** | **FAIL** |
| TC-NEG-007 | **git + pvc** both provided | 400 (mutually exclusive) | 400 rejected | PASS |
| TC-NEG-008 | **git + s3 + pvc** all provided | 400 (mutually exclusive) | 400 rejected | PASS |
| TC-NEG-009 | Client-supplied **`resolved_sha`** with git | 400 (read-only field) | 400 rejected | PASS |
| TC-NEG-010 | **Empty** `test_data_ref` (no source) | 400 (one source required) | 400 rejected | PASS |
| TC-GIT-001 | Init container clones repo at branch ref, populates `/test_data` | init exits 0, volume populated | exit 0, populated | PASS |
| TC-GIT-002 | Init container clones at tag ref | clones & stages | staged | PASS |
| TC-GIT-003 | Init container clones at commit SHA | checks out exact SHA | resolved_sha == pinned SHA | PASS |
| TC-GIT-005 | Init container clones **sub_path** only | only sub-dir at `/test_data` | sub_path staged | PASS |
| TC-GIT-006 | Cloned data accessible to eval container via shared volume | adapter reads `/test_data` | accessible | PASS |
| TC-META-001 | Resolved SHA recorded (branch ref) | resolved_sha recorded | recorded | PASS |
| TC-META-002 | Resolved SHA recorded (tag ref) | tag's commit recorded | recorded | PASS |
| TC-META-003 | Recorded SHA matches actual cloned commit | matches cloned HEAD | matches pinned SHA | PASS |
| TC-META-004 | Resolved SHA retrievable via `GET job` | present in GET response | present | PASS |
| TC-SEC-002 | Init container runs as **non-root** | `runAsNonRoot: true` | confirmed | PASS |
| TC-SEC-003 | Init container seccomp **RuntimeDefault** | `RuntimeDefault` | confirmed | PASS |
| TC-SEC-004 | Init container **drops ALL** capabilities | `drop: [ALL]` | confirmed | PASS |
| TC-E2E-002 | Public git repo evaluation end-to-end | completes with metrics, SHA recorded | completes, metrics present | PASS |
| TC-E2E-003 | Git sub_path evaluation end-to-end | targeted data, completes with metrics | completes, metrics present | PASS |

**Totals: 28 tests — 27 PASS, 1 FAIL.**

---

### TC-NEG-006: Non-HTTPS git URL is accepted without credentials

**What the test checks**
When a git `test_data_ref` uses an insecure `http://` URL, the API should reject
the request. Cloning test data over plain HTTP is insecure (susceptible to
tampering / MITM), so the scheme should be rejected regardless of whether
credentials are attached.

**Expected behaviour**
`POST /api/v1/evaluations/jobs` returns a 4xx validation error rejecting the
non-HTTPS git URL. No job is created.

**Actual behaviour**
The request is **accepted with HTTP 202** and a job is created. The non-HTTPS
scheme is only rejected when a `secret_ref` (credentials) is also present — a
plain `http://` URL **without** credentials passes validation.

#### Exact curl command used

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

**API response returned** — `HTTP 202`

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

**Contrast — the same `http://` URL *with* a `secret_ref` IS rejected** (`HTTP 400`):

```json
{
  "message_code": "request_validation_failed",
  "message": "The request validation failed: 'git url with credentials must use https scheme'. Please check the request and try again.",
  "trace": "ee08e31a-8bd1-4f60-8277-ce24cce6c12f"
}
```

**Why this is considered a failure**
The HTTPS requirement is enforced **conditionally** — only when credentials are
attached (`"git url with credentials must use https scheme"`). A plain `http://`
URL with no credentials bypasses the check entirely and the job is accepted. The
expectation is that insecure `http://` schemes are rejected unconditionally, so
this is a real gap between expected and actual behaviour (not a test-harness
artifact). Recommend raising with the eval-hub team to confirm whether
non-HTTPS should be rejected outright, or whether accepting credential-less
`http://` is intended.
