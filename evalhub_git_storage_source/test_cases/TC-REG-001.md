---
test_case_id: TC-REG-001
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: both
---
# TC-REG-001: Existing S3 evaluation job succeeds after API extension

**Objective**: Verify that existing evaluation jobs using S3
storage sources continue to work without modification after the
`test_data_ref.git` API schema extension is deployed.

**Preconditions**:

- EvalHub API deployed with both S3 and git storage source
  support (post-extension)
- S3 bucket with evaluation test data is accessible
- Existing S3 evaluation job configuration is available

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` using the existing S3-based
   `test_data_ref` configuration (no `git` field)
2. Wait for the job to complete
3. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`
4. Verify the job completed and the response structure matches
   the pre-extension format

**Expected Results**:

- API returns HTTP 200 or 201 status code for the S3 job
  submission
- The evaluation job completes with a successful status
- The GET response preserves the S3 `test_data_ref`
  configuration unchanged
- No `git_commit_sha` or `git`-related fields appear in the
  response

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "s3_test_data_ref": {
        "bucket": "evalhub-test-data",
        "key": "datasets/regression-test/eval-data.csv"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
