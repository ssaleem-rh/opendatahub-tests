---
test_case_id: TC-REG-002
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: both
---
# TC-REG-002: Existing PVC evaluation job succeeds after API extension

**Objective**: Verify that existing evaluation jobs using PVC
storage sources continue to work without modification after the
`test_data_ref.git` API schema extension is deployed.

**Preconditions**:

- EvalHub API deployed with git storage source support
  (post-extension)
- A PersistentVolumeClaim with evaluation test data exists in
  the evaluation job namespace

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` using a PVC-based
   `test_data_ref` configuration
2. Wait for the job to complete
3. Retrieve the job via
   `GET /api/v1/evaluations/jobs/{id}`
4. Verify the job completed and the response structure matches
   the pre-extension format

**Expected Results**:

- API returns HTTP 200 or 201 status code for the PVC job
  submission
- The evaluation job completes with a successful status
- The GET response preserves the PVC `test_data_ref`
  configuration unchanged

**Notes**: To be filled later in the process.
