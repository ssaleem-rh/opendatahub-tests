---
test_case_id: TC-UPGRADE-002
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: both
---
# TC-UPGRADE-002: Pre-existing S3 jobs run successfully after upgrade

**Objective**: Verify that evaluation jobs created with S3
storage sources before the git storage feature upgrade continue
to process correctly after the upgrade, with no data loss or
behavioral changes.

**Preconditions**:

- OpenShift cluster with EvalHub upgraded from a pre-git-storage
  version to a version supporting `test_data_ref.git`
- At least one S3-based evaluation job was submitted and
  completed before the upgrade

**Test Steps**:

1. Retrieve a pre-upgrade S3 evaluation job via
   `GET /api/v1/evaluations/jobs/{id}` using a job ID from
   before the upgrade
2. Verify the response structure and data are intact
3. Submit a new S3-based evaluation job after the upgrade
4. Wait for the new job to complete
5. Verify the new S3 job completes with the same behavior as
   pre-upgrade jobs

**Expected Results**:

- Pre-upgrade S3 job data is retrievable and unchanged via the
  GET endpoint
- The pre-upgrade job response does not contain any
  git-related fields
- A new S3 job submitted post-upgrade completes with a
  successful status
- New S3 job behavior matches pre-upgrade baseline

**Notes**: To be filled later in the process.
