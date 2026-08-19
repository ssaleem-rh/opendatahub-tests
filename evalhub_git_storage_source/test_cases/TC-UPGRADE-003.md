---
test_case_id: TC-UPGRADE-003
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-UPGRADE-003: New git jobs coexist with legacy S3 jobs post-upgrade

**Objective**: Verify that after upgrading to the git storage
feature version, new git-based evaluation jobs can be submitted
and run alongside existing legacy S3 jobs without interference.

**Preconditions**:

- OpenShift cluster with EvalHub upgraded to a version supporting
  `test_data_ref.git`
- At least one pre-existing S3-based evaluation job in the system
- Public test repository `evalhub-test-data-public` exists

**Test Steps**:

1. Submit a new git-based evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
2. Simultaneously, submit a new S3-based evaluation job
3. Wait for both jobs to complete
4. Retrieve both jobs via
   `GET /api/v1/evaluations/jobs/{id}`
5. List all jobs via
   `GET /api/v1/evaluations/jobs` and verify both storage types
   appear in the listing

**Expected Results**:

- Both jobs are accepted by the API
- Both jobs complete with a successful status
- The git job response contains `git_commit_sha` and
  `test_data_ref.git` fields
- The S3 job response contains `test_data_ref.s3_test_data_ref`
  without git fields
- The job listing includes both storage types without errors

**Notes**: To be filled later in the process.
