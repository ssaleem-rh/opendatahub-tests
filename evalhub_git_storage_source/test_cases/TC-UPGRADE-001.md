---
test_case_id: TC-UPGRADE-001
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-UPGRADE-001: CRD schema accepts test_data_ref.git after upgrade

**Objective**: Verify that after upgrading EvalHub to a version
with git storage support, the CRD schema migration adds the
`test_data_ref.git` field and the API accepts git-based job
submissions.

**Preconditions**:

- OpenShift cluster with EvalHub upgraded from a pre-git-storage
  version to a version supporting `test_data_ref.git`
- TrustyAI Service Operator with updated EvalHub CRD deployed

**Test Steps**:

1. Verify the upgraded EvalHub CRD includes the
   `test_data_ref.git` schema fields:
   `oc get crd <evalhub-crd> -o yaml | grep -A 10 "git"`
2. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying a public repository
3. Verify the API accepts the job and returns a successful
   response

**Expected Results**:

- The EvalHub CRD definition includes `test_data_ref.git` with
  sub-fields for `repository_url`, `ref`, `sub_path`, and
  `secret_ref`
- POST returns HTTP 200/201 with a job ID
- The git-based evaluation job is accepted and scheduled for
  execution

**Notes**: To be filled later in the process.
