---
test_case_id: TC-GIT-006
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-006: Cloned data accessible to evaluation container via shared volume

**Objective**: Verify that the git-clone init container writes
cloned data to a shared `emptyDir` volume and that the evaluation
container can read those files at `/test_data/`.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` exists with
  known file contents (e.g., a file `dataset.csv`)

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying the public repository
2. Wait for the job pod to be running (init container completed,
   evaluation container started)
3. Inspect the pod spec to confirm an `emptyDir` volume is
   mounted at `/test_data/` in both the init container and the
   evaluation container
4. Verify the evaluation container can read specific files from
   `/test_data/`

**Expected Results**:

- Pod spec contains an `emptyDir` volume shared between the
  git-clone init container and the evaluation container
- Both containers mount the volume at `/test_data/`
- The evaluation container can read files (e.g., `dataset.csv`)
  from `/test_data/`

**Validation**:

- `oc get pod <job-pod> -o jsonpath='{.spec.volumes[?(@.emptyDir)].name}'`
  returns the shared volume name
- `oc get pod <job-pod> -o jsonpath='{.spec.containers[0].volumeMounts[?(@.mountPath=="/test_data")]}'`
  confirms the mount

**Notes**: To be filled later in the process.
