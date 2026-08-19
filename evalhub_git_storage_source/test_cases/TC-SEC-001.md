---
test_case_id: TC-SEC-001
source_key: RHAISTRAT-2058
priority: P0
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-SEC-001: Git credential Secret mounted only in init container

**Objective**: Verify that the Kubernetes Secret containing git
credentials is volume-mounted exclusively in the git-clone init
container and is not accessible from the evaluation container,
adapter containers, or sidecar containers.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Private test repository `evalhub-test-data-private` exists
- Kubernetes Secret `git-test-creds` exists in the evaluation
  job namespace

**Test Steps**:

1. Submit an evaluation job with `test_data_ref.git` specifying
   `secret_ref: "git-test-creds"` for a private repository <!-- pragma: allowlist secret -->
2. Wait for the job pod to be created
3. Inspect the pod spec to identify which containers have the
   Secret volume mounted
4. Verify the Secret volume mount is present only in the
   git-clone init container definition

**Expected Results**:

- Pod spec shows the `git-test-creds` Secret volume mounted in
  the git-clone init container
- No other container (evaluation, adapter, sidecar) in the pod
  spec has a volume mount referencing the `git-test-creds`
  Secret
- The Secret volume does not appear in
  `.spec.containers[*].volumeMounts`

**Validation**:

- `oc get pod <job-pod> -o json | jq '.spec.initContainers[] | select(.name | contains("git-clone")) | .volumeMounts[] | select(.name | contains("git"))'`
  returns the Secret mount
- `oc get pod <job-pod> -o json | jq '.spec.containers[].volumeMounts[] | select(.name | contains("git"))'`
  returns empty (no matches)

**Notes**: To be filled later in the process.
