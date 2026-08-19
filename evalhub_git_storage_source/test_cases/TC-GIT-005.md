---
test_case_id: TC-GIT-005
source_key: RHAISTRAT-2058
priority: P1
status: Draft
automation_status: Not Started
last_updated: "2026-07-28"
upgrade_phase: post
---
# TC-GIT-005: Init container clones repository sub-path

**Objective**: Verify that when a `sub_path` is specified in the
git configuration, only the files from that sub-directory are
made available at `/test_data/` for the evaluation container.

**Preconditions**:

- EvalHub API v0.4.0+ deployed with git storage source support
- Public test repository `evalhub-test-data-public` contains a
  sub-directory `datasets/subset-a/` with evaluation data files

**Test Steps**:

1. Submit an evaluation job via
   `POST /api/v1/evaluations/jobs` with `test_data_ref.git`
   specifying `sub_path: "datasets/subset-a"`
2. Wait for the init container to complete
3. Verify that only the files from the `datasets/subset-a/`
   sub-directory are present at `/test_data/`

**Expected Results**:

- Init container completes with exit code 0
- Files at `/test_data/` match the content of
  `datasets/subset-a/` from the repository
- Files from other directories in the repository are not
  present at `/test_data/`

**Test Data**:

```bash
curl -X POST "${EVALHUB_API}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_config": { "model": "test-model" },
    "test_data_ref": {
      "git": {
        "repository_url": "https://github.com/trustyai/evalhub-test-data-public.git",
        "ref": "main",
        "sub_path": "datasets/subset-a"
      }
    }
  }'
```

**Notes**: To be filled later in the process.
