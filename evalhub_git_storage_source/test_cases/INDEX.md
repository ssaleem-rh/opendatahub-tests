# Test Case Index — EvalHub Git Storage Source

**Parent Test Plan**: [TestPlan.md](../TestPlan.md)
**Source**: [RHAISTRAT-2058](https://issues.redhat.com/browse/RHAISTRAT-2058)

## Quick Stats

- **Total Test Cases**: 39
- **P0 (Critical)**: 21
- **P1 (High)**: 17
- **P2 (Medium)**: 1

---

## API Endpoint Validation (TC-API)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-API-001](TC-API-001.md) | Submit evaluation job with git storage using branch ref | P0 |
| [TC-API-002](TC-API-002.md) | Submit evaluation job with git storage using tag ref | P0 |
| [TC-API-003](TC-API-003.md) | Submit evaluation job with git storage using commit SHA ref | P0 |
| [TC-API-004](TC-API-004.md) | Submit evaluation job with git storage and sub-path | P0 |
| [TC-API-005](TC-API-005.md) | Submit evaluation job with secret_ref for private repository | P0 |
| [TC-API-006](TC-API-006.md) | Submit evaluation job with public repo without secret_ref | P1 |
| [TC-API-007](TC-API-007.md) | Reject job with missing repository_url in git config | P0 |
| [TC-API-008](TC-API-008.md) | Reject job with missing ref in git config | P0 |

## Git Clone Operations (TC-GIT)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-GIT-001](TC-GIT-001.md) | Init container clones public repository at branch ref | P0 |
| [TC-GIT-002](TC-GIT-002.md) | Init container clones public repository at tag ref | P0 |
| [TC-GIT-003](TC-GIT-003.md) | Init container clones public repository at commit SHA | P0 |
| [TC-GIT-004](TC-GIT-004.md) | Init container clones private repository with credentials | P0 |
| [TC-GIT-005](TC-GIT-005.md) | Init container clones repository sub-path | P1 |
| [TC-GIT-006](TC-GIT-006.md) | Cloned data accessible to evaluation container via shared volume | P0 |

## Security and Credential Isolation (TC-SEC)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-SEC-001](TC-SEC-001.md) | Git credential Secret mounted only in init container | P0 |
| [TC-SEC-002](TC-SEC-002.md) | Init container runs as non-root user | P0 |
| [TC-SEC-003](TC-SEC-003.md) | Init container has SeccompProfile RuntimeDefault | P1 |
| [TC-SEC-004](TC-SEC-004.md) | Init container has dropped ALL capabilities | P1 |
| [TC-SEC-005](TC-SEC-005.md) | Credential Secret namespace-scoped to evaluation job namespace | P1 |

## Commit SHA Metadata (TC-META)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-META-001](TC-META-001.md) | Commit SHA recorded after job with branch ref | P0 |
| [TC-META-002](TC-META-002.md) | Commit SHA recorded after job with tag ref | P0 |
| [TC-META-003](TC-META-003.md) | Recorded commit SHA matches actual cloned commit | P0 |
| [TC-META-004](TC-META-004.md) | Commit SHA retrievable via GET job endpoint | P0 |

## Regression Testing (TC-REG)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-REG-001](TC-REG-001.md) | Existing S3 evaluation job succeeds after API extension | P1 |
| [TC-REG-002](TC-REG-002.md) | Existing PVC evaluation job succeeds after API extension | P1 |
| [TC-REG-003](TC-REG-003.md) | S3 job response does not include git-specific fields | P1 |

## Negative Testing (TC-NEG)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-NEG-001](TC-NEG-001.md) | Reject job with invalid git repository URL | P1 |
| [TC-NEG-002](TC-NEG-002.md) | Job fails with non-existent git ref | P1 |
| [TC-NEG-003](TC-NEG-003.md) | Job fails with invalid credentials for private repo | P1 |
| [TC-NEG-004](TC-NEG-004.md) | Reject job specifying both s3_test_data_ref and git | P2 |
| [TC-NEG-005](TC-NEG-005.md) | Job fails with non-existent secret_ref | P1 |
| [TC-NEG-006](TC-NEG-006.md) | Reject job with non-HTTPS git URL | P1 |

## End-to-End Workflows (TC-E2E)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-E2E-001](TC-E2E-001.md) | Private git repo evaluation job end-to-end | P0 |
| [TC-E2E-002](TC-E2E-002.md) | Public git repo evaluation job end-to-end | P0 |
| [TC-E2E-003](TC-E2E-003.md) | Git sub-path evaluation job end-to-end | P0 |
| [TC-E2E-004](TC-E2E-004.md) | S3 evaluation job unaffected alongside git jobs | P1 |

## Upgrade Testing (TC-UPGRADE)

| Test Case ID | Title | Priority |
| --- | --- | --- |
| [TC-UPGRADE-001](TC-UPGRADE-001.md) | CRD schema accepts test_data_ref.git after upgrade | P1 |
| [TC-UPGRADE-002](TC-UPGRADE-002.md) | Pre-existing S3 jobs run successfully after upgrade | P1 |
| [TC-UPGRADE-003](TC-UPGRADE-003.md) | New git jobs coexist with legacy S3 jobs post-upgrade | P1 |
