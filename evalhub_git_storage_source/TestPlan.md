---
feature: evalhub_git_storage_source
source_key: RHAISTRAT-2058
source_type: strat
status: Draft
author: TrustyAI
components:
- AI Evaluations
additional_docs: []
last_updated: '2026-07-28'
version: 1.2.0
reviewers: []
---
# EvalHub Git Storage Source Test Plan

## TrustyAI – EvalHub Git Repository Storage Testing

**Strategy**: [RHAISTRAT-2058](https://issues.redhat.com/browse/RHAISTRAT-2058)

---

## 1. Executive Summary

### 1.1 Purpose

This test plan validates the addition of a Git repository storage
source to EvalHub's `test_data_ref` API. The feature enables
evaluation jobs to clone test data directly from a Git repository
via a new git-clone init container, eliminating the manual S3
staging step and preserving commit-level provenance for
reproducibility.

Testing focuses on verifying correct Git clone operations for
public and private repositories, credential isolation within
Kubernetes Secrets, commit SHA metadata recording, API validation
for the new `git` field, and backward compatibility with existing
S3 storage paths.

### 1.2 Scope

#### In Scope (TrustyAI Responsibilities)

- Job submission API extension with `test_data_ref.git`
  configuration (repository URL, ref, optional sub-path)
- Git-clone init container implementation for populating shared
  volumes with repository content at `/test_data/`
- Private repository credential support via namespace-scoped
  Kubernetes Secrets
- Resolved commit SHA recording as evaluation job metadata for
  reproducibility
- Public repository cloning without credential requirements
- API validation to prevent simultaneous S3 and Git storage
  specification
- Security isolation ensuring git credentials are mounted only
  in init containers
- Backward compatibility with existing S3 and PVC job paths

#### Out of Scope (Other Teams)

- Git LFS support (RFE explicit exclusion)
- Webhook-triggered evaluations on git push (RFE explicit
  exclusion)
- UI changes for git source selection in odh-dashboard
  eval-hub-ui (RFE explicit exclusion)
- Support for non-HTTPS git protocols (SSH, git://)

### 1.3 Test Objectives

1. Verify successful evaluation job execution with public Git
   repositories using branch, tag, and commit SHA references
2. Validate private Git repository access using Kubernetes
   Secrets with proper credential isolation to the init
   container only
3. Confirm commit SHA metadata recording and retrieval via
   `GET /api/v1/evaluations/jobs/{id}` for reproducibility
   tracking
4. Ensure backward compatibility with existing S3-based
   evaluation jobs remains unaffected
5. Validate API request validation correctly rejects jobs
   specifying both `s3_test_data_ref` and `git` storage sources
6. Verify git-clone init container security posture (non-root
   execution, capability restrictions, SeccompProfile)
7. Test sub-path repository cloning functionality for targeted
   dataset access

---

## 2. Test Strategy

### 2.1 Test Levels

- **API Integration Testing** — REST endpoint testing for the
  new `test_data_ref.git` field in
  `POST /api/v1/evaluations/jobs` and commit SHA retrieval via
  `GET /api/v1/evaluations/jobs/{id}`
- **Data Validation Testing** — Git clone operations, commit
  SHA recording, file system integrity at `/test_data/` mount
  path
- **Functional Testing** — Evaluation job workflow with git
  storage source, credential handling, repository access
- **Security Testing** — Credential isolation verification,
  init container privilege restrictions, Secret mounting
  validation

### 2.2 Test Types

- **Positive Testing** — Valid git URLs, supported refs
  (branch/tag/commit), successful clone operations, public and
  private repository access
- **Negative Testing** — Invalid repository URLs,
  authentication failures, malformed Secret credentials, timeout
  scenarios, missing required fields
- **Boundary Testing** — Large repository sizes, deep commit
  histories, network latency limits, emptyDir volume capacity
- **Regression Testing** — Existing S3 and PVC storage sources
  remain unaffected after API schema extension

### 2.3 Test Priorities

- **P0 (Critical)** — Core git storage functionality: clone
  operations succeed for private repos with credentials,
  credential isolation enforced, commit SHA recorded, and API
  validation for git-specific fields
- **P1 (High)** — Public repository cloning without requiring a
  secret_ref, existing S3 and PVC job paths remain unaffected
  (backward compatibility), error handling for invalid inputs,
  job builder construction verification
- **P2 (Medium)** — API validation for mutually exclusive
  storage sources, timeout handling, edge cases with large
  repositories or unusual ref formats

---

## 3. Test Environment

### 3.1 Test Cluster Configuration

- OpenShift cluster (version TBD -- pending release planning;
  the strategy does not specify a minimum OpenShift version)
- RHOAI with TrustyAI components deployed
- EvalHub API v0.4.0+ with extensible storage sources
  (per strategy Dependencies: "EvalHub API v0.4.0+ (internal,
  exists): Base API with test_data_ref field and init container
  support")
- Extensible storage source framework from RHAISTRAT-2056
  (per strategy Dependencies: "RHAISTRAT-2056 -- Extensible
  Storage Sources (internal, exists): Parent outcome providing
  the architectural framework for additional storage types")
- TrustyAI Service Operator with EvalHub CRD support
- UBI 9 minimal base image with `git` binary (per strategy
  Dependencies: "UBI 9 minimal image with git (external,
  exists): Git binary available in base image for init
  container")
- FIPS-compliant TLS for HTTPS git operations
- Namespace-scoped Secret mounting capabilities

### 3.2 Test Data Requirements

- Public git repositories containing sample evaluation datasets
- Private git repositories requiring HTTPS authentication
- Repositories with multiple ref types (branches, tags, commit
  SHAs)
- Repositories with sub-path directory structures for targeted
  cloning
- Large repositories for timeout and performance scenario
  testing
- Kubernetes Secrets containing git credentials in the
  documented format (HTTPS basic auth: username + password/token,
  per strategy risk mitigation: "Document the supported
  credential Secret format explicitly (HTTPS basic auth:
  username + password/token fields)")
- Existing S3-based evaluation job configurations for
  regression testing

#### Test Repository Setup

Testers must prepare the following repositories before test
execution:

1. **Public test repository**: Create a public GitHub repository
   (e.g., `evalhub-test-data-public`) containing sample
   evaluation datasets. The repository must include:
   - At least one branch (e.g., `main`), one tag (e.g.,
     `v1.0`), and multiple commits for ref-type testing
   - A sub-directory (e.g., `datasets/subset-a/`) containing a
     subset of evaluation data for sub-path cloning tests
   - A small dataset (< 10 MB) for functional tests

2. **Private test repository**: Create a private GitHub or
   enterprise git repository (e.g.,
   `evalhub-test-data-private`) with the same structure as the
   public repository. Configure HTTPS basic auth credentials
   (username + personal access token) and store them in a
   Kubernetes Secret:

   ```bash
   kubectl create secret generic git-test-creds \
     --from-literal=username=<git-username> \
     --from-literal=password=<git-token> \
     -n <evaluation-job-namespace>
   ```

3. **Large test repository**: Identify or create a repository
   exceeding 500 MB for timeout and shallow clone testing

### 3.3 Test Users

- Users with RBAC permissions to submit evaluation jobs via the
  EvalHub API
- EvalHub job ServiceAccount with sufficient RBAC to mount
  Secrets in the evaluation job namespace (per strategy
  Assumptions: "EvalHub's existing job ServiceAccount has
  sufficient RBAC to mount Secrets in the evaluation job
  namespace"). Specific Role/RoleBinding names are TBD --
  pending implementation documentation.
- Namespace administrators capable of creating and managing git
  credential Secrets (must be able to create Secrets in the
  evaluation job namespace, as the strategy requires "The Secret
  must reside in the same namespace as the evaluation job")
- Users with varying permission levels for security boundary
  testing

---

## 4. API Endpoints Under Test

| Endpoint | Method | Purpose | Priority |
| ---------- | -------- | --------- | ---------- |
| `/api/v1/evaluations/jobs` | POST | Job submission with extended `test_data_ref.git` schema | P0 |
| `/api/v1/evaluations/jobs/{id}` | GET | Job retrieval to verify `git_commit_sha` metadata recording | P0 |
| `test_data_ref.git` validation | API | Validation for git-specific fields (URL, ref, sub-path, secret_ref) | P0 |
| git-clone init container | Component | Kubernetes init container for repository cloning and commit SHA recording | P0 |
| Secret mounting logic | Component | Credential Secret isolation to init container only | P0 |
| Job builder extension | Component | Kubernetes Job construction with git-clone init container | P1 |
| S3 job path validation | Component | Backward compatibility verification for existing S3 workflows | P1 |
| Mutual exclusion validation | API | Validation rejecting simultaneous `s3_test_data_ref` and `git` | P2 |

### 4.1 Example API Payloads

**Sample `POST /api/v1/evaluations/jobs` request body with
`test_data_ref.git` (private repository)**:

Per the strategy, the caller provides a repository URL, a ref
(branch, tag, or commit SHA), an optional sub-path, and a
`secret_ref` for private repositories:

```json
{
  "evaluation_config": { ... },
  "test_data_ref": {
    "git": {
      "repository_url": "https://github.com/org/eval-datasets.git",
      "ref": "v1.0",
      "sub_path": "datasets/subset-a",
      "secret_ref": "git-test-creds" <!-- pragma: allowlist secret -->
    }
  }
}
```

**Sample `POST` request body for public repository** (no
`secret_ref` required, per strategy P1 requirement):

```json
{
  "evaluation_config": { ... },
  "test_data_ref": {
    "git": {
      "repository_url": "https://github.com/org/public-eval-data.git",
      "ref": "main"
    }
  }
}
```

**Sample `GET /api/v1/evaluations/jobs/{id}` response showing
`git_commit_sha` metadata** (per strategy P0 requirement:
"Resolved commit SHA recorded as evaluation job metadata for
reproducibility"):

```json
{
  "id": "job-12345",
  "status": "completed",
  "test_data_ref": {
    "git": {
      "repository_url": "https://github.com/org/eval-datasets.git",
      "ref": "v1.0",
      "sub_path": "datasets/subset-a"
    }
  },
  "git_commit_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" <!-- pragma: allowlist secret -->
}
```

> **Note**: The exact JSON field names and response structure are
> derived from the strategy's technical approach. Final field
> names are TBD -- pending API implementation documentation.

---

## 5. Test Cases

> Test cases have been generated. See the full index at
> [test_cases/INDEX.md](test_cases/INDEX.md).

**Test Cases Directory**: [test_cases/](test_cases/)
**Complete Test Case Index**: [test_cases/INDEX.md](test_cases/INDEX.md)

### 5.1 Test Case Organization

| Category | Test Cases | Priority Distribution |
| ---------- | ------------ | ---------------------- |
| TC-API — API Endpoint Validation | 8 | 7 P0, 1 P1 |
| TC-GIT — Git Clone Operations | 6 | 5 P0, 1 P1 |
| TC-SEC — Security & Credential Isolation | 5 | 2 P0, 3 P1 |
| TC-META — Commit SHA Metadata | 4 | 4 P0 |
| TC-REG — Regression Testing | 3 | 3 P1 |
| TC-NEG — Negative Testing | 6 | 5 P1, 1 P2 |
| TC-E2E — End-to-End Workflows | 4 | 3 P0, 1 P1 |
| TC-UPGRADE — Upgrade Testing | 3 | 3 P1 |

### 5.2 Test Case Naming Convention

Test cases follow the naming pattern: `TC-<CATEGORY>-<NUMBER>`

- `TC-API` — API endpoint validation and request/response testing
- `TC-GIT` — Git clone operations and repository access
- `TC-SEC` — Security and credential isolation testing
- `TC-META` — Commit SHA metadata recording and retrieval
- `TC-REG` — Regression testing for existing S3/PVC paths
- `TC-NEG` — Negative testing and error handling
- `TC-E2E` — End-to-end evaluation job workflows

---

## 6. E2E Test Scenarios

End-to-end scenarios that validate the user journeys defined in the
strategy. Each scenario maps to one or more TC-E2E-*.md test cases
generated by `/test-plan-create-cases`.

> **Requirement**: At least one E2E scenario MUST be generated for
> each P0 endpoint in Section 4.
> E2E scenarios will be filled by `/test-plan-create-cases`.

### 6.1 Scenario Summary

| ID | Scenario | Endpoints Covered | Priority |
| ---- | ---------- | ------------------- | ---------- |
| TC-E2E-001 | Private git repo evaluation job end-to-end | POST jobs, GET jobs/{id}, git-clone init container, Secret mounting, test_data_ref.git validation | P0 |
| TC-E2E-002 | Public git repo evaluation job end-to-end | POST jobs, GET jobs/{id}, git-clone init container, test_data_ref.git validation | P0 |
| TC-E2E-003 | Git sub-path evaluation job end-to-end | POST jobs, GET jobs/{id}, git-clone init container, test_data_ref.git validation | P0 |
| TC-E2E-004 | S3 evaluation job unaffected alongside git jobs | POST jobs, GET jobs/{id}, S3 job path validation | P1 |

### 6.2 E2E Coverage Matrix

| Endpoint (from Section 4) | E2E Scenarios |
| ---------------------------- | --------------- |
| `POST /api/v1/evaluations/jobs` | TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-E2E-004 |
| `GET /api/v1/evaluations/jobs/{id}` | TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-E2E-004 |
| `test_data_ref.git` validation | TC-E2E-001, TC-E2E-002, TC-E2E-003 |
| git-clone init container | TC-E2E-001, TC-E2E-002, TC-E2E-003 |
| Secret mounting logic | TC-E2E-001 |
| Job builder extension | TC-E2E-001, TC-E2E-002, TC-E2E-003 |
| S3 job path validation | TC-E2E-004 |
| Mutual exclusion validation | — |

---

## 7. Non-Functional Requirements

Each category below must be explicitly addressed. If a category
does not apply to this feature, state **Not Applicable** with a
brief justification.

### 7.1 Disconnected/Air-Gapped

**Not Applicable** — This feature requires network access for
HTTPS git clone operations against external repositories. Git
repository storage is fundamentally incompatible with
disconnected environments. The feature depends on outbound HTTPS
connectivity to git hosting services.

### 7.2 Upgrade/Migration

Verify that existing evaluation jobs with S3 or PVC storage
sources continue to work after the API schema extension. Test CRD
schema migration for the new `test_data_ref.git` field. Validate
that jobs created before the feature upgrade process correctly
and new git-based jobs work alongside legacy storage jobs.

### 7.3 Performance/Scalability

Test git clone timeout behavior with large repositories (the
strategy identifies this as a risk). Validate init container
resource consumption during clone operations. Test concurrent
evaluation jobs using git storage to ensure no resource
contention on the shared `emptyDir` volume. Measure impact of
git clone operations on overall job startup time compared to
S3/PVC sources.

### 7.4 RBAC/Authorization

Verify that git credential Secrets are only accessible to the
git-clone init container and not mounted in adapter or sidecar
containers. Test namespace-scoped Secret access enforcement.
Validate that the EvalHub job ServiceAccount has appropriate
RBAC permissions for Secret mounting without over-privileging
other containers in the pod spec.

---

## 8. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
| ------ | -------- | ------------- | ------------ |
| Git clone timeout on large repositories causing init container failure | High | Medium | Implement shallow clone (`--depth 1`) by default for branch/tag refs; support configurable clone timeout with sensible default |
| Credential format mismatch — users providing SSH keys instead of HTTPS tokens | Medium | High | Document supported Secret format explicitly (HTTPS basic auth); validate Secret structure in API before creating the job |
| Init container image size increase from adding git binary | Low | Low | Git already available in UBI 9 minimal; use dedicated lightweight git-clone image if needed |
| Network dependency breaks air-gapped deployments | Medium | Low | Document network requirements clearly; feature not supported in disconnected environments |
| Secret credential leakage to adapter or sidecar containers | High | Low | Implement strict volume mount isolation; validate through pod spec inspection testing |

---

## 9. Test Environment Requirements

### 9.1 Infrastructure

- OpenShift cluster with namespace isolation and RBAC support
- TrustyAI Service Operator deployment with EvalHub CRD support
- EvalHub API v0.4.0+ deployment with extensible storage sources
  (per strategy Dependencies: "EvalHub API v0.4.0+ (internal,
  exists): Base API with test_data_ref field and init container
  support")
- Extensible storage source framework from RHAISTRAT-2056
  (per strategy Dependencies: "RHAISTRAT-2056 -- Extensible
  Storage Sources (internal, exists): Parent outcome providing
  the architectural framework for additional storage types")
- Git repository hosting (public repos e.g. GitHub, private
  enterprise git hosting)
- Container registry access for UBI 9 images (per strategy
  Dependencies: "UBI 9 minimal image with git (external,
  exists)")
- Network connectivity allowing HTTPS outbound connections to
  git repositories

### 9.2 Configuration

- Extended EvalHub CRD schema supporting `test_data_ref.git`
  fields
- RBAC configuration granting EvalHub ServiceAccount Secret
  access in job namespaces
- Network policies permitting HTTPS outbound connections to git
  repositories
- Container security policies enforcing SeccompProfile
  RuntimeDefault
- FIPS crypto policies configuration for TLS compliance
- `emptyDir` volume configuration with sufficient capacity for
  git repositories

### 9.3 Test Tools

- `git` client for repository setup and credential testing
- `kubectl`/`oc` for Kubernetes resource inspection and job
  management
- `curl`/`httpie` for EvalHub API endpoint testing
- Container inspection tools for verifying init container
  security posture (non-root, dropped capabilities)
- Kubernetes event and log analysis tools for debugging git
  clone operations
- Secret creation and validation tools for git credential
  management
- JSON/YAML validation tools for API request/response
  verification

---

## 10. Appendix

### 10.1 Test Case Summary

| Category | Total | P0 | P1 | P2 |
| ---------- | ------- | ---- | ---- | ----- |
| TC-API | 8 | 7 | 1 | 0 |
| TC-GIT | 6 | 5 | 1 | 0 |
| TC-SEC | 5 | 2 | 3 | 0 |
| TC-META | 4 | 4 | 0 | 0 |
| TC-REG | 3 | 0 | 3 | 0 |
| TC-NEG | 6 | 0 | 5 | 1 |
| TC-E2E | 4 | 3 | 1 | 0 |
| TC-UPGRADE | 3 | 0 | 3 | 0 |
| **Total** | **39** | **21** | **17** | **1** |

### 10.2 Endpoint Coverage

| Endpoint | Test Cases | Coverage |
| ---------- | ------------ | ---------- |
| `POST /api/v1/evaluations/jobs` | TC-API-001, TC-API-002, TC-API-003, TC-API-004, TC-API-005, TC-API-006, TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-E2E-004 | |
| `GET /api/v1/evaluations/jobs/{id}` | TC-META-001, TC-META-002, TC-META-003, TC-META-004, TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-E2E-004 | |
| `test_data_ref.git` validation | TC-API-007, TC-API-008, TC-NEG-001, TC-NEG-006, TC-E2E-001, TC-E2E-002, TC-E2E-003 | |
| git-clone init container | TC-GIT-001, TC-GIT-002, TC-GIT-003, TC-GIT-004, TC-GIT-005, TC-GIT-006, TC-E2E-001, TC-E2E-002, TC-E2E-003 | |
| Secret mounting logic | TC-SEC-001, TC-SEC-002, TC-SEC-003, TC-SEC-004, TC-SEC-005, TC-E2E-001 | |
| Job builder extension | TC-GIT-006, TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-UPGRADE-001 | |
| S3 job path validation | TC-REG-001, TC-REG-002, TC-REG-003, TC-E2E-004, TC-UPGRADE-002 | |
| Mutual exclusion validation | TC-NEG-004 | |

### 10.3 Document Change Log

| Version | Date | Changes |
| --------- | ------ | --------- |
| 1.0.0 | 2026-07-28 | Initial test plan |
| 1.1.0 | 2026-07-28 | Auto-revision: grounding citations added to Sections 3.1 and 9.1; priority definitions in Section 2.3 aligned with strategy; test repository setup instructions and example API payloads added for actionability |
| 1.2.0 | 2026-07-28 | Test cases generated: 39 TCs across 8 categories (API, GIT, SEC, META, REG, NEG, E2E, UPGRADE); Sections 5, 6, 10 updated with test case data |

---

## End of Test Plan
