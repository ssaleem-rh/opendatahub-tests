---
feature: evalhub_git_storage_source
source_key: RHAISTRAT-2058
score: 9
pass: true
verdict: Ready
scores:
  specificity: 2
  grounding: 2
  scope_fidelity: 2
  actionability: 1
  consistency: 2
last_updated: '2026-07-28'
auto_revised: true
before_score: 7
before_scores:
  specificity: 2
  grounding: 1
  scope_fidelity: 2
  actionability: 1
  consistency: 1
error: null
---

# Test Plan Review

## Rubric Scores

| Criterion | Score | Notes |
| ----------- | ------- | ------- |
| Specificity | 2/2 | Priorities reference feature-specific scenarios (git-clone init container, credential isolation, commit SHA recording). Risks name specific dependencies and failure modes unique to this feature. Test levels justified by interface types. Section 2.3 priority definitions map directly to strategy requirement tiers. Section 8 risks are not boilerplate. |
| Grounding | 2/2 | All Section 4 entries traceable to strategy. Dependencies cited with parenthetical strategy quotes. API payloads derived from strategy fields with explicit TBD caveat for final field names. One minor extrapolation: air-gap risk in Section 8 not in strategy risks, but reasonable inference. |
| Scope Fidelity | 2/2 | All 7 strategy requirements map to test objectives. All 4 out-of-scope items listed verbatim. Strategy NFRs covered in Section 7. Sub-path cloning covered in objective 7. No orphans in either direction. Strategy acceptance criteria (5 items) each map to at least one test objective or Section 4 entry. |
| Actionability | 1/2 | EvalHub v0.4.0+ specified, OpenShift version TBD with rationale. Detailed test repo setup instructions with kubectl command. Three defined test user roles. Example API payloads provided. Gaps: ServiceAccount Role/RoleBinding names TBD, final API field names TBD, 2-3 clarifying questions remain. |
| Consistency | 2/2 | All cross-references align. Section 4 priorities match Section 2.3 definitions. Section 10.2 lists all 8 Section 4 entries. Section 7 NFR categories appropriate. Section 6 has pre-create-cases placeholder. Consistent terminology throughout. |

### Total: 9/10 -- Verdict: Ready

## Grounding Cross-Reference

| Section 4 Entry | Source Match | Status |
| ----------------- | ------------- | -------- |
| `POST /api/v1/evaluations/jobs` (P0) | "The test_data_ref field in the job submission API (POST /api/v1/evaluations/jobs) gains a git key" | GROUNDED |
| `GET /api/v1/evaluations/jobs/{id}` (P0) | "visible in GET /api/v1/evaluations/jobs/{id} as git_commit_sha" | GROUNDED |
| `test_data_ref.git` validation (P0) | "The API validates that exactly one storage source is specified per job" | GROUNDED |
| git-clone init container (P0) | "extended to construct a git-clone init container when the git storage source is specified" | GROUNDED |
| Secret mounting logic (P0) | "The Secret is mounted exclusively in the git-clone init container" | GROUNDED |
| Job builder extension (P1) | "The Kubernetes Job builder in internal/eval_hub/runtimes/k8s/job_builders.go is extended" | GROUNDED |
| S3 job path validation (P1) | "[P1] Existing S3 and PVC job paths remain unaffected (backward compatibility)" | GROUNDED |
| Mutual exclusion validation (P2) | "[P2] API validation rejects jobs specifying both s3_test_data_ref and git simultaneously" | GROUNDED |

## Section-by-Section Feedback

### Actionability (scored 1/2)

The plan provides solid environmental details (EvalHub v0.4.0+, test repo setup with kubectl commands, three test user roles, example API payloads) but has remaining gaps that prevent a score of 2:

1. **ServiceAccount RBAC details**: The ServiceAccount Role and RoleBinding names used for git-clone init container permissions are marked TBD. These should be specified once the implementation stabilizes, as tests will need to validate RBAC configuration.

2. **Final API field names**: While the plan correctly notes that field names are TBD pending final implementation, test authors will need confirmed field names (e.g., whether the response field is `git_commit_sha` or `gitCommitSha`) before writing assertions.

3. **Remaining clarifying questions**: Approximately 2-3 open questions remain around edge cases (e.g., behavior when a git ref resolves to a non-existent commit, timeout values for large repository clones). Resolving these would make the plan fully actionable without requiring test authors to consult additional sources.

These gaps are understandable given the feature's implementation stage and do not block test case generation, but they prevent full actionability.

## Revision History

Initial assessment

### Auto-Revision (v1.1.0)

**Grounding (Sections 3.1, 9.1)**:

- Added inline citations to Section 3.1 grounding "EvalHub API v0.4.0+" to the strategy's Dependencies section verbatim text.
- Added inline citation to Section 3.1 grounding "RHAISTRAT-2056" to the strategy's Dependencies section verbatim text.
- Added inline citation to Section 3.1 grounding UBI 9 minimal image to the strategy's Dependencies section.
- Added inline citations to Section 9.1 grounding "EvalHub API v0.4.0+", "RHAISTRAT-2056", and UBI 9 image to the strategy's Dependencies section.

**Actionability (Sections 3.1, 3.2, 3.3, 4)**:

- Section 3.1: Changed OpenShift version from bare "TBD" to "TBD -- pending release planning" with note that the strategy does not specify a minimum version.
- Section 3.2: Added "Test Repository Setup" subsection with concrete instructions for creating public, private, and large test repositories, including kubectl Secret creation command.
- Section 3.2: Added inline citation for credential format from strategy's risk mitigation text.
- Section 3.3: Referenced strategy Assumptions for ServiceAccount RBAC and noted specific Role/RoleBinding names are TBD -- pending implementation documentation.
- Section 3.3: Added strategy citation for namespace-scoped Secret requirement.
- Section 4: Added new Section 4.1 "Example API Payloads" with sample POST request bodies (private and public repos) and sample GET response showing git_commit_sha, all derived from strategy technical approach text.

**Consistency (Sections 2.3, 4)**:

- Section 2.3 P0: Narrowed from "clone operations succeed for public/private repos" to "clone operations succeed for private repos with credentials" to align with strategy's P1 assignment for public repos.
- Section 2.3 P0: Removed "existing S3 storage source remains functional" to align with strategy's P1 assignment for backward compatibility.
- Section 2.3 P1: Added "Public repository cloning without requiring a secret_ref" and "existing S3 and PVC job paths remain unaffected (backward compatibility)" to match strategy priority assignments.
- Section 4 endpoint priorities were already correct and required no changes.

### Cycle 1 Revision

- **Grounding**: Added inline strategy citations in Sections 3.1 and 9.1
- **Actionability**: Added test repository setup instructions, example API payloads, ServiceAccount RBAC context
- **Consistency**: Aligned Section 2.3 P0/P1 definitions with strategy priority tiers
- **Specificity**: N/A -- scored 2
- **Scope Fidelity**: N/A -- scored 2
