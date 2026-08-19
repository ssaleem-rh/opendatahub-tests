---
feature: evalhub_git_storage_source
source_key: RHAISTRAT-2058
status: Open
gap_count: 12
last_updated: '2026-07-28'
---
# Gaps — EvalHub Git Storage Source

## Scope & Endpoints

- Git credential Secret format and structure requirements not
  specified — would be resolved by: API spec / design doc
- Init container implementation details (new binary vs extension
  of eval-runtime-init) — would be resolved by: ADR / design doc
- RBAC permission requirements for EvalHub ServiceAccount Secret
  access — would be resolved by: design doc / security review
- Maximum repository size limits and timeout configurations not
  defined — would be resolved by: design doc / performance
  requirements
- Specific API schema structure for `test_data_ref.git` fields
  not specified — would be resolved by: API spec

## Test Strategy & Risks

- Git repository size limits and performance characteristics not
  specified — would be resolved by: performance testing
  specification or design doc
- Shallow clone behavior and ref resolution logic not defined —
  would be resolved by: ADR defining git clone implementation
  details
- Error handling and retry mechanisms for git operations not
  specified — would be resolved by: design doc covering failure
  scenarios

## Environment & Infrastructure

- OpenShift and RHOAI version requirements not specified —
  would be resolved by: ADR / feature refinement
- Configurable clone timeout values and defaults not defined —
  would be resolved by: ADR / feature refinement
- EmptyDir volume size requirements for typical evaluation
  datasets not specified — would be resolved by: feature
  refinement / design doc

## Test Case Coverage Gaps

No coverage gaps found. All endpoints from Section 4 are covered
by at least one test case, all P0 endpoints have E2E coverage,
and all test objectives from Section 1.3 are addressed. No test
cases were created for areas flagged as pending in the gaps above.

Note: The `Mutual exclusion validation` endpoint (P2) does not
have dedicated E2E coverage, which is acceptable given its P2
priority — it is covered by the unit-level TC-NEG-004.
