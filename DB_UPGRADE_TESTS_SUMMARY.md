# TrustyAI DB Upgrade Tests - Summary

This document details the changes made to support TrustyAI Service database storage persistence testing through OpenDataHub/RHOAI upgrades.

## Changes Overview

### 1. conftest.py - Modified Fixtures for Upgrade Testing

Modified **6 fixtures** to support pre/post-upgrade test modes:

#### Pattern Used
All modified fixtures follow this pattern:
```python
if pytestconfig.option.post_upgrade:
    # Post-upgrade: reference existing resource, don't create
    resource = Resource(..., ensure_exists=True)
    yield resource
    resource.clean_up()
else:
    # Pre-upgrade: create new resource
    with Resource(..., teardown=teardown_resources) as resource:
        yield resource
```

#### Modified Fixtures

1. **`trustyai_service`** (lines 72-107)
   - Pre-upgrade: Creates new TrustyAIService with PVC or DB storage
   - Post-upgrade: References existing service, cleans up after test
   - **Impact**: Core fixture - affects all TrustyAI tests using this parametrized fixture

2. **`db_credentials_secret`** (lines 172-206)
   - Pre-upgrade: Creates secret with MariaDB credentials
   - Post-upgrade: References existing secret
   - **Impact**: Only used by DB storage tests

3. **`mariadb`** (lines 210-259)
   - Pre-upgrade: Deploys MariaDB instance with TLS
   - Post-upgrade: References existing MariaDB
   - **Impact**: Only used by DB storage tests

4. **`trustyai_db_ca_secret`** (lines 262-292)
   - Pre-upgrade: Creates CA cert secret from MariaDB
   - Post-upgrade: References existing CA secret
   - **Impact**: Only used by DB storage tests

5. **`mlserver_runtime`** (lines 295-324)
   - Pre-upgrade: Creates MLServer ServingRuntime
   - Post-upgrade: References existing runtime
   - **Impact**: Used by tests that deploy ML models

6. **`gaussian_credit_model`** (lines 327-369)
   - Pre-upgrade: Creates InferenceService
   - Post-upgrade: References existing model
   - **Impact**: Used by specific model-based tests

### 2. utils.py - No Functional Changes

- File was formatted by pre-commit hooks
- No logic changes detected

### 3. test_trustyai_service_upgrade.py - New DB Upgrade Tests

Added two new test classes for **pure DB storage** upgrade verification:

#### `TestPreUpgradeTrustyAIServiceDB` (lines 212-311)
Tests run **before** ODH/RHOAI upgrade:
- `test_trustyai_service_db_credentials_exist` - Verifies DB setup is correct
- `test_trustyai_service_db_pre_upgrade_inference` - Sends inference data
- `test_trustyai_service_db_pre_upgrade_data_upload` - Uploads training data
- `test_trustyai_service_db_pre_upgrade_drift_metric_schedule_meanshift` - Schedules metric

#### `TestPostUpgradeTrustyAIServiceDB` (lines 328-434)
Tests run **after** ODH/RHOAI upgrade:
- `test_trustyai_service_db_credentials_survived_upgrade` - Verifies DB persisted
- `test_trustyai_service_db_post_upgrade_preexisting_metric_can_be_deleted` - Deletes pre-upgrade metric
- `test_trustyai_service_db_post_upgrade_new_metric_schedule_and_cleanup` - Schedules new metric

**Namespace**: Uses `test-trustyaiservice-db-upgrade` (separate from PVC tests)

## Key Differences from Existing Tests

| Aspect | Existing Tests | New DB Tests |
|--------|---------------|--------------|
| **Storage** | PVC → DB migration | Pure DB from start |
| **Focus** | Migration process | DB persistence |
| **Namespace** | `test-trustyaiservice-upgrade` | `test-trustyaiservice-db-upgrade` |
| **Test Flow** | Deploy PVC → Migrate → Verify | Deploy DB → Upgrade → Verify |

## Impact Analysis on Other Tests

### Affected Tests by Modified Fixtures

**Low Risk** - The fixture modifications use conditional logic based on `pytestconfig.option.post_upgrade`:
- Standard tests (without `--pre-upgrade` or `--post-upgrade` flags) will follow the `else` path
- This means **existing tests continue to work exactly as before**
- Only tests explicitly marked with `@pytest.mark.pre_upgrade` or `@pytest.mark.post_upgrade` are affected

### Tests Using Modified Fixtures

Based on grep analysis, approximately **20 tests** use these fixtures:
- `tests/model_explainability/trustyai_service/drift/test_drift.py`
- `tests/model_explainability/trustyai_service/fairness/test_fairness.py`
- `tests/model_explainability/trustyai_service/service/test_trustyai_service.py`
- `tests/model_explainability/trustyai_service/upgrade/test_trustyai_service_upgrade.py`
- `tests/model_explainability/trustyai_service/service/multi_ns/test_trustyai_service_multi_ns.py`

**None of these tests will be affected** unless they explicitly use `--pre-upgrade` or `--post-upgrade` pytest options.

## How to Run

```bash
# Pre-upgrade phase (before ODH/RHOAI upgrade)
uv run pytest -m "pre_upgrade and rawdeployment" \
    tests/model_explainability/trustyai_service/upgrade/ \
    --pre-upgrade

# Perform manual ODH/RHOAI upgrade here

# Post-upgrade phase (after ODH/RHOAI upgrade)
uv run pytest -m "post_upgrade and rawdeployment" \
    tests/model_explainability/trustyai_service/upgrade/ \
    --post-upgrade
```

## Upgrade Test Approach Assessment

### Strengths ✓
1. **Isolated namespaces** - DB tests use separate namespace from PVC tests
2. **Non-destructive** - Post-upgrade mode references existing resources without recreation
3. **Comprehensive** - Tests credentials, data, metrics, and service functionality
4. **Dependency tracking** - Uses pytest dependency markers for test ordering
5. **Backward compatible** - Existing tests unaffected by fixture changes

### Considerations ⚠️
1. **Manual upgrade step** - Requires human intervention between pre/post test runs
2. **State persistence** - Relies on cluster state surviving between test runs
3. **Cleanup strategy** - Post-upgrade mode manually calls `clean_up()` instead of using context managers
4. **Resource reuse** - Uses `ensure_exists=True` which assumes resources exist exactly as created

### Recommended Best Practices
- Run pre-upgrade tests to completion before upgrading
- Document upgrade procedure between test phases
- Verify cluster state before running post-upgrade tests
- Consider adding health checks for resource existence in post-upgrade tests
