from typing import Any

import pytest
import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_APP_LABEL,
    EVALHUB_COMPONENT_LABEL,
    EVALHUB_HEALTH_PATH,
    EVALHUB_METRICS_COMPONENT_LABEL,
    EVALHUB_METRICS_PORT,
    EVALHUB_METRICS_QUERY_TIMEOUT,
    EVALHUB_METRICS_SERVICE_SUFFIX,
    EVALHUB_REQUEST_COUNT_METRIC_REGEX,
    EVALHUB_SCRAPE_INTERVAL,
)
from utilities.guardrails import get_auth_headers
from utilities.monitoring import validate_metrics_field

LOGGER = structlog.get_logger(name=__name__)

USER_WORKLOAD_MONITORING_NAMESPACE: str = "openshift-user-workload-monitoring"


def query_prometheus_result(prometheus: Prometheus, query: str) -> list[dict[str, Any]]:
    """Run an instant PromQL query and return the result series.

    Args:
        prometheus: Prometheus client pointing at the Thanos Querier.
        query: PromQL expression.

    Returns:
        List of result series, or an empty list when the query fails or returns nothing.
    """
    try:
        return list(prometheus.query(query=query).get("data", {}).get("result", []))
    except (requests.exceptions.RequestException, ValueError) as exc:
        LOGGER.warning(f"Prometheus query '{query}' failed: {exc}")
        return []


def describe_user_workload_monitoring(admin_client: DynamicClient, namespace: Namespace) -> list[str]:
    """Describe whether user-workload monitoring is able to scrape the given namespace.

    Args:
        admin_client: Cluster client.
        namespace: Namespace holding the EvalHub ServiceMonitor.

    Returns:
        Human-readable findings, one per line.
    """
    findings: list[str] = []

    monitoring_config = ConfigMap(
        client=admin_client, name="cluster-monitoring-config", namespace="openshift-monitoring"
    )
    if monitoring_config.exists:
        config_yaml = yaml.safe_load(monitoring_config.instance.data.get("config.yaml", "")) or {}
        findings.append(f"cluster-monitoring-config enableUserWorkload={config_yaml.get('enableUserWorkload')}")
    else:
        findings.append(
            "ConfigMap openshift-monitoring/cluster-monitoring-config not found: user-workload monitoring is disabled"
        )

    uwm_pods = [
        f"{pod.name}={pod.instance.status.phase}"
        for pod in Pod.get(client=admin_client, namespace=USER_WORKLOAD_MONITORING_NAMESPACE)
    ]
    findings.append(f"Pods in {USER_WORKLOAD_MONITORING_NAMESPACE}: {uwm_pods or 'none'}")

    namespace_labels = dict(namespace.instance.metadata.labels or {})
    findings.append(
        f"Namespace {namespace.name} label openshift.io/cluster-monitoring="
        f"{namespace_labels.get('openshift.io/cluster-monitoring')} (must not be 'true' for user-workload scraping)"
    )
    return findings


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubServiceMonitor:
    """Tests for the EvalHub ServiceMonitor and metrics Service resources."""

    def test_servicemonitor_exists(
        self,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """Verify the operator created a ServiceMonitor with correct owner reference."""
        assert evalhub_service_monitor.exists, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' not found in namespace "
            f"'{evalhub_service_monitor.namespace}'"
        )

        owner_refs = evalhub_service_monitor.instance.metadata.ownerReferences
        assert owner_refs, "ServiceMonitor has no ownerReferences"

        cr_uid = evalhub_cr.instance.metadata.uid
        owner_uids = [ref.uid for ref in owner_refs]
        assert cr_uid in owner_uids, (
            f"Expected ownerReference UID '{cr_uid}' (EvalHub CR) not found in "
            f"ServiceMonitor ownerReferences: {owner_uids}"
        )

    def test_servicemonitor_spec(
        self,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """Validate the ServiceMonitor spec matches the operator's expected configuration."""
        spec = evalhub_service_monitor.instance.spec

        endpoints = spec.endpoints
        assert len(endpoints) == 1, f"Expected 1 endpoint, found {len(endpoints)}"

        ep = endpoints[0]
        assert ep.path == "/metrics", f"Expected endpoint path '/metrics', got '{ep.path}'"
        assert ep.port == "metrics", f"Expected endpoint port 'metrics', got '{ep.port}'"
        assert ep.scheme == "http", f"Expected scheme 'http', got '{ep.scheme}'"
        assert str(ep.interval) == EVALHUB_SCRAPE_INTERVAL, (
            f"Expected scrape interval '{EVALHUB_SCRAPE_INTERVAL}', got '{ep.interval}'"
        )
        assert ep.honorLabels is True, f"Expected honorLabels=true, got {ep.honorLabels}"

        expected_labels = {
            "app": EVALHUB_APP_LABEL,
            "instance": evalhub_cr.name,
            "component": EVALHUB_METRICS_COMPONENT_LABEL,
        }
        selector_labels = dict(spec.selector.matchLabels)
        assert selector_labels == expected_labels, (
            f"Expected selector matchLabels {expected_labels}, got {selector_labels}"
        )

        ns_selector = spec.namespaceSelector
        assert model_namespace.name in ns_selector.matchNames, (
            f"Expected namespace '{model_namespace.name}' in namespaceSelector.matchNames, got {ns_selector.matchNames}"
        )

    def test_metrics_service_exists(
        self,
        evalhub_cr: EvalHub,
        evalhub_metrics_service: Service,
    ) -> None:
        """Verify the metrics Service exists with correct spec."""
        assert evalhub_metrics_service.exists, f"Metrics Service '{evalhub_metrics_service.name}' not found"

        spec = evalhub_metrics_service.instance.spec
        assert spec.type == "ClusterIP", f"Expected Service type 'ClusterIP', got '{spec.type}'"

        ports = spec.ports
        assert len(ports) >= 1, "Expected at least 1 port on the metrics Service"

        metrics_port = next((p for p in ports if p.name == "metrics"), None)
        assert metrics_port is not None, f"Expected port named 'metrics', found: {[p.name for p in ports]}"
        assert metrics_port.port == EVALHUB_METRICS_PORT, (
            f"Expected port {EVALHUB_METRICS_PORT}, got {metrics_port.port}"
        )

        selector = dict(spec.selector)
        assert selector.get("app") == EVALHUB_APP_LABEL, (
            f"Expected selector app='{EVALHUB_APP_LABEL}', got '{selector.get('app')}'"
        )
        assert selector.get("instance") == evalhub_cr.name, (
            f"Expected selector instance='{evalhub_cr.name}', got '{selector.get('instance')}'"
        )
        assert selector.get("component") == EVALHUB_COMPONENT_LABEL, (
            f"Expected selector component='{EVALHUB_COMPONENT_LABEL}', got '{selector.get('component')}'"
        )

    def test_prometheus_target_up(
        self,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
        prometheus: Prometheus,
    ) -> None:
        """Verify EvalHub appears as an active scrape target (UP) in Prometheus."""
        sm_name = f"{evalhub_cr.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=f'up{{job="{sm_name}"}}',
            expected_value="1",
            timeout=120,
        )

    def test_metrics_queryable_via_thanos(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
        prometheus: Prometheus,
    ) -> None:
        """Given: a running EvalHub with a ServiceMonitor that has served an API request.
        When: the HTTP request counter is queried through the Thanos Querier.
        Then: the series for the EvalHub metrics job is returned with a value of at least 0.
        """
        job_name = f"{evalhub_cr.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"

        # Request counters only get a series after the first request, so generate one
        health_resp = requests.get(
            url=f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}",
            headers=get_auth_headers(token=current_client_token),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200, f"Expected 200 from health endpoint, got {health_resp.status_code}"

        LOGGER.info(f"ServiceMonitor '{job_name}' spec: {evalhub_service_monitor.instance.spec.to_dict()}")
        monitoring_findings = describe_user_workload_monitoring(admin_client=admin_client, namespace=model_namespace)
        for finding in monitoring_findings:
            LOGGER.info(finding)

        request_count_query = f'{{__name__=~"{EVALHUB_REQUEST_COUNT_METRIC_REGEX}",job="{job_name}"}}'
        try:
            validate_metrics_field(
                prometheus=prometheus,
                metrics_query=request_count_query,
                expected_value="0",
                greater_than=True,
                timeout=EVALHUB_METRICS_QUERY_TIMEOUT,
            )
        except TimeoutExpiredError:
            up_series = query_prometheus_result(prometheus=prometheus, query=f'up{{job="{job_name}"}}')
            namespace_up_series = query_prometheus_result(
                prometheus=prometheus, query=f'up{{namespace="{model_namespace.name}"}}'
            )
            scraped_names = query_prometheus_result(
                prometheus=prometheus, query=f'count by (__name__) ({{job="{job_name}"}})'
            )
            pytest.fail(
                f"No series matched '{request_count_query}' within {EVALHUB_METRICS_QUERY_TIMEOUT}s. "
                f"up for job '{job_name}': {up_series}; "
                f"up in namespace '{model_namespace.name}': {namespace_up_series}; "
                f"metric names scraped for the job: {[item['metric'].get('__name__') for item in scraped_names]}; "
                f"monitoring state: {monitoring_findings}. "
                "An empty 'up' result means user-workload Prometheus has not discovered the target; "
                "scraped names without a request counter mean eval-hub exposes a different metric name."
            )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-sm-cleanup"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubServiceMonitorCleanup:
    """Tests for ServiceMonitor garbage collection on EvalHub CR deletion."""

    def test_servicemonitor_cleanup(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify ServiceMonitor is garbage-collected when EvalHub CR is deleted."""
        with EvalHub(
            client=admin_client,
            name="evalhub-cleanup",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ) as evalhub:
            deployment = Deployment(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=300)

            sm_name = f"{evalhub.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"
            sm = ServiceMonitor(
                client=admin_client,
                name=sm_name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert sm.exists, f"ServiceMonitor '{sm_name}' should exist before CR deletion"

        try:
            for sample in TimeoutSampler(
                wait_timeout=60,
                sleep=5,
                func=lambda: (
                    not ServiceMonitor(
                        client=admin_client,
                        name=sm_name,
                        namespace=model_namespace.name,
                    ).exists
                ),
            ):
                if sample:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"ServiceMonitor '{sm_name}' was not garbage-collected within 60s after EvalHub CR deletion")
