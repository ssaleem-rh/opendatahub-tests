import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HTTP_METRIC_NAMES,
)
from tests.ai_safety.evalhub.utils import fetch_evalhub_metrics
from utilities.guardrails import get_auth_headers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-metrics"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMetrics:
    """Tests for the EvalHub Prometheus metrics endpoint."""

    def test_evalhub_metrics_endpoint(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_metrics_service: Service,
    ) -> None:
        """Given: a running EvalHub instance that has served a request.
        When: the cluster-internal /metrics endpoint on the metrics Service is scraped.
        Then: it returns Prometheus text including the HTTP request, duration and in-flight metrics.

        The metrics endpoint is on the cluster-internal port 8081 with no Route, so it is
        reached through the pod backing the metrics Service (port-forward, or Service DNS
        when the tests run inside the cluster).
        """
        health_resp = requests.get(
            url=f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}",
            headers=get_auth_headers(token=current_client_token),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200, f"Expected 200 from health endpoint, got {health_resp.status_code}"

        body = fetch_evalhub_metrics(admin_client=admin_client, metrics_service=evalhub_metrics_service)
        for metric_names in EVALHUB_HTTP_METRIC_NAMES:
            assert any(name in body for name in metric_names), (
                f"None of the metric names {metric_names} found in /metrics response"
            )

    def test_evalhub_metrics_recorded_for_requests(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_metrics_service: Service,
    ) -> None:
        """Given: a running EvalHub instance with metrics service.
        When: GET /api/v1/health is called, then /metrics is scraped.
        Then: /metrics contains a request count for the health path.
        """
        headers = get_auth_headers(token=current_client_token)

        # Hit the health endpoint through the Route to generate a metric entry
        health_url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        health_resp = requests.get(
            url=health_url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200

        # Scrape metrics from the internal service and verify the health path appears
        metrics_text = fetch_evalhub_metrics(admin_client=admin_client, metrics_service=evalhub_metrics_service)
        assert EVALHUB_HEALTH_PATH in metrics_text, (
            f"Expected request count for '{EVALHUB_HEALTH_PATH}' in /metrics output"
        )
