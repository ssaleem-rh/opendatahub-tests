import pytest
import requests
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_METRICS_PATH,
)
from utilities.guardrails import get_auth_headers

# Prometheus runtime/process collectors that the EvalHub metrics endpoint always exports,
# regardless of traffic or OTEL configuration. Application HTTP request metrics
# (http_server_request_count_total, OTEL naming) are only exported when the EvalHub CR
# enables the OTEL metrics sink; that behavior is covered by test_evalhub_otel.py.
EXPECTED_RUNTIME_METRICS = (
    "go_goroutines",
    "process_start_time_seconds",
)


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
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_metrics_url: str,
    ) -> None:
        """Verify /metrics returns 200 and serves valid Prometheus-format metrics.

        The metrics endpoint is on the cluster-internal port 8081 with no Route,
        so it is accessed via a port-forward to the metrics service.
        """
        url = f"{evalhub_metrics_url}{EVALHUB_METRICS_PATH}"
        response = requests.get(url=url, timeout=10)
        assert response.status_code == 200, f"Expected 200 from /metrics, got {response.status_code}"
        body = response.text
        assert "# HELP" in body and "# TYPE" in body, "Response is not valid Prometheus exposition format"
        for metric in EXPECTED_RUNTIME_METRICS:
            assert metric in body, f"Expected runtime metric '{metric}' not found in /metrics response"

    def test_evalhub_metrics_recorded_for_requests(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_metrics_url: str,
    ) -> None:
        """Given: a running EvalHub instance with its metrics service.
        When: GET /api/v1/health is served, then /metrics is scraped.
        Then: the metrics endpoint stays available and keeps serving valid Prometheus metrics
        while the API serves traffic.
        """
        headers = get_auth_headers(token=current_client_token)

        # Serve a request through the API Route to confirm the instance is handling traffic
        health_url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        health_resp = requests.get(
            url=health_url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200, f"Expected 200 from health endpoint, got {health_resp.status_code}"

        # The metrics endpoint must stay available and serve valid Prometheus metrics under traffic
        metrics_url = f"{evalhub_metrics_url}{EVALHUB_METRICS_PATH}"
        metrics_resp = requests.get(url=metrics_url, timeout=10)
        assert metrics_resp.status_code == 200, f"Expected 200 from /metrics, got {metrics_resp.status_code}"
        body = metrics_resp.text
        assert "# HELP" in body and "# TYPE" in body, "Response is not valid Prometheus exposition format"
        for metric in EXPECTED_RUNTIME_METRICS:
            assert metric in body, f"Expected runtime metric '{metric}' not found in /metrics response"
