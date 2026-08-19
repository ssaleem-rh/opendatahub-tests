#!/bin/bash
set -e
export KUBECONFIG=~/.kube/configA
echo "Running uv sync..."
uv sync
echo "Running test against cluster-A (KUBECONFIG=$KUBECONFIG): $@"
KUBECONFIG=~/.kube/configA uv run pytest "$@"
