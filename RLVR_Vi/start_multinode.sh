#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# Multi-Node Ray Cluster Startup Script for RLVR Training
# ============================================================================
# Usage:
#   # On Head Node (Node 0):
#   NODE_RANK=0 NNODES=2 bash start_multinode.sh
#
#   # On Worker Node (Node 1, 2, ...):
#   NODE_RANK=1 NNODES=2 HEAD_IP=<head_node_ip> bash start_multinode.sh
# ============================================================================

# Configuration
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-2}
HEAD_IP=${HEAD_IP:-"127.0.0.1"}
RAY_PORT=${RAY_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Node Configuration:"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  NNODES: ${NNODES}"
echo "  HEAD_IP: ${HEAD_IP}"
echo "  RAY_PORT: ${RAY_PORT}"
echo "============================================"

if [ "${NODE_RANK}" == "0" ]; then
    # ========== HEAD NODE ==========
    echo "Starting Ray HEAD node..."
    
    # Check if Ray is already running
    if ray status &>/dev/null; then
        echo "Ray is already running. Stopping existing cluster..."
        ray stop --force
        sleep 3
    fi
    
    # Start Ray Head
    ray start --head \
        --port=${RAY_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --disable-usage-stats
    
    echo ""
    echo "============================================"
    echo "Ray HEAD started successfully!"
    echo ""
    echo "Worker nodes should run:"
    echo "  NODE_RANK=<rank> HEAD_IP=$(hostname -I | awk '{print $1}') bash start_multinode.sh"
    echo ""
    echo "Dashboard available at:"
    echo "  http://$(hostname -I | awk '{print $1}'):${DASHBOARD_PORT}"
    echo "============================================"
    
    # Wait for all worker nodes to join
    echo ""
    echo "Waiting for all ${NNODES} nodes to join the cluster..."
    
    while true; do
        node_count=$(ray nodes 2>/dev/null | grep -c "node:" || echo "0")
        echo "  Current nodes: ${node_count}/${NNODES}"
        
        if [ "${node_count}" -ge "${NNODES}" ]; then
            echo "All nodes have joined!"
            break
        fi
        sleep 5
    done
    
    # Show cluster status
    echo ""
    echo "Cluster Status:"
    ray status
    
    # Ask user whether to start training
    echo ""
    echo "============================================"
    echo "Ready to start training!"
    echo ""
    echo "To start training, run:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  bash train_grpo.sh"
    echo ""
    echo "Or submit as a Ray job:"
    echo "  ray job submit --address=\"http://localhost:${DASHBOARD_PORT}\" -- bash ${SCRIPT_DIR}/train_grpo.sh"
    echo "============================================"
    
else
    # ========== WORKER NODE ==========
    echo "Starting Ray WORKER node..."
    
    # Check if Ray is already running
    if ray status &>/dev/null; then
        echo "Ray is already running. Stopping..."
        ray stop --force
        sleep 3
    fi
    
    # Wait a bit for head node to be ready
    sleep 5
    
    # Start Ray Worker
    ray start --address="${HEAD_IP}:${RAY_PORT}" --disable-usage-stats
    
    echo ""
    echo "============================================"
    echo "Ray WORKER started successfully!"
    echo "Connected to HEAD at: ${HEAD_IP}:${RAY_PORT}"
    echo ""
    echo "This node will wait for tasks from the HEAD node."
    echo "============================================"
fi
