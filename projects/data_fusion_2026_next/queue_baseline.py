#!/usr/bin/env python3
"""Queue baseline experiment for the orchestrator service via API."""

import requests
import sys
from pathlib import Path

def main():
    # The dashboard should be running on port 8090
    api_url = "http://localhost:8090/api/launch"
    
    response = requests.post(api_url, json={
        "name": "baseline_optimized",
        "prompt": "Run optimized baseline Deep Residual MLP with knowledge distillation. "
                  "Features: GPU tensors, torch.compile, fused AdamW, AMP, no CPU sync. "
                  "This is the baseline for comparison.",
        "base_experiment": "default",
        "task_type": "experiment",
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"Queued experiment: {data['id']}")
        print(f"Status: {data['status']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

if __name__ == "__main__":
    main()
