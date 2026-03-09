#!/bin/bash
set -e
cd /app/workspace
export ANTHROPIC_BASE_URL="http://${PROXY_HOST:-localhost}:${PROXY_PORT:-8080}"
echo "=== Agent starting ==="
echo "  Proxy: $ANTHROPIC_BASE_URL"
echo "  Workspace: $(pwd)"
echo "  GPU: $CUDA_DEVICE"
echo "  Max turns: ${MAX_TURNS:-50}"
echo "--- CUDA check ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi: not available"
python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.cuda.device_count():', torch.cuda.device_count())" 2>/dev/null || echo "torch CUDA check failed"
echo "------------------"
mkdir -p /app/events
claude \
    --print \
    --verbose \
    --dangerously-skip-permissions \
    --no-session-persistence \
    --max-turns "${MAX_TURNS:-50}" \
    --output-format stream-json \
    --include-partial-messages \
    -p "$(cat /app/CLAUDE.md)" 2>&1 | tee /app/events/events.jsonl
exit ${PIPESTATUS[0]}
