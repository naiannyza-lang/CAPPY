#!/bin/bash
# =============================================================================
# CAPPY Dual-Board Launcher
# Launches both ATS-9352 (v1.3) and ATS-9462 (v1.0) GUIs simultaneously
# Usage: ./launch_cappy.sh [--capture-only]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
GREEN='\033[0;32m'
PINK='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     CAPPY Dual-Board Launcher            ║${NC}"
echo -e "${CYAN}║     ATS-9352 (v1.3) + ATS-9462 (v1.0)   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"

# Check that both scripts exist
V13="CAPPY_v1_3.py"
V10="CAPPY_v1_0.py"

if [ ! -f "$V13" ]; then
    echo -e "${PINK}ERROR: $V13 not found in $SCRIPT_DIR${NC}"
    exit 1
fi
if [ ! -f "$V10" ]; then
    echo -e "${PINK}ERROR: $V10 not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Ensure config files exist (auto-create defaults)
V13_CFG="CAPPY_v1_3.yaml"
V10_CFG="CAPPY_ATS9462_v1_3.yaml"

if [ ! -f "$V13_CFG" ]; then
    echo -e "${GREEN}[v1.3] Creating default config: $V13_CFG${NC}"
    python3 "$V13" init 2>/dev/null
fi
if [ ! -f "$V10_CFG" ]; then
    echo -e "${GREEN}[v1.0] Creating default config: $V10_CFG${NC}"
    python3 "$V10" init 2>/dev/null
fi

# Ensure data directories exist
mkdir -p dataFile
mkdir -p dataFile_ATS9462

# Track child PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo -e "${CYAN}[Launcher] Shutting down...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${CYAN}[Launcher] Stopping PID $pid${NC}"
            kill -INT "$pid" 2>/dev/null
        fi
    done
    # Wait briefly then force-kill stragglers
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${PINK}[Launcher] Force-killing PID $pid${NC}"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    echo -e "${CYAN}[Launcher] Done.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

if [ "$1" == "--capture-only" ]; then
    # Headless mode: run capture subprocesses directly (no GUI)
    echo -e "${GREEN}[v1.3] Starting capture (ATS-9352)...${NC}"
    python3 "$V13" capture --config "$V13_CFG" &
    PIDS+=($!)
    echo -e "  PID: ${PIDS[-1]}"

    echo -e "${GREEN}[v1.0] Starting capture (ATS-9462)...${NC}"
    python3 "$V10" capture --config "$V10_CFG" &
    PIDS+=($!)
    echo -e "  PID: ${PIDS[-1]}"

    echo ""
    echo -e "${CYAN}Both captures running. Press Ctrl+C to stop both.${NC}"
    echo ""

    # Wait for both
    wait
else
    # GUI mode: launch both GUIs
    echo -e "${GREEN}[v1.3] Launching GUI (ATS-9352)...${NC}"
    python3 "$V13" gui &
    PIDS+=($!)
    echo -e "  PID: ${PIDS[-1]}"

    # Small delay so the two Tk instances don't collide on init
    sleep 1

    echo -e "${GREEN}[v1.0] Launching GUI (ATS-9462)...${NC}"
    python3 "$V10" gui &
    PIDS+=($!)
    echo -e "  PID: ${PIDS[-1]}"

    echo ""
    echo -e "${CYAN}Both GUIs running. Press Ctrl+C or close both windows to stop.${NC}"
    echo ""

    # Wait for both
    wait
fi
