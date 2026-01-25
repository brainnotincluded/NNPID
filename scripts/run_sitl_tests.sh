#!/bin/bash
# SITL Integration Test Runner
# 
# This script runs all SITL integration tests, including:
# 1. Unit tests (no PX4 required)
# 2. Integration tests with mock SITL
# 3. Full SITL tests with PX4 (if available)
#
# Usage:
#   ./scripts/run_sitl_tests.sh          # Run all tests
#   ./scripts/run_sitl_tests.sh unit     # Only unit tests
#   ./scripts/run_sitl_tests.sh mock     # With mock SITL
#   ./scripts/run_sitl_tests.sh full     # Full PX4 tests (requires PX4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test mode
MODE="${1:-all}"

echo -e "${BLUE}"
echo "============================================================"
echo "     NNPID SITL Integration Test Suite"
echo "============================================================"
echo -e "${NC}"

cd "$PROJECT_ROOT"

# Check Python environment
echo -e "${YELLOW}Checking environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Ensure dependencies
echo "Installing dependencies..."
pip install -q -e ".[dev]" 2>/dev/null || true

# ============================================================================
# Unit Tests
# ============================================================================

run_unit_tests() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  UNIT TESTS (No PX4 Required)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    # Run pytest for all test files
    echo -e "${YELLOW}Running pytest...${NC}"
    python -m pytest tests/ -v --tb=short \
        --ignore=tests/test_sitl_integration.py \
        2>&1 | tail -50
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "\n${GREEN}✓ Unit tests passed${NC}"
        return 0
    else
        echo -e "\n${RED}✗ Unit tests failed${NC}"
        return 1
    fi
}

# ============================================================================
# Integration Tests with Mock SITL
# ============================================================================

run_mock_tests() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  INTEGRATION TESTS (Mock SITL)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    echo -e "${YELLOW}Running SITL integration tests...${NC}"
    python -m pytest tests/test_sitl_integration.py -v --tb=short 2>&1 | tail -100
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "\n${GREEN}✓ Integration tests passed${NC}"
        return 0
    else
        echo -e "\n${RED}✗ Integration tests failed${NC}"
        return 1
    fi
}

# ============================================================================
# Full SITL Tests (Requires PX4)
# ============================================================================

check_px4() {
    # Check if PX4 SITL is available
    if [ -n "$PX4_HOME" ] && [ -d "$PX4_HOME" ]; then
        return 0
    fi
    
    if [ -d "$HOME/PX4-Autopilot" ]; then
        export PX4_HOME="$HOME/PX4-Autopilot"
        return 0
    fi
    
    return 1
}

start_px4() {
    echo -e "${YELLOW}Starting PX4 SITL...${NC}"
    
    if ! check_px4; then
        echo -e "${RED}PX4 not found. Set PX4_HOME or install PX4-Autopilot${NC}"
        return 1
    fi
    
    cd "$PX4_HOME"
    
    # Start PX4 SITL in background
    make px4_sitl none_iris HEADLESS=1 &
    PX4_PID=$!
    
    # Wait for PX4 to start
    echo "Waiting for PX4 to start..."
    sleep 10
    
    cd "$PROJECT_ROOT"
    
    return 0
}

stop_px4() {
    if [ -n "$PX4_PID" ]; then
        echo -e "${YELLOW}Stopping PX4...${NC}"
        kill $PX4_PID 2>/dev/null || true
        wait $PX4_PID 2>/dev/null || true
    fi
}

run_full_sitl_tests() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  FULL SITL TESTS (Requires PX4)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    if ! check_px4; then
        echo -e "${YELLOW}Skipping full SITL tests - PX4 not installed${NC}"
        echo "To run full tests, install PX4-Autopilot:"
        echo "  git clone https://github.com/PX4/PX4-Autopilot.git"
        echo "  cd PX4-Autopilot && make px4_sitl"
        return 0
    fi
    
    # Start PX4
    start_px4 || return 1
    
    # Trap to ensure PX4 is stopped
    trap stop_px4 EXIT
    
    # Run manual tests
    echo -e "\n${YELLOW}Running SITL flight tests...${NC}"
    
    # Test takeoff
    timeout 60 python scripts/test_sitl.py takeoff || {
        echo -e "${RED}Takeoff test failed${NC}"
        return 1
    }
    
    echo -e "\n${GREEN}✓ Full SITL tests passed${NC}"
    
    # Stop PX4
    stop_px4
    trap - EXIT
    
    return 0
}

# ============================================================================
# Quick Verification Tests
# ============================================================================

run_quick_verify() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  QUICK VERIFICATION${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    echo -e "${YELLOW}Testing imports...${NC}"
    python -c "
from src.communication.mavlink_bridge import MAVLinkBridge
from src.communication.messages import SetPositionTargetLocalNED, CommandLong
from src.controllers.offboard_controller import OffboardController
from src.environments.setpoint_env import SetpointHoverEnv
from src.environments.sitl_env import SITLEnv
print('✓ All imports successful')
"
    
    echo -e "\n${YELLOW}Testing environment...${NC}"
    python -c "
from src.environments.setpoint_env import SetpointHoverEnv
import numpy as np

env = SetpointHoverEnv()
obs, info = env.reset(seed=42)
print(f'✓ Environment created, obs shape: {obs.shape}')

for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

print(f'✓ Environment stepped successfully')
env.close()
"
    
    echo -e "\n${GREEN}✓ Quick verification passed${NC}"
}

# ============================================================================
# Main
# ============================================================================

FAILED=0

case "$MODE" in
    "unit")
        run_unit_tests || FAILED=1
        ;;
    "mock")
        run_mock_tests || FAILED=1
        ;;
    "full")
        run_full_sitl_tests || FAILED=1
        ;;
    "quick")
        run_quick_verify || FAILED=1
        ;;
    "all"|*)
        run_quick_verify || FAILED=1
        run_unit_tests || FAILED=1
        run_mock_tests || FAILED=1
        
        if check_px4; then
            run_full_sitl_tests || FAILED=1
        else
            echo -e "\n${YELLOW}Note: Full SITL tests skipped (PX4 not installed)${NC}"
        fi
        ;;
esac

# Summary
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  ALL TESTS PASSED${NC}"
else
    echo -e "${RED}  SOME TESTS FAILED${NC}"
fi
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit $FAILED
