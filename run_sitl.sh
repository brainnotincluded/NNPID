#!/bin/bash
# Launch ArduPilot SITL for Webots integration

set -e

ARDUPILOT_DIR="${HOME}/ardupilot"
SIM_VEHICLE="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Launching ArduPilot SITL for Webots...${NC}"

# Check ArduPilot installation
if [ ! -d "${ARDUPILOT_DIR}" ]; then
    echo -e "${RED}❌ ArduPilot not found at ${ARDUPILOT_DIR}${NC}"
    echo "Install with: git clone https://github.com/ArduPilot/ardupilot.git ~/ardupilot"
    exit 1
fi

if [ ! -f "${SIM_VEHICLE}" ]; then
    echo -e "${RED}❌ sim_vehicle.py not found${NC}"
    exit 1
fi

# Change to ArduCopter directory
cd "${ARDUPILOT_DIR}/ArduCopter"

echo -e "${YELLOW}📝 Starting SITL (first run may take 1-2 minutes to compile)...${NC}"
echo ""
echo -e "SITL will listen on:"
echo -e "  • Physics (JSON): ${GREEN}127.0.0.1:9002${NC}"
echo -e "  • MAVLink:        ${GREEN}127.0.0.1:5760${NC}"
echo ""
echo -e "After SITL starts, in another terminal run:"
echo -e "  ${GREEN}python webots_human_tracker.py --model runs/best_model${NC}"
echo ""

# Launch SITL with JSON physics backend
python3 "${SIM_VEHICLE}" -v ArduCopter -f JSON --console "$@"
