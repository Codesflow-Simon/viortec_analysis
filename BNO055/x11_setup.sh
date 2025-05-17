#!/bin/bash

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== X11 Connection Tester ===${NC}"

# Function to test X11 connection
test_connection() {
    local method=$1
    local display=$2
    
    echo -e "\n${YELLOW}Testing Method: ${method}${NC}"
    echo "Setting DISPLAY=$display"
    
    export DISPLAY=$display
    
    # Try to enable connections
    xhost + >/dev/null 2>&1
    
    # Test connection with xdpyinfo
    if xdpyinfo >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Connection SUCCESSFUL${NC}"
        echo -e "Method: $method\nDISPLAY=$display\n"
        return 0
    else
        echo -e "${RED}✗ Connection FAILED${NC}"
        return 1
    fi
}

# Function to get IP addresses
get_ips() {
    echo -e "${BLUE}Detecting IP Addresses...${NC}"
    
    NAMESERVER_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
    DEFAULT_IP=$(ip route | grep default | awk '{print $3}')
    HOSTNAME_IP=$(hostname -I | awk '{print $1}')
    
    echo "Nameserver IP: $NAMESERVER_IP"
    echo "Default Route IP: $DEFAULT_IP"
    echo "Hostname IP: $HOSTNAME_IP"
}

# Clean up existing settings
cleanup() {
    echo -e "${BLUE}Cleaning up existing X11 settings...${NC}"
    unset DISPLAY
    xhost - >/dev/null 2>&1
}

# Main testing sequence
main() {
    cleanup
    get_ips
    
    # Store successful methods
    declare -a successful_methods=()
    
    echo -e "\n${BLUE}Testing different connection methods...${NC}"
    
    # Method 1: Basic local display
    if test_connection "Basic :0" ":0"; then
        successful_methods+=("export DISPLAY=:0")
    fi
    
    # Method 2: Localhost
    if test_connection "Localhost" "localhost:0.0"; then
        successful_methods+=("export DISPLAY=localhost:0.0")
    fi
    
    # Method 3: Loopback IP
    if test_connection "Loopback IP" "127.0.0.1:0"; then
        successful_methods+=("export DISPLAY=127.0.0.1:0")
    fi
    
    # Method 4: Nameserver IP (WSL2 common method)
    if test_connection "Nameserver IP" "$NAMESERVER_IP:0"; then
        successful_methods+=("export DISPLAY=$NAMESERVER_IP:0")
    fi
    
    # Method 5: Default Route IP
    if test_connection "Default Route IP" "$DEFAULT_IP:0"; then
        successful_methods+=("export DISPLAY=$DEFAULT_IP:0")
    fi
    
    # Method 6: With LIBGL_ALWAYS_INDIRECT
    export LIBGL_ALWAYS_INDIRECT=1
    if test_connection "With LIBGL_ALWAYS_INDIRECT" "$NAMESERVER_IP:0"; then
        successful_methods+=("export LIBGL_ALWAYS_INDIRECT=1; export DISPLAY=$NAMESERVER_IP:0")
    fi
    
    # Report results
    echo -e "\n${BLUE}=== Test Results ===${NC}"
    if [ ${#successful_methods[@]} -eq 0 ]; then
        echo -e "${RED}No successful connection methods found.${NC}"
        echo -e "\nTroubleshooting steps:"
        echo "1. Is VcXsrv running with 'Disable access control' checked?"
        echo "2. Check Windows Defender Firewall settings"
        echo "3. Try running XLaunch as Administrator"
        echo "4. Verify no other X server is running"
    else
        echo -e "${GREEN}Found ${#successful_methods[@]} working method(s):${NC}"
        echo -e "\nCopy and paste one of these commands to set up X11:"
        for method in "${successful_methods[@]}"; do
            echo -e "${YELLOW}$method${NC}"
        done
    fi
}

# Run main function
main

echo -e "\n${BLUE}=== Script Complete ===${NC}"
echo "Copy and paste your preferred method from above to enable X11" 