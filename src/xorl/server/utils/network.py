"""
Network utilities for multi-node server deployment.

Provides IP detection and address building for ZMQ communication between
Engine and Workers in distributed setups.
"""

import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """
    Get the routable local IP address.

    Priority:
    1. XORL_LOCAL_IP environment variable (explicit override)
    2. Auto-detect by connecting to external address

    Returns:
        Local IP address string (e.g., "192.168.1.100")
        Falls back to "127.0.0.1" if detection fails

    Example:
        >>> os.environ["XORL_LOCAL_IP"] = "10.0.0.5"
        >>> get_local_ip()
        '10.0.0.5'

        >>> # Without env var, auto-detects
        >>> get_local_ip()
        '192.168.1.100'
    """
    # Priority 1: Explicit environment variable
    local_ip = os.environ.get("XORL_LOCAL_IP")
    if local_ip:
        logger.debug(f"Using IP from XORL_LOCAL_IP: {local_ip}")
        return local_ip

    # Priority 2: Auto-detect by connecting to external address
    # This finds the IP address used for outbound connections
    try:
        # Create a UDP socket (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS server (doesn't send data, just determines route)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        logger.debug(f"Auto-detected local IP: {local_ip}")
        return local_ip
    except Exception as e:
        logger.warning(f"Failed to auto-detect local IP: {e}, falling back to 127.0.0.1")
        return "127.0.0.1"


def build_worker_bind_address(host: str = "0.0.0.0", port: int = 5556) -> str:
    """
    Build ZMQ bind address for worker ROUTER socket.

    Args:
        host: Host to bind to. Use "0.0.0.0" to accept connections from any interface.
        port: Port number to bind.

    Returns:
        ZMQ address string (e.g., "tcp://0.0.0.0:5556")

    Example:
        >>> build_worker_bind_address("0.0.0.0", 5556)
        'tcp://0.0.0.0:5556'
    """
    return f"tcp://{host}:{port}"


def build_worker_connect_address(host: str, port: int) -> str:
    """
    Build ZMQ connect address for Engine DEALER socket.

    Args:
        host: Worker host IP address (must be routable from Engine)
        port: Worker port number

    Returns:
        ZMQ address string (e.g., "tcp://192.168.1.100:5556")

    Example:
        >>> build_worker_connect_address("192.168.1.100", 5556)
        'tcp://192.168.1.100:5556'
    """
    return f"tcp://{host}:{port}"


def parse_zmq_address(address: str) -> tuple[str, int]:
    """
    Parse a ZMQ TCP address into host and port.

    Args:
        address: ZMQ address string (e.g., "tcp://192.168.1.100:5556")

    Returns:
        Tuple of (host, port)

    Raises:
        ValueError: If address format is invalid

    Example:
        >>> parse_zmq_address("tcp://192.168.1.100:5556")
        ('192.168.1.100', 5556)
    """
    if not address.startswith("tcp://"):
        raise ValueError(f"Invalid ZMQ address format: {address} (must start with tcp://)")

    # Remove tcp:// prefix
    addr_part = address[6:]

    # Split host and port
    if ":" not in addr_part:
        raise ValueError(f"Invalid ZMQ address format: {address} (missing port)")

    host, port_str = addr_part.rsplit(":", 1)

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"Invalid port in ZMQ address: {address}")

    return host, port


def write_address_file(address: str, output_dir: str, filename: str = ".rank0_address") -> str:
    """
    Write worker address to a file for discovery by Engine.

    Used in multi-node setups where the Engine needs to discover the
    rank 0 worker's address. The file is written to a shared filesystem.

    Args:
        address: ZMQ address string (e.g., "tcp://192.168.1.100:5556")
        output_dir: Directory to write the file (should be on shared filesystem)
        filename: Name of the address file (default: ".rank0_address")

    Returns:
        Full path to the written address file

    Example:
        >>> write_address_file("tcp://10.0.0.5:5556", "/shared/outputs")
        '/shared/outputs/.rank0_address'
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write address to file
    address_file = Path(output_dir) / filename
    address_file.write_text(address)

    logger.info(f"Wrote rank 0 address to: {address_file}")
    return str(address_file)


def read_address_file(
    output_dir: str,
    filename: str = ".rank0_address",
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> Optional[str]:
    """
    Read worker address from discovery file.

    Used in multi-node setups by the Engine to discover the rank 0
    worker's address. Polls the file until it exists or timeout.

    Args:
        output_dir: Directory containing the address file
        filename: Name of the address file (default: ".rank0_address")
        timeout: Maximum time to wait for file in seconds (default: 120.0)
        poll_interval: Time between file checks in seconds (default: 1.0)

    Returns:
        ZMQ address string, or None if timeout

    Example:
        >>> read_address_file("/shared/outputs", timeout=60.0)
        'tcp://10.0.0.5:5556'
    """

    address_file = Path(output_dir) / filename
    start_time = time.time()

    logger.info(f"Waiting for rank 0 address file: {address_file} (timeout={timeout}s)")

    while (time.time() - start_time) < timeout:
        if address_file.exists():
            address = address_file.read_text().strip()
            if address:
                logger.info(f"Read rank 0 address: {address}")
                return address

        time.sleep(poll_interval)

    logger.warning(f"Timeout waiting for address file: {address_file}")
    return None
