"""
Launch API server with mock engine for testing.

Usage:
    python -m xorl.server.api_server [--port PORT] [--host HOST] [--mock]

Examples:
    # Launch with mock engine on default ports
    python -m xorl.server.api_server --mock

    # Launch on custom port
    python -m xorl.server.api_server --mock --port 8080

    # Launch with real engine (requires engine to be running separately)
    python -m xorl.server.api_server --engine-input tcp://127.0.0.1:6000 --engine-output tcp://127.0.0.1:6001
"""

import argparse
import asyncio
import logging
import sys
import uvicorn
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from xorl.server.api_server.api_server import APIServer, app
from xorl.server.engine.engine_core_proc_dummy import EngineCoreProc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch API server with optional mock engine"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=20000,
        help="API server port (default: 20000)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Launch with mock engine for testing"
    )
    parser.add_argument(
        "--engine-input",
        type=str,
        default="tcp://127.0.0.1:6000",
        help="Engine input address (default: tcp://127.0.0.1:6000)"
    )
    parser.add_argument(
        "--engine-output",
        type=str,
        default="tcp://127.0.0.1:6001",
        help="Engine output address (default: tcp://127.0.0.1:6001)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Default timeout for engine operations (default: 120.0)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Start mock engine if requested
    mock_engine = None
    if args.mock:
        logger.info("🚀 Starting mock engine for testing...")
        mock_engine = EngineCoreProc(
            input_addr=args.engine_input,
            output_addr=args.engine_output,
        )
        mock_engine.start()
        await asyncio.sleep(1.0)  # Wait for engine to bind sockets
        logger.info(f"✅ Mock engine started on {args.engine_input} -> {args.engine_output}")

    try:
        logger.info(f"✅ Mock engine ready" if args.mock else "✅ Connecting to engine...")
        logger.info(f"📡 Engine input: {args.engine_input}")
        logger.info(f"📡 Engine output: {args.engine_output}")
        logger.info(f"🌐 API will be available at http://{args.host}:{args.port}")
        logger.info(f"📚 API docs at http://{args.host}:{args.port}/docs")
        logger.info(f"🧪 Run 'python tests/server/api_server/test_api_client.py' to test the API")

        # Run uvicorn server (lifespan will handle APIServer start/stop)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        # Lifespan will stop API server automatically

        # Stop mock engine if running
        if mock_engine:
            mock_engine.stop()
            logger.info("✅ Mock engine stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
