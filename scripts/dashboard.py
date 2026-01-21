#!/usr/bin/env python3
"""
Launch the NNPID Web Dashboard.

Usage:
    uv run python scripts/dashboard.py
    uv run python scripts/dashboard.py --port 8080
"""
import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Launch web dashboard."""
    parser = argparse.ArgumentParser(description='Launch NNPID Web Dashboard')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    args = parser.parse_args()
    
    import uvicorn
    
    print("\n" + "=" * 50)
    print("🚀 NNPID Combat Tracking System")
    print("=" * 50)
    print(f"\n🌐 Dashboard: http://localhost:{args.port}")
    print(f"📊 Training:  http://localhost:{args.port}/training")
    print(f"🎮 Demo:      http://localhost:{args.port}/demo")
    print(f"📈 Metrics:   http://localhost:{args.port}/dashboard")
    print("\n" + "=" * 50 + "\n")
    
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
