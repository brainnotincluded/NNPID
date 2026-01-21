#!/usr/bin/env python3
"""
NNPID - Neural Network PID Replacement for Drone Target Tracking

Main CLI entry point. Run with:
    uv run nnpid --help
    uv run python main.py --help
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nnpid',
        description='Neural Network PID Controller for Drone Target Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nnpid train                    # Start training
  nnpid train --steps 50000      # Train for 50K steps
  nnpid dashboard                # Launch web dashboard
  nnpid demo                     # Run quick simulation demo
  nnpid info                     # Show system info

For more options:
  nnpid train --help
  nnpid dashboard --help
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the RSAC model')
    train_parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    train_parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                              help='Device to train on')
    train_parser.add_argument('--config', type=str, default='config/training_config.yaml',
                              help='Path to config file')
    train_parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard')
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dash_parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    dash_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    dash_parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick simulation demo')
    demo_parser.add_argument('--episodes', type=int, default=3, help='Number of demo episodes')
    demo_parser.add_argument('--trajectory', type=str, default='evasive',
                             help='Trajectory type: stationary, linear, evasive, predator, etc.')
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'train':
        return run_train(args)
    elif args.command == 'dashboard':
        return run_dashboard(args)
    elif args.command == 'demo':
        return run_demo(args)
    elif args.command == 'info':
        return run_info()
    
    return 0


def run_train(args):
    """Run training."""
    print("🚀 Starting NNPID Training...")
    print(f"   Steps: {args.steps:,}")
    print(f"   Device: {args.device}")
    print(f"   Config: {args.config}")
    print()
    
    # Import here to avoid slow startup for help
    from scripts.train import main as train_main
    
    # Override sys.argv for the train script
    sys.argv = [
        'train.py',
        '--steps', str(args.steps),
        '--device', args.device,
        '--config', args.config,
    ]
    if args.no_tensorboard:
        sys.argv.append('--no-tensorboard')
    
    return train_main()


def run_dashboard(args):
    """Run web dashboard."""
    print("🌐 Starting NNPID Web Dashboard...")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Auto-reload: {not args.no_reload}")
    print()
    
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info"
    )
    return 0


def run_demo(args):
    """Run quick demo."""
    print("🎮 Running NNPID Demo...")
    print(f"   Episodes: {args.episodes}")
    print(f"   Trajectory: {args.trajectory}")
    print()
    
    import numpy as np
    from src.environment.simple_drone_sim import SimpleDroneSimulator
    from src.utils.trajectory_generator import TrajectoryType
    
    # Map string to enum
    traj_map = {t.value: t for t in TrajectoryType}
    traj_type = traj_map.get(args.trajectory, TrajectoryType.LISSAJOUS)
    
    env = SimpleDroneSimulator(
        dt=0.05,
        max_episode_steps=200,
        trajectory_type=traj_type
    )
    
    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Random policy for demo
            action = np.random.randn(3) * 0.5
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            if step % 50 == 0:
                env.render()
        
        print(f"\n📊 Episode {ep+1}: Reward={total_reward:.1f}, Steps={step}")
    
    print("\n✅ Demo complete!")
    return 0


def run_info():
    """Show system info."""
    import torch
    
    print("=" * 50)
    print("🚁 NNPID System Information")
    print("=" * 50)
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Check optional deps
    print("\n📦 Optional Dependencies:")
    try:
        import noise
        print("  ✅ noise (Perlin noise)")
    except ImportError:
        print("  ❌ noise (install with: uv pip install noise)")
    
    try:
        import pymavlink
        print("  ✅ pymavlink (drone communication)")
    except ImportError:
        print("  ❌ pymavlink (install with: uv pip install pymavlink)")
    
    # Project paths
    print("\n📁 Project Paths:")
    project_root = Path(__file__).parent
    print(f"  Root: {project_root}")
    print(f"  Config: {project_root / 'config'}")
    print(f"  Logs: {project_root / 'logs'}")
    print(f"  Checkpoints: {project_root / 'checkpoints'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
