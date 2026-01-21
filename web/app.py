"""
NNPID Web Dashboard - FastAPI Backend
Serves training dashboard, live demo, and training control endpoints.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import torch
import glob
import os

from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.utils.trajectory_generator import TrajectoryType, TrajectoryConfig, TrajectoryGenerator
from src.models.gru_networks import RSACSharedEncoder
from src.utils.domain_randomization import DronePhysicsParams, EnvironmentParams

import subprocess
import threading
import signal
import yaml
import time
from typing import Optional

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="NNPID Combat Tracking System",
    description="Neural Network PID Controller for Drone Target Tracking",
    version="1.0.0"
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Data directories
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/demo")
async def demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

# Global Training State
class TrainingManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_buffer = []
        self.is_running = False
        self.lock = threading.Lock()

    def start_training(self, steps: int = 100000, resume: bool = True):
        with self.lock:
            if self.is_running:
                return False, "Training already running"
            
            cmd = [
                sys.executable, 
                "scripts/train.py", 
                "--steps", str(steps),
                "--device", "cpu",  # Configurable later
                "--no-tensorboard"  # We visualize via CSV
            ]
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent)
            )
            self.is_running = True
            
            # Start log monitor thread
            threading.Thread(target=self._monitor_logs, daemon=True).start()
            return True, "Training started"

    def stop_training(self):
        with self.lock:
            if self.process and self.is_running:
                self.process.send_signal(signal.SIGINT)  # Graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                
                self.is_running = False
                self.process = None
                return True, "Training stopped"
            return False, "No training running"

    def _monitor_logs(self):
        """Read stdout from subprocess and buffer it"""
        if not self.process: return
        
        for line in self.process.stdout:
            self.log_buffer.append(line)
            if len(self.log_buffer) > 1000:
                self.log_buffer.pop(0)  # Keep last 1000 lines
        
        with self.lock:
            self.is_running = False
            self.process = None

trainer = TrainingManager()

@app.get("/training")
async def training_page(request: Request):
    # Load current config
    config_path = Path(__file__).parent.parent / "config/training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return templates.TemplateResponse("training.html", {"request": request, "config": config})

@app.post("/api/training/start")
async def start_training(steps: int = 100000):
    success, msg = trainer.start_training(steps)
    return {"success": success, "message": msg}

@app.post("/api/training/stop")
async def stop_training():
    success, msg = trainer.stop_training()
    return {"success": success, "message": msg}

@app.get("/api/training/status")
async def training_status():
    return {
        "is_running": trainer.is_running,
        "logs": trainer.log_buffer[-50:]  # Send last 50 lines
    }

@app.post("/api/training/config")
async def update_config(request: Request):
    """Update training_config.yaml"""
    try:
        new_config = await request.json()
        config_path = Path(__file__).parent.parent / "config/training_config.yaml"
        
        # Validate critical fields or merge carefully
        # For now, simplistic overwrite (careful!)
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)
            
        return {"success": True, "message": "Configuration saved"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/api/training-data")
async def get_training_data():
    """Get training metrics from CSV log."""
    csv_path = LOG_DIR / "training_log.csv"
    if not csv_path.exists():
        return JSONResponse({"error": "No training data found"}, status_code=404)
    
    try:
        df = pd.read_csv(csv_path)
        # Downsample if too large
        if len(df) > 1000:
            df = df.iloc[::len(df)//1000]
            
        data = {
            "episode": df["episode"].tolist(),
            "reward": df["reward"].tolist(),
            "length": df["length"].tolist(),
            "critic_loss": df["critic_loss"].tolist(),
            "actor_loss": df["actor_loss"].tolist()
        }
        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/run-simulation")
async def run_simulation(
    model_type: str = Query("latest", description="Model type: latest, random"),
    trajectory_type: str = Query("lissajous", description="Trajectory pattern"),
    target_speed: float = Query(1.0, ge=0.1, le=5.0, description="Target speed multiplier"),
    perturbation_level: float = Query(0.0, ge=0.0, le=1.0, description="Perturbation intensity"),
    wind_speed: float = Query(0.0, ge=0.0, le=10.0, description="Wind speed m/s"),
    wind_direction: float = Query(0.0, ge=0.0, le=360.0, description="Wind direction degrees"),
    episode_length: int = Query(200, ge=50, le=500, description="Episode length in steps")
):
    """
    Run a simulation episode with customizable parameters.
    Returns trajectory data for visualization.
    """
    # Map trajectory type string to enum
    traj_type_map = {
        "stationary": TrajectoryType.STATIONARY,
        "linear": TrajectoryType.LINEAR,
        "circular": TrajectoryType.CIRCULAR,
        "lissajous": TrajectoryType.LISSAJOUS,
        "lissajous_perlin": TrajectoryType.LISSAJOUS_PERLIN,
        "random_walk": TrajectoryType.RANDOM_WALK,
        "evasive": TrajectoryType.EVASIVE,
        "chaotic": TrajectoryType.CHAOTIC,
        "spiral_dive": TrajectoryType.SPIRAL_DIVE,
        "zigzag": TrajectoryType.ZIGZAG,
        "figure_eight": TrajectoryType.FIGURE_EIGHT,
        "drunk_walk": TrajectoryType.DRUNK_WALK,
        "predator": TrajectoryType.PREDATOR,
    }
    
    traj_type = traj_type_map.get(trajectory_type.lower(), TrajectoryType.LISSAJOUS)
    
    # Initialize environment with custom parameters
    env = SimpleDroneSimulator(
        dt=0.05,
        max_episode_steps=episode_length,
        trajectory_type=traj_type,
        use_domain_randomization=(perturbation_level > 0)
    )
    
    # Apply custom wind settings
    wind_rad = np.radians(wind_direction)
    env.current_env_params = EnvironmentParams(
        wind_velocity=np.array([
            wind_speed * np.cos(wind_rad),
            wind_speed * np.sin(wind_rad),
            0.0
        ]),
        wind_turbulence=wind_speed * 0.2 * perturbation_level
    )
    
    # Load model if requested
    policy = None
    if model_type == "latest":
        checkpoints = sorted(glob.glob(str(CHECKPOINT_DIR / "*.pt")))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            try:
                # Load checkpoint
                checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
                policy = RSACSharedEncoder(env.obs_dim, env.action_dim)
                policy.load_state_dict(checkpoint['policy_state_dict'])
                policy.eval()
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                policy = None
    
    # Run episode
    obs = env.reset()
    
    # Scale trajectory velocities by target_speed
    if target_speed != 1.0 and env.target_velocities is not None:
        env.target_velocities = env.target_velocities * target_speed
        # Recompute positions from scaled velocities
        positions = [env.target_trajectory[0]]
        for i in range(1, len(env.target_velocities)):
            new_pos = positions[-1] + env.target_velocities[i] * env.dt
            positions.append(new_pos)
        env.target_trajectory = np.array(positions)
    
    done = False
    
    trajectory = {
        "drone_pos": [],
        "target_pos": [],
        "drone_vel": [],
        "actions": [],
        "reward": 0.0,
        "config": {
            "trajectory_type": trajectory_type,
            "target_speed": target_speed,
            "perturbation_level": perturbation_level,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction
        }
    }
    
    hidden_state = None
    if policy:
        hidden_state = policy.init_hidden(1, torch.device('cpu'))
    
    # Perturbation injection state
    perturbation_event = None
    perturbation_timer = 0
    
    while not done:
        # Get target pos (for visualization)
        target_pos_ned = env.target_trajectory[min(env.current_step, len(env.target_trajectory)-1)]
        drone_pos_ned = env.drone_state.position_ned
        
        trajectory["drone_pos"].append(drone_pos_ned.tolist())
        trajectory["target_pos"].append(target_pos_ned.tolist())
        trajectory["drone_vel"].append(env.drone_state.velocity_ned.tolist())
        
        # Random perturbation injection
        if perturbation_level > 0 and np.random.random() < 0.01 * perturbation_level:
            perturbation_event = np.random.choice(['gust', 'dropout', 'impulse'])
            perturbation_timer = np.random.randint(5, 20)
        
        # Select action
        if policy:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Simulate sensor dropout during perturbation
                if perturbation_event == 'dropout' and perturbation_timer > 0:
                    obs_tensor = obs_tensor * 0.5 + torch.randn_like(obs_tensor) * 0.5
                
                encoded, hidden_state = policy.encode(obs_tensor, hidden_state)
                action, _ = policy.actor_forward(encoded, deterministic=True)
                action = action.numpy()[0]
        else:
            # Random action
            action = np.random.randn(3) * 0.5
        
        # Apply perturbation to action
        if perturbation_event == 'gust' and perturbation_timer > 0:
            action = action + np.random.randn(3) * perturbation_level * 2
        elif perturbation_event == 'impulse' and perturbation_timer > 0:
            action = action * (1 + np.random.randn() * perturbation_level)
        
        if perturbation_timer > 0:
            perturbation_timer -= 1
            if perturbation_timer == 0:
                perturbation_event = None
            
        trajectory["actions"].append(action.tolist())
        
        # Step
        obs, reward, done, info = env.step(action)
        trajectory["reward"] += reward
        
    return trajectory


@app.get("/api/trajectory-types")
async def get_trajectory_types():
    """Return available trajectory types for the UI dropdown."""
    return {
        "types": [
            {"value": "stationary", "label": "Stationary", "difficulty": 1},
            {"value": "linear", "label": "Linear", "difficulty": 2},
            {"value": "circular", "label": "Circular", "difficulty": 3},
            {"value": "lissajous", "label": "Lissajous", "difficulty": 4},
            {"value": "figure_eight", "label": "Figure-8", "difficulty": 4},
            {"value": "spiral_dive", "label": "Spiral Dive", "difficulty": 5},
            {"value": "lissajous_perlin", "label": "Lissajous + Noise", "difficulty": 6},
            {"value": "random_walk", "label": "Random Walk", "difficulty": 6},
            {"value": "zigzag", "label": "Zigzag", "difficulty": 7},
            {"value": "drunk_walk", "label": "Drunk Walk", "difficulty": 7},
            {"value": "chaotic", "label": "Chaotic", "difficulty": 8},
            {"value": "evasive", "label": "Evasive", "difficulty": 9},
            {"value": "predator", "label": "Predator (Adversarial)", "difficulty": 10},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
