"""Test configuration consistency between YAML and Python defaults.

This test ensures that YAML configuration files match Python dataclass defaults
to prevent issues like #004 where mismatches caused training inconsistencies.
"""

import pytest
import yaml
from pathlib import Path

from src.environments.yaw_tracking_env import YawTrackingConfig


class TestConfigConsistency:
    """Test that YAML configs match Python defaults for critical parameters."""
    
    def test_yaw_tracking_rewards_match(self):
        """Test that reward parameters match between YAML and Python."""
        # Load YAML config directly
        config_path = Path(__file__).parent.parent / "config" / "yaw_tracking.yaml"
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        # Get Python defaults
        py_cfg = YawTrackingConfig()
        
        # Check crash_penalty (Bug #1 from Issue #004)
        yaml_crash = yaml_cfg['environment']['rewards']['crash_penalty']
        assert yaml_crash == py_cfg.crash_penalty, \
            f"crash_penalty mismatch: YAML={yaml_crash}, Python={py_cfg.crash_penalty}"
        
        # Check alive_bonus (Bug #2 from Issue #004)
        yaml_alive = yaml_cfg['environment']['rewards']['alive_bonus']
        assert yaml_alive == py_cfg.alive_bonus, \
            f"alive_bonus mismatch: YAML={yaml_alive}, Python={py_cfg.alive_bonus}"
    
    def test_yaw_tracking_stabilizer_match(self):
        """Test that stabilizer parameters match between YAML and Python."""
        # Load YAML config directly
        config_path = Path(__file__).parent.parent / "config" / "yaw_tracking.yaml"
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        # Get Python defaults
        py_cfg = YawTrackingConfig()
        
        # Check yaw_authority (Bug #3 from Issue #004)
        yaml_yaw_auth = yaml_cfg['environment']['stabilizer']['yaw_authority']
        assert yaml_yaw_auth == py_cfg.yaw_authority, \
            f"yaw_authority mismatch: YAML={yaml_yaw_auth}, Python={py_cfg.yaw_authority}"
    
    def test_critical_parameters_documentation(self):
        """Test that critical parameters have comments explaining their values."""
        config_path = Path(__file__).parent.parent / "config" / "yaw_tracking.yaml"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check that updated parameters have explanation comments
        assert "crash_penalty: 50.0" in content, "crash_penalty not set to 50.0"
        assert "alive_bonus: 0.1" in content, "alive_bonus not set to 0.1"
        assert "yaw_authority: 0.20" in content, "yaw_authority not set to 0.20"
        
        # Check for explanatory comments
        assert "updated from" in content.lower() or "allows" in content.lower(), \
            "Missing explanatory comments for parameter changes"
    
    def test_yaml_syntax_valid(self):
        """Test that YAML config has valid syntax."""
        config_path = Path(__file__).parent.parent / "config" / "yaw_tracking.yaml"
        
        # Should not raise
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check structure
        assert 'environment' in config
        assert 'rewards' in config['environment']
        assert 'stabilizer' in config['environment']
    
    def test_parameter_ranges_are_reasonable(self):
        """Test that parameter values are in reasonable ranges."""
        py_cfg = YawTrackingConfig()
        
        # crash_penalty should be high enough to matter
        assert py_cfg.crash_penalty >= 10.0, \
            "crash_penalty too low - won't discourage crashes"
        
        # alive_bonus should be positive but not dominate
        assert 0.0 < py_cfg.alive_bonus < 1.0, \
            "alive_bonus out of reasonable range"
        
        # yaw_authority should allow meaningful control
        assert py_cfg.yaw_authority >= 0.1, \
            "yaw_authority too low - won't allow effective yaw control"
        assert py_cfg.yaw_authority <= 1.0, \
            "yaw_authority too high - may cause instability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
