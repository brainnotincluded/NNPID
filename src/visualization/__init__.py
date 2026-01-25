"""Visualization and rendering components.

This package provides comprehensive visualization tools for the NNPID project:

- MuJoCoViewer: Core MuJoCo rendering wrapper
- TelemetryDashboard: matplotlib-based telemetry plotting
- SceneObjectManager: 3D scene objects (arrows, trails, etc.)
- NNVisualizer: Neural network structure and activation visualization
- TelemetryHUD: Real-time heads-up display with gauges
- MegaVisualizer: Combined overlay system integrating all components
"""

# Conditionally import components that require OpenCV
import importlib.util

from .dashboard import TelemetryDashboard
from .viewer import MuJoCoViewer

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

# 3D Scene objects (require MuJoCo only)
try:
    from .scene_objects import (  # noqa: F401
        GeomBuilder,
        SceneObjectConfig,
        SceneObjectManager,
        create_default_scene_objects,
        create_full_scene_objects,
        create_minimal_scene_objects,
    )

    SCENE_OBJECTS_AVAILABLE = True
except ImportError:
    SCENE_OBJECTS_AVAILABLE = False

# Neural network visualizer (requires OpenCV)
if CV2_AVAILABLE:
    try:
        from .nn_visualizer import (  # noqa: F401
            NetworkExtractor,
            NNVisualizer,
            NNVisualizerConfig,
            create_compact_nn_visualizer,
            create_default_nn_visualizer,
        )

        NN_VISUALIZER_AVAILABLE = True
    except ImportError:
        NN_VISUALIZER_AVAILABLE = False
else:
    NN_VISUALIZER_AVAILABLE = False

# Telemetry HUD (requires OpenCV)
if CV2_AVAILABLE:
    try:
        from .telemetry_hud import (  # noqa: F401
            TelemetryHUD,
            TelemetryHUDConfig,
            create_compact_hud,
            create_default_hud,
        )

        TELEMETRY_HUD_AVAILABLE = True
    except ImportError:
        TELEMETRY_HUD_AVAILABLE = False
else:
    TELEMETRY_HUD_AVAILABLE = False

# Mega visualizer (requires OpenCV and other components)
if CV2_AVAILABLE:
    try:
        from .mujoco_overlay import (  # noqa: F401
            FrameAnnotator,
            MegaVisualizer,
            MegaVisualizerConfig,
            create_default_visualizer,
            create_full_visualizer,
            create_minimal_visualizer,
            create_recording_visualizer,
        )

        MEGA_VISUALIZER_AVAILABLE = True
    except ImportError:
        MEGA_VISUALIZER_AVAILABLE = False
else:
    MEGA_VISUALIZER_AVAILABLE = False

# Build exports list
__all__ = [
    # Core
    "MuJoCoViewer",
    "TelemetryDashboard",
    # Availability flags
    "CV2_AVAILABLE",
    "SCENE_OBJECTS_AVAILABLE",
    "NN_VISUALIZER_AVAILABLE",
    "TELEMETRY_HUD_AVAILABLE",
    "MEGA_VISUALIZER_AVAILABLE",
]

# Add scene objects if available
if SCENE_OBJECTS_AVAILABLE:
    __all__.extend(
        [
            "SceneObjectManager",
            "SceneObjectConfig",
            "GeomBuilder",
            "create_default_scene_objects",
            "create_minimal_scene_objects",
            "create_full_scene_objects",
        ]
    )

# Add NN visualizer if available
if NN_VISUALIZER_AVAILABLE:
    __all__.extend(
        [
            "NNVisualizer",
            "NNVisualizerConfig",
            "NetworkExtractor",
            "create_default_nn_visualizer",
            "create_compact_nn_visualizer",
        ]
    )

# Add telemetry HUD if available
if TELEMETRY_HUD_AVAILABLE:
    __all__.extend(
        [
            "TelemetryHUD",
            "TelemetryHUDConfig",
            "create_default_hud",
            "create_compact_hud",
        ]
    )

# Add mega visualizer if available
if MEGA_VISUALIZER_AVAILABLE:
    __all__.extend(
        [
            "MegaVisualizer",
            "MegaVisualizerConfig",
            "FrameAnnotator",
            "create_default_visualizer",
            "create_minimal_visualizer",
            "create_full_visualizer",
            "create_recording_visualizer",
        ]
    )
