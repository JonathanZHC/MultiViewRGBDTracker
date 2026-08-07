# Validation status

Validated in the artifact environment:

- all Python files compile;
- all shell launch scripts pass `bash -n`;
- 12 unit tests pass;
- the synthetic RGB-D → mask tracking → depth filtering → point-cloud pipeline completes;
- the Python wheel builds without dependency resolution;
- archives are extracted and tested again before delivery.

The Isaac Sim code cannot be executed in the artifact environment because it
has no NVIDIA/Isaac runtime. It is structured around the same standalone
lifecycle used by the working `ScenePredictor` YOLOE branch:

- `/isaac-sim/python.sh path/to/run_isaacsim.py`;
- `SimulationApp` before Omniverse imports;
- direct `UsdGeom.Camera` creation;
- `Gf.Quatf` camera orientation;
- CUDA RGB and depth annotators;
- a direct `simulation_app.is_running()` publication loop.

The user-provided RTX 5090 log already confirms that Isaac Sim 6.0.1, Vulkan,
system Jazzy and the ROS bridge start successfully. The previous crash was a
camera quaternion type mismatch; version 0.3.0 removes that implementation.
