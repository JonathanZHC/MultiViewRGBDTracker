# Architecture

## Process layout

The repository uses one Docker image and three processes:

1. `isaac_sim/run_isaacsim.py` creates one of the three procedural scenes and publishes RGB, depth, camera calibration, TF and simulator ground truth.
2. `sam_rgbd_tracking_benchmark.node` runs SAM3 keyframes, one selected tracker backend, RGB-D ownership filtering, point-cloud extraction and profiling.
3. RViz consumes only ROS 2 topics and is independent of model execution.

The model objects are shared across camera sessions to avoid duplicate weights. Each camera has an independent tracking state. A process-wide CUDA lock is enabled by default so the timing traces remain deterministic and concurrent calls do not race inside upstream predictors.

## Keyframe path

A keyframe is triggered on the first frame, periodically, or by an anomaly. SAM3 produces masks, prompt labels and scores. Existing tracks are associated using label, mask IoU, 3D centroid and depth consistency. The selected tracker is then re-seeded and its bounded streaming window starts again from that frame.

SAM-MT's current public multi-target interface is point-prompt based. Its adapter samples interior points from each SAM3 mask and runs all objects jointly. EfficientTAM accepts each SAM3 mask directly and internally propagates the objects through its SAM2-style predictor.

## Per-frame post-processing

For all tracker logits:

1. Generate thresholded candidates.
2. Resolve cross-instance overlap with logit competition and track-specific depth models.
3. Reject pixels whose measured depth is inconsistent with the assigned track.
4. Reject depth discontinuities.
5. Erode masks and remove small components.
6. Back-project only surviving visible pixels.
7. Publish an empty current point cloud for a fully occluded object while retaining its track state.

The final owner map is rebuilt after erosion and depth-edge filtering, so published ownership and evaluation correspond exactly to the emitted point clouds.

## Integration boundary

The future ScenePredictor integration should depend only on `ProcessedInstance` / `FrameResult`. No downstream code needs to know whether SAM-MT or EfficientTAM generated the masks.
