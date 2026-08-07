# Benchmark protocol

## Required comparisons

Run the three scenes (`static`, `dynamic`, `occlusion`) with both tracker backends. For every run, record tracking-only frames and SAM3 keyframes separately.

Recommended matrix:

- Tracker: SAM-MT, EfficientTAM
- Detector: simulator ground truth, SAM3
- Refresh interval: 0.5 s, 1.0 s, 2.0 s
- Depth filtering: disabled, overlap-only, all-candidate depth gate
- Cameras: one and two

## Warm-up

Discard at least three model frames after model construction and after each backend switch. Preserve the first keyframe latency in a separate cold-start report.

## Timing

Report CPU wall time and CUDA event time where available. Do not synchronize after every stage; record events and synchronize once at the end of a frame. Report mean, median, p95, p99 and max for:

- tracking-only pipeline
- keyframe pipeline
- full online stream
- sensor-to-output latency

Also report peak allocated and reserved CUDA memory.

## Accuracy

Primary metrics:

1. Cross-instance point contamination.
2. False visible points during full occlusion.
3. ID switches and reappearance recovery latency.
4. Visible-point recall.
5. Mask IoU and boundary F-score.

For fair A/B comparison, record the Isaac stream once and replay the same RGB-D frames, GT and cached SAM3 keyframes through both backends.

## Selection rule

Choose the backend with the lowest point contamination and acceptable ID stability subject to tracking-only p95 below 33.3 ms on the RTX 5090. Mean FPS alone is not sufficient.
