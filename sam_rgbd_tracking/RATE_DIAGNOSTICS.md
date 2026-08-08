# End-to-end rate diagnostics

This folder adds diagnostic instrumentation only. It does not change the
EfficientTAM/SAM-MT tracking algorithm, CUDAGraph settings, stream reuse, or the
no-depth-occlusion mask path.

## What is reported

For each camera, every ~5 seconds:

- synchronized RGB-D input rate;
- processed frame rate;
- published frame rate;
- exact bounded-queue drop count and drop percentage;
- synchronized sensor-header timestamp rate;
- worker errors, keyframe count, and current queue depth;
- queue wait time;
- ROS image/depth decode time;
- TF lookup time;
- component time;
- RViz message-build time;
- ROS publish-call time;
- complete worker time;
- selected existing component/tracker stages: pipeline total, postprocess,
  SAM3, tracker lock wait, propagation, tracker CUDA total, and tracker wall
  total.

Each timing reports n / mean / median / p95 / max for the current window.
Cumulative counts are also printed.

## Optional config

No YAML change is required. Defaults are:

```yaml
profiling:
  rate_diagnostics: true
  rate_summary_interval_seconds: 5.0
```

Set `rate_diagnostics: false` to disable this extra report.

## Important definitions

`RGB-D synchronized input` counts callbacks from
`ApproximateTimeSynchronizer`, so losses before synchronization are not counted
as worker queue drops.

`dropped` counts actual synchronized packets discarded by the bounded worker
queue. With `drop_when_busy: true`, this is normally an older pending packet
that is replaced by the newest packet. With `drop_when_busy: false`, it is the
new packet rejected when the queue is full.

`queue_wait_cpu` measures enqueue -> worker dequeue latency. `worker_total_cpu`
measures dequeue -> end of publish/error handling, and therefore does not
include waiting for the next input frame.

`component_cpu` is wall-clock time around `process_arrays()`. The nested
`pipeline / periodic stalls` values are the existing profiler stages returned
inside `FrameResult.timings_ms`.
