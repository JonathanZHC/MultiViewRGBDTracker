# Camera / Synchronization Diagnostics

This version keeps the existing tracking runtime unchanged and extends the
5-second `[Rate:<camera>]` report with diagnostics *before* the tracking worker.

## Raw topic statistics

For each of `color`, `depth`, and `CameraInfo`, the report shows:

- wall-clock callback rate and message count;
- header-stamp rate;
- wall-clock inter-arrival gap mean/median/p95/max;
- header-stamp gap mean/median/p95/max;
- duplicate or backwards (`nonmono`) header-stamp count;
- the exact ROS topic name being measured.

The raw callbacks are attached directly to the existing `message_filters.Subscriber`
objects. They do not decode images, copy image payloads, or modify messages.

## Synchronizer statistics

For the output of `ApproximateTimeSynchronizer`, the report shows:

- configured synchronization slop;
- synchronized/raw count ratio for color, depth, and CameraInfo;
- absolute color-depth timestamp skew;
- absolute color-info timestamp skew;
- absolute depth-info timestamp skew;
- total timestamp span of each matched tuple.

The count ratio is intentionally reported independently for each raw topic. It
is a diagnostic ratio, not an assumed probability; some message-filter patterns
can reuse or otherwise pair messages in ways that make a single ratio alone
misleading.

## Existing scheduling/runtime diagnostics

The existing report remains intact:

- synchronized input Hz;
- processed Hz;
- published Hz;
- real bounded-queue drop count;
- queue wait;
- ROS decode;
- TF lookup;
- component runtime;
- RViz message building;
- ROS publishing;
- worker total;
- SAM3 and EfficientTAM profiler stages.

## Reading the result

Typical interpretations:

- `color ~30 Hz`, `depth ~30 Hz`, `info ~1 Hz`, `sync ~1 Hz`:
  CameraInfo publication is limiting synchronization.
- all three raw streams ~30 Hz, but sync is much lower:
  inspect timestamp skew and `sync_slop_seconds`.
- all three raw streams themselves are low:
  the problem is upstream of the tracker/synchronizer (for example the Isaac
  camera publication path or ROS delivery).
- raw streams and sync are ~30 Hz, but `processed` is lower with nonzero drops:
  the worker/GPU path is the limiting stage.

## Optional YAML

No YAML changes are required. The diagnostics default to enabled. They can be
controlled with:

```yaml
profiling:
  rate_diagnostics: true
  camera_diagnostics: true
  rate_summary_interval_seconds: 5.0
```
