"""CUDA depth-boundary flyer filter for tracked instance masks.

Replaces unconditional boundary erosion with a per-instance, depth-adaptive recovery:

  1. ``erode_pack``  – pack all tracked masks of a view into one ``uint32`` per pixel
                       (bit = instance slot) and compute the ``w``-pixel 4-connected
                       (Manhattan-ball) eroded *core* in a single launch. Reliable
                       core-internal 4-neighbour depth differences are sampled into 512
                       reservoir slots per instance (hash-keyed ``atomicMax`` -> deterministic).
  2. ``mad_stats``   – one block per instance: shared-memory bitonic sort -> median, MAD,
                       ``T = median + k * 1.4826 * MAD``.
  3. ``grow``        – ``w`` fused Jacobi rounds inside one launch (shared tile + halo):
                       a pixel of the input mask is re-accepted when a 4-neighbour is already
                       accepted and ``|D(p) - D(q)| < T``. Writes one ``uint8`` mask per record.

All views and instances are processed batched: grid.z = view, bits = instances. Everything
is CUDA-resident on the caller's stream; no host synchronisation, no per-frame allocation.

Instances whose core yields fewer than ``MIN_SAMPLES`` reliable edges (thin / tiny objects)
fall back to the unfiltered input mask so they are never silently erased.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .cuda_jit import Module

MAX_PER_VIEW = 32      # instances per view (bits of uint32)
NSAMP = 512            # reservoir slots per instance for the MAD estimate
MIN_SAMPLES = 16       # below this the instance falls back to its input mask
TX, TY = 32, 16        # tile size; halo == erosion width

_SOURCE = r"""
#ifndef EW
#define EW 5
#endif
#define TX 32
#define TY 16
#define HALO EW
#define SW (TX + 2 * HALO)
#define SH (TY + 2 * HALO)
#define NSAMP 512
#define MAXB 32
#define BIG 1e30f

// table layout (int32): [0,R) rec_view | [R,2R) rec_bit | [2R, 2R + V*MAXB) view_bit_rec (-1 = none)

extern "C" __global__ void __launch_bounds__(TX * TY)
erode_pack(const unsigned char* __restrict__ masks,    // [R, H, W] 0/1
           const float* __restrict__ depth,            // [V, H, W]
           const int* __restrict__ table,
           unsigned int* __restrict__ orig,            // [V, H, W] packed, gated by depth > 0
           unsigned int* __restrict__ core,            // [V, H, W] packed, eroded
           unsigned long long* __restrict__ samples,   // [R, NSAMP] (0 = empty)
           int R, int H, int W)
{
    __shared__ unsigned int s[SH][SW];
    __shared__ unsigned int sc[TY][TX];
    __shared__ int s_rec_view[MAXB], s_rec_bit[MAXB], s_bit_rec[MAXB];

    const int v = blockIdx.z;
    const int tid = threadIdx.y * TX + threadIdx.x;
    const int HW = H * W;
    if (tid < R) { s_rec_view[tid] = table[tid]; s_rec_bit[tid] = table[R + tid]; }
    if (tid < MAXB) s_bit_rec[tid] = table[2 * R + v * MAXB + tid];
    __syncthreads();

    const int x0 = blockIdx.x * TX - HALO, y0 = blockIdx.y * TY - HALO;
    const float* dview = depth + (size_t)v * HW;
    for (int i = tid; i < SW * SH; i += TX * TY) {
        int lx = i % SW, ly = i / SW;
        int x = x0 + lx, y = y0 + ly;
        unsigned int val = 0;
        if (x >= 0 && x < W && y >= 0 && y < H) {
            int p = y * W + x;
            if (dview[p] > 0.f) {
                for (int r = 0; r < R; ++r)
                    if (s_rec_view[r] == v && masks[(size_t)r * HW + p]) val |= 1u << s_rec_bit[r];
            }
        }
        s[ly][lx] = val;
    }
    __syncthreads();

    const int x = blockIdx.x * TX + threadIdx.x, y = blockIdx.y * TY + threadIdx.y;
    const bool inside = (x < W && y < H);
    const int lx = threadIdx.x + HALO, ly = threadIdx.y + HALO;
    unsigned int c = inside ? s[ly][lx] : 0u;
    if (c) {
#pragma unroll
        for (int dy = -EW; dy <= EW; ++dy) {
            int r = EW - (dy < 0 ? -dy : dy);
            for (int dx = -r; dx <= r; ++dx) c &= s[ly + dy][lx + dx];
        }
    }
    sc[threadIdx.y][threadIdx.x] = c;
    __syncthreads();
    if (!inside) return;
    const int p = y * W + x;
    orig[(size_t)v * HW + p] = s[ly][lx];
    core[(size_t)v * HW + p] = c;

    if (c) {
        unsigned int cr = (threadIdx.x + 1 < TX && x + 1 < W) ? sc[threadIdx.y][threadIdx.x + 1] : 0u;
        unsigned int cd = (threadIdx.y + 1 < TY && y + 1 < H) ? sc[threadIdx.y + 1][threadIdx.x] : 0u;
        unsigned int both = c & (cr | cd);
        if (both) {
            unsigned int h = (unsigned int)p * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
            float d = dview[p];
            float gr = cr ? fabsf(d - dview[p + 1]) : 0.f;
            float gd = cd ? fabsf(d - dview[p + W]) : 0.f;
            unsigned long long hk = (unsigned long long)(h | 1u) << 32;
            int slot = h % NSAMP;
            while (both) {
                int b = __ffs(both) - 1;
                both &= both - 1;
                int rec = s_bit_rec[b];
                if (rec >= 0) {
                    float g = (cr >> b & 1u) ? gr : gd;
                    atomicMax(&samples[(size_t)rec * NSAMP + slot], hk | __float_as_uint(g));
                }
            }
        }
    }
}

__device__ __forceinline__ void bitonic_sort(float* s, int tid)
{
    for (int k = 2; k <= NSAMP; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                float a = s[tid], b = s[ixj];
                if ((a > b) == ((tid & k) == 0)) { s[tid] = b; s[ixj] = a; }
            }
            __syncthreads();
        }
}

extern "C" __global__ void __launch_bounds__(NSAMP)
mad_stats(unsigned long long* __restrict__ samples,   // [R, NSAMP]; consumed and reset
          float* __restrict__ stats,                  // [R, 4] = {threshold(-1 = fallback), median, mad, n}
          float k_mad, float min_threshold, int min_samples)
{
    __shared__ float s[NSAMP];
    const int rec = blockIdx.x, tid = threadIdx.x;
    unsigned long long key = samples[(size_t)rec * NSAMP + tid];
    samples[(size_t)rec * NSAMP + tid] = 0ull;
    float g = key ? __uint_as_float((unsigned int)(key & 0xffffffffu)) : BIG;
    s[tid] = g;
    int n = __syncthreads_count(g < BIG);

    float thr = -1.f, med = 0.f, mad = 0.f;
    if (n >= min_samples) {
        bitonic_sort(s, tid);
        med = (n & 1) ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]);
        __syncthreads();
        s[tid] = (g < BIG) ? fabsf(g - med) : BIG;
        __syncthreads();
        bitonic_sort(s, tid);
        mad = (n & 1) ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]);
        thr = fmaxf(med + k_mad * 1.4826f * mad, min_threshold);
    }
    if (tid == 0) {
        stats[rec * 4 + 0] = thr; stats[rec * 4 + 1] = med;
        stats[rec * 4 + 2] = mad; stats[rec * 4 + 3] = (float)n;
    }
}

extern "C" __global__ void __launch_bounds__(TX * TY)
grow(const unsigned int* __restrict__ orig,
     const unsigned int* __restrict__ core,
     const float* __restrict__ depth,
     const float* __restrict__ stats,
     const int* __restrict__ table,
     unsigned char* __restrict__ out,     // [R, H, W]
     int R, int H, int W)
{
    __shared__ float sd[SH][SW];
    __shared__ unsigned int so[SH][SW];
    __shared__ unsigned int sa[2][SH][SW];
    __shared__ float thr[MAXB];
    __shared__ int s_rec_view[MAXB], s_rec_bit[MAXB];

    const int v = blockIdx.z;
    const int tid = threadIdx.y * TX + threadIdx.x;
    const int HW = H * W;
    if (tid < R) { s_rec_view[tid] = table[tid]; s_rec_bit[tid] = table[R + tid]; }
    if (tid < MAXB) {
        int rec = table[2 * R + v * MAXB + tid];
        thr[tid] = (rec >= 0) ? stats[rec * 4] : -1.f;
    }
    __syncthreads();
    unsigned int fallback = 0u;              // bits whose stats are unreliable -> keep input mask
    for (int b = 0; b < MAXB; ++b) if (thr[b] < 0.f) fallback |= 1u << b;

    const int x0 = blockIdx.x * TX - HALO, y0 = blockIdx.y * TY - HALO;
    const float* dview = depth + (size_t)v * HW;
    const unsigned int* oview = orig + (size_t)v * HW;
    const unsigned int* cview = core + (size_t)v * HW;
    for (int i = tid; i < SW * SH; i += TX * TY) {
        int lx = i % SW, ly = i / SW;
        int x = x0 + lx, y = y0 + ly;
        float d = 0.f; unsigned int o = 0u, c = 0u;
        if (x >= 0 && x < W && y >= 0 && y < H) {
            int p = y * W + x;
            d = dview[p]; o = oview[p]; c = cview[p] | (o & fallback);
        }
        sd[ly][lx] = d; so[ly][lx] = o; sa[0][ly][lx] = c;
    }
    __syncthreads();

    int cur = 0;
    for (int it = 0; it < EW; ++it) {
        int nxt = cur ^ 1;
        for (int i = tid; i < SW * SH; i += TX * TY) {
            int lx = i % SW, ly = i / SW;
            unsigned int a = sa[cur][ly][lx];
            unsigned int cand = so[ly][lx] & ~a;
            if (cand && lx > 0 && lx < SW - 1 && ly > 0 && ly < SH - 1) {
                float d = sd[ly][lx];
                unsigned int nb[4] = { sa[cur][ly][lx - 1], sa[cur][ly][lx + 1], sa[cur][ly - 1][lx], sa[cur][ly + 1][lx] };
                float dd[4] = { fabsf(d - sd[ly][lx - 1]), fabsf(d - sd[ly][lx + 1]),
                                fabsf(d - sd[ly - 1][lx]), fabsf(d - sd[ly + 1][lx]) };
                while (cand) {
                    int b = __ffs(cand) - 1;
                    cand &= cand - 1;
                    float t = thr[b];
                    bool ok = false;
#pragma unroll
                    for (int q = 0; q < 4; ++q) ok |= ((nb[q] >> b & 1u) && dd[q] < t);
                    if (ok) a |= 1u << b;
                }
            }
            sa[nxt][ly][lx] = a;
        }
        __syncthreads();
        cur = nxt;
    }

    const int x = blockIdx.x * TX + threadIdx.x, y = blockIdx.y * TY + threadIdx.y;
    if (x >= W || y >= H) return;
    const unsigned int a = sa[cur][threadIdx.y + HALO][threadIdx.x + HALO];
    const int p = y * W + x;
    for (int r = 0; r < R; ++r)
        if (s_rec_view[r] == v) out[(size_t)r * HW + p] = (unsigned char)((a >> s_rec_bit[r]) & 1u);
}
"""


class DepthBoundaryFilter:
    """Batched, CUDA-resident depth-boundary filter (see module docstring).

    User-facing parameters: ``erosion_width_px`` (compile-time, JIT per value),
    ``mad_multiplier`` (k), ``min_threshold_m`` (floor for T; 0 disables).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        erosion_width_px: int = 5,
        mad_multiplier: float = 5.0,
        min_threshold_m: float = 0.0,
    ) -> None:
        if erosion_width_px < 1:
            raise ValueError("erosion_width_px must be >= 1")
        self.device = torch.device(device)
        self.erosion_width_px = int(erosion_width_px)
        self.mad_multiplier = float(mad_multiplier)
        self.min_threshold_m = float(min_threshold_m)
        module = Module(_SOURCE, name="depth_boundary_filter.cu",
                        defines={"EW": self.erosion_width_px}, device=self.device)
        self._erode = module.get("erode_pack")
        self._stats = module.get("mad_stats")
        self._grow = module.get("grow")
        self._shape: tuple[int, int, int, int] | None = None   # (R, V, H, W) capacity
        self._orig: torch.Tensor | None = None
        self._core: torch.Tensor | None = None
        self._samples: torch.Tensor | None = None
        self._stats_buf: torch.Tensor | None = None
        self._out: torch.Tensor | None = None
        self._table_host: torch.Tensor | None = None
        self._table_dev: torch.Tensor | None = None

    # ------------------------------------------------------------------ buffers
    def _ensure(self, R: int, V: int, H: int, W: int) -> None:
        need = (R, V, H, W)
        if self._shape is not None and all(a <= b for a, b in zip(need, self._shape)):
            return
        R = max(R, self._shape[0] if self._shape else 0)
        V = max(V, self._shape[1] if self._shape else 0)
        H = max(H, self._shape[2] if self._shape else 0)
        W = max(W, self._shape[3] if self._shape else 0)
        dev = self.device
        self._orig = torch.empty((V, H, W), dtype=torch.int32, device=dev)
        self._core = torch.empty((V, H, W), dtype=torch.int32, device=dev)
        self._samples = torch.zeros((R, NSAMP), dtype=torch.int64, device=dev)
        self._stats_buf = torch.zeros((R, 4), dtype=torch.float32, device=dev)
        self._out = torch.empty((R, H, W), dtype=torch.uint8, device=dev)
        table_len = 2 * R + V * MAX_PER_VIEW
        self._table_host = torch.empty(table_len, dtype=torch.int32, pin_memory=True)
        self._table_dev = torch.empty(table_len, dtype=torch.int32, device=dev)
        self._shape = (R, V, H, W)

    # ------------------------------------------------------------------ API
    def __call__(
        self,
        masks_u8: torch.Tensor,          # [R, H, W] uint8 0/1, CUDA, contiguous
        record_views: Sequence[int],     # len R, view index of each record
        depth: torch.Tensor,             # [V, H, W] float32, CUDA, contiguous (metres, 0/NaN invalid)
    ) -> torch.Tensor:
        """Returns filtered masks ``[R, H, W]`` uint8 (a view into a persistent buffer)."""
        R, H, W = (int(s) for s in masks_u8.shape)
        V = int(depth.shape[0])
        if R == 0:
            return masks_u8
        assert masks_u8.dtype == torch.uint8 and masks_u8.is_contiguous() and masks_u8.is_cuda
        assert depth.dtype == torch.float32 and depth.is_contiguous() and tuple(depth.shape[1:]) == (H, W)
        self._ensure(R, V, H, W)
        assert self._table_host is not None and self._table_dev is not None

        table = self._table_host[: 2 * R + V * MAX_PER_VIEW].numpy()
        table[2 * R:] = -1
        per_view = [0] * V
        for r, view in enumerate(record_views):
            view = int(view)
            bit = per_view[view]
            if bit >= MAX_PER_VIEW:
                raise ValueError(f"depth boundary filter supports at most {MAX_PER_VIEW} instances per view")
            per_view[view] += 1
            table[r] = view
            table[R + r] = bit
            table[2 * R + view * MAX_PER_VIEW + bit] = r
        self._table_dev[: table.shape[0]].copy_(self._table_host[: table.shape[0]], non_blocking=True)

        grid = ((W + TX - 1) // TX, (H + TY - 1) // TY, V)
        orig, core = self._orig, self._core
        samples, stats, out = self._samples, self._stats_buf, self._out
        assert orig is not None and core is not None and samples is not None and stats is not None and out is not None
        # Buffers may be larger than (V,H,W); kernels index with the *current* H/W, so pass
        # contiguous views of exactly the needed size.
        orig_v = orig.view(-1)[: V * H * W]
        core_v = core.view(-1)[: V * H * W]
        out_v = out.view(-1)[: R * H * W]
        self._erode.launch(grid, (TX, TY), masks_u8, depth, self._table_dev, orig_v, core_v, samples, R, H, W)
        self._stats.launch((R,), (NSAMP,), samples, stats, self.mad_multiplier, self.min_threshold_m, MIN_SAMPLES)
        self._grow.launch(grid, (TX, TY), orig_v, core_v, depth, stats, self._table_dev, out_v, R, H, W)
        return out_v.view(R, H, W)

    def last_stats(self) -> torch.Tensor | None:
        """``[R, 4]`` = {threshold (-1 = fallback to input), median, MAD, n_samples} of the last call."""
        return self._stats_buf
