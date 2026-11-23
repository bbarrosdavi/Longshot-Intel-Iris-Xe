# Intel Iris Xe / Gen12 Vulkan Optimization Guide (`opt/iris-xe`)

This document summarizes the engineering work, architectural findings, and empirically validated optimizations developed for **Longshot** on **Intel Iris Xe (Alder Lake GT2, Gen12 / Xe-LP, 96 EUs, 26W TDP, Mesa ANV 26.2.4, Linux Wayland)**, targeting both **maximum playability at high resolution (`1787×1005 → 1920×1080` FSR 3.1 + 100% sharpness)** and **minimum end-to-end input latency** with Frame Generation (`Auto` and `2x`).

All vendor-specific heuristics are scoped to `VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA` (or gated by runtime switches in `gpu/vk/switches.def`) so NVIDIA and AMD paths remain unaffected.

---

## 1. Empirical Progression (Unoptimized Upstream → Functional Baseline → Final Optimized Build)

Measured on **Intel Core i5-1235U / Iris Xe 96 EUs (26W)**, 16 GB LPDDR4x UMA, EndeavourOS Linux, Mesa ANV 26.2.4, `1920×1080 @ 60 Hz` (`vsync=2` MAILBOX). Final max-resolution benchmarks run in **Hyrule Castle Courtyard (`slot_01`)** at **`1787×1005 → 1920×1080`**, **AMD FSR 3.1 (`dlss=3`, `resolution=9`)**, **100% Sharpness**:

| Milestone | Base FPS | Displayed FPS | GPU / Frame | FrameGen Handover | Visual & Stability State |
|---|---:|---:|---:|---:|---|
| **Unoptimized Upstream** (`nightly-469-9af4808f` / `main`) | `~9` | `~9` | `>110 ms` | N/A | **Unplayable:** severe black/rainbow surface corruption, i915 10s GPU hangs/TDR |
| **First Functional Baseline** (early Batch 5–10) | `~13` | `~13–24` | `~75 ms` | `>160 ms` | First bootable/non-hanging pass; low-res/blocky, Gen12 spill neon/white blow-out, hair fringes, `~7 ms` optical-flow FG |
| **Batch 14 (Visual Stability Checkpoint)** | `14.0` | `28.0` (2x) | `53.9 ms` | `156.9 ms` (`BATCH=1`) | Clean visuals (`0` neon, `0` hair fringes, `0` spikes), high input lag |
| **Batch 15 Final — FG Auto** | **`20.56`** (**2.3×** vs upstream / **+58%** vs func. baseline) | **`41.1`** | **`41.8–42.8 ms`** | **`35.1 ms`** (`0 late`) | Clean across Kokiri, Market & Castle; `ahead=1` saves extra `~50 ms` input queue |
| **Batch 15 Final — FG 2x** | **`20.1`** | **`40.2`** | **`42.2 ms`** | **`65.5 ms`** (`0 late`) | Clean across all areas, `0 <12 ms` bursts, `k0=25.5 / k1=24.0 ms` |

### Visual Comparison (Before vs. After)

| Before — Unoptimized Upstream (`~9 FPS`) | After — Optimized Build (`20.6 Base / 41.1 Displayed FPS`, Kokiri Forest) |
|:---:|:---:|
| ![Unoptimized upstream 9 FPS](images/iris-xe/01_baseline_unoptimized_kokiri_9fps.jpg) | ![Optimized Kokiri Forest](images/iris-xe/04_optimized_kokiri_clean.jpg) |

| Before — First Functional Baseline (`~13 FPS`) | After — Optimized Build (`20.6 Base / 41.1 Displayed FPS`, Hyrule Castle Path) |
|:---:|:---:|
| ![First functional baseline 13 FPS](images/iris-xe/02_functional_baseline_deku_13fps.jpg) | ![Optimized Castle Wagon](images/iris-xe/06_optimized_castle_wagon_clean.jpg) |

**3D-MV FrameGen Real vs. Synthesized (`_made`) Frames in Active Sprint + Camera Pan:**

![Real vs Synthesized FrameGen frames](images/iris-xe/07_framegen_3dmv_real_vs_made.jpg)

### Per-Pass GPU Breakdown (`RX_VK_PASS_TIMING=2`, `1787×1005 → 1920×1080`)

| Pass / Shader | Before (Batch 14) | After (Batch 15 Final) | Gen12 ISA / Notes |
|---|---:|---:|---|
| `fsr3 accumulate` + `dlss_out.comp` | `3.60 ms` + `~1.00 ms` | **`2.18 ms`** (`dlss_out` eliminated) | FP16 + 10×10 packed `uvec2` SLM + `kSquareOutput` SpecId 0 (`SIMD8 19,094 cyc` → **`SIMD16 8,899 cyc`, `0:0` spills**) |
| `fsr3 prepare_reactivity` | `1.50 ms` | **`0.96 ms`** (-36%) | FP16 + `2 textureGather + 2 texelFetch`, `kHasReactive` SpecId 0 (`42 → 25` sends, `SIMD16`, `0:0` spills) |
| `fsr3 prepare_inputs` | `1.00 ms` | **`0.83 ms`** (-17%) | FP16 + `2 textureGather + 2 texelFetch` (`31 → 26` sends, `SIMD16`, `0:0` spills) |
| `fsr3 luma_instability` | `0.86 ms` | **`0.69 ms`** (-20%) | FP16, removed cancelled exposure & dead depth fetches (`SIMD16`, `0:0` spills) |
| `fg_rcas.comp` (runs 2×–4×/frame at 1080p) | `12,066 cyc` (SLM + barrier) | **`2,148 cyc`** (**5.6×** fewer cycles) | Direct 5-tap L1 FP16 `texelFetch` (`0` loops, `0` barriers, `SIMD16`, `0:0` spills) |
| `26fe0c70ab9fc547` CS (compute hot shader) | `1.57 ms` | **`0.82 ms`** (**1.91×** faster) | Custom SPIR-V (`kCustomSpv_26fe0c70`): `SIMD8 38,364 cyc` → **`SIMD16 15,530 cyc`, `0:0` spills** |
| Built-in FrameGen (`fg_copy_hud` + `fg_synth2_mv`) | `~7.0 ms` (optical flow) | **`~0.70 ms`** (**~10×** faster) | Exact 3D camera + actor motion vectors + depth + fused 2-frame synthesis |

---

## 2. Key Optimization Pillars

### A. Shader Recompiler & SPIR-V Passes (`gpu/vk/eden/`, `gpu/vk/vk_draw.cpp`, `gpu/vk/vk_custom_spv.h`)
1. **BDA → SSBO Lowering (`RX_VK_BDA_SSBO=1` default on Mesa Intel):** Lowers `PhysicalStorageBuffer` page-table helper calls backed by UBO-aliased storage buffers to direct SSBO `OpAccessChain` loads/stores/atomics, cutting register pressure and preventing Intel i915 GPU hangs.
2. **Warp-32 & Subgroup Support:** Native `REDUX_nx2`/`R2UR_nx2`, `EliminateUniformWarpOps`, 1-barrier SIMD8-assisted `wg_warp32` with divergent-`if` linearization, and `DerivativeGroupLinearKHR`.
3. **Mesh Fallback Batching & Lossless Culling (`RX_VK_MFB_BATCH=4`):** Batches mesh-fallback compute draws, preserves `pending_clear`, and culls degenerate/backface/frustum triangles losslessly.
4. **Gen12 Register Pressure & Custom SPIR-V Overrides (`RX_VK_CUSTOM_SPV=1`):**
   - `48590e0b8589d6ca` CS (SLM reduction, `-29%`)
   - `e40dc98fca54ad01` FS (`SIMD16` unlock)
   - `26fe0c70ab9fc547` CS (`SIMD8 → SIMD16`, `0:0` spills, `1.57 ms → 0.82 ms`)
   - `MINREG_SCHED` and `COALESCE_GUARD` on giant VS/CS basic blocks.
5. **Prepass `EQUAL` Depth Bias Fix:** Keeps `equal_depth_bias = 64.0f` on opaque `EQUAL` draws (eliminating distant "white forest" flashing hills under FSR 3.1 jitter) while disabling it on alpha-tested/blended draws (eliminating black hair-card and foliage fringes).

### B. FP16 FSR 3.1 & Fused Perceptual-to-Linear Output (`gpu/vk/vk_fsr3.cpp`, `gpu/vk/vk_temporal.cpp`, `gpu/vk/shaders/fsr3_*`)
1. Rewrote `fsr3_accumulate`, `fsr3_prepare_reactivity`, `fsr3_prepare_inputs`, and `fsr3_luma_instability` in true FP16 (`GL_EXT_shader_explicit_arithmetic_types`, `SIMD16`, `0:0` spills).
2. **Fused `P_ACCUMULATE_SQ` (`kSquareOutput` SpecId 0):** Squares the perceptual RGB output (`c * c`) directly into the game's `VK_FORMAT_B10G11R11_UFLOAT_PACK32` target via `StorageImageWriteWithoutFormat`, completely eliminating `dlss_out.comp` and saving **33.2 MB/frame** of `RGBA16F` roundtrip bandwidth.
3. **Exact HUD Diff Preservation:** Exposes `fsr3::last_upscaled_view()` (`g.internal_upscaled[g.parity].view`) so FrameGen's `fg_copy_hud` still performs exact pre/post-HUD alpha masking with zero extra copies.
4. **L1 Direct FP16 RCAS (`gpu/vk/shaders/fg_rcas.comp`):** Replaced the `10×10` SLM + barrier RCAS pass with a 5-tap L1 FP16 `texelFetch` pass (`5.6×` fewer Gen12 cycles).

### C. Artifact-Free 3D-MV Frame Generation & Low Input Latency (`gpu/vk/vk_framegen.cpp`, `gpu/vk/vk_replay.cpp`)
1. **Universal 3D-MV Synthesis (`fg_synth_mv.comp`, `fg_synth2_mv.comp`):** Uses exact 3D camera + skinned-actor motion vectors + depth in both `Auto` (`frame_generation=1`) and fixed `2x/3x` modes, with trajectory consistency gating and safe unwarped `near` fallback (preventing character/shield/shadow shredding during sprints and fast camera pans).
2. **Exact HUD Alpha Masking in Auto & `dlss=0`:** Enabled `exact_hud` (`fg_copy_hud`) and fused `fg_synth2_mv` in `Auto` mode as well as `2x/3x`.
3. **Low-Latency Pacing & Queue Control:**
   - **`RX_VK_BATCH=0` on Intel ANV:** Batched submits delayed `RX_VK_READBACK_EARLY` writebacks, causing `+20.5 ms` timeline waits, `+70 ms` handover latency, and `-18%` FPS. Disabling batching on ANV recovers both throughput and latency.
   - **1-Frame In-Flight Depth (`ahead=1`) on Mesa Intel FrameGen:** Waits on frame `N-1` right after submitting frame `N`, overlapping `~14 ms` CPU input+recording inside `~42 ms` GPU execution with zero FPS loss and `-50 ms` end-to-end input queue depth.
   - **Unconditional Backlog Drain + `slack_ns` Phase Convergence:** Drains `q_now >= 1` immediately and smoothly trims dead-air phase slack (`slack_ns > 3 ms`) so both `Auto` and `2x` converge to minimum handover latency with `0 late` frames and `0 <12 ms` bursts.

### D. Host CPU & Unified Memory (`host/nvdrv.c`, `host/rx_backend.c`, `gpu/vk/vk_memory.cpp`)
1. Dedicated Vulkan recorder thread (`wait` + `20 µs` spin) and P-core pinning (`HOS_RESERVE_CORE_MASK=0x3`).
2. AVX2 vectorized dirty-page scanning (`collect_in`) + adaptive `256 KB` range merging (`hos_gpu_guest_ranges`).
3. Hardware fast-clear fusion (`VK_ATTACHMENT_LOAD_OP_CLEAR`), `CCS_E` compression preservation via explicit `VkImageFormatListCreateInfo`, and conservative UMA VRAM budgeting (`RX_VK_VRAM_MB=3600`, `RX_VK_WORKERS=2`).

---

## 3. Documented Dead-Ends (What NOT to Enable on Intel Gen12 / Mesa ANV)
- **`shader_spilling_rate=2` (Mesa ANV 26.x):** Degrades Gen12 register allocation and triggers a hardware spill corruption bug (`4×2` neon blocks on heavy PBR fragment shaders). Always keep Mesa default (`11`).
- **`RX_VK_BDA_HOIST=1` (Default OFF):** Without a per-call fallback, dynamic offsets crossing chunked/sparse pool boundaries (`vk_memory.cpp`) read `zero_page` and corrupt materials in dense areas (e.g., Castle Town Market `slot_05`). With per-call `OpSelectionMerge` fallback, Mesa NIR inlines `guest_load_*` 130+ times per shader and triggers OOM during compilation.
- **`RX_VK_BATCH=1` on Intel ANV:** Delays readback/writeback completion and adds `+70 ms` latency.
- **Stripping `NoContraction`/`SignedZeroInfNanPreserve` on heavy PBR FS:** Breaks constant promotion on Gen12 and doubles register spills (`57:124 → 110:185`).

---

## 4. Visual QA Verification
Verified 1:1 via captured PNGs (both real and synthesized `_made` frames) across:
- **Hyrule Castle Courtyard (`slot_01`)** — still, sprint, and camera pan in `Auto` and `2x` (`dlss=3` and `dlss=0`).
- **Hyrule Castle Town Market (`slot_05`)** — dense banners, windows, planters, guard armor, and distant castle.
- **Malon's Wagon / Castle Path (`slot_00`)** — cliff rock geometry, foliage, and milk cans (zero polygon spikes, zero neon bloom, zero hair fringes).
