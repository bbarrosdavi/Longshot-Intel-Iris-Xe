# perf(vk): Intel Iris Xe / Gen12 (Mesa ANV) optimizations, ANV PointCoord SBE fix, iGPU UMA accounting, FP16 FSR 3.1 & 3D-MV FrameGen

## Summary

On **Intel Iris Xe / Gen12 integrated GPUs (Linux Mesa ANV 26.2.4, 16 GB UMA)**, upstream `main @ 209389fa` exhibits four major blockers during gameplay:
1. **Severe throughput & latency bottlenecks** on 96-EU Gen12 (mesh fallback barriers, SIMD8 register spills, FP32 FSR 3.1 + `dlss_out.comp` roundtrip, SLM-heavy RCAS, and 64-bit guest page-table indexing).
2. **Rapid OOM crashes on 16 GB unified-memory laptops:** `system_heap_usage()` in `gpu/vk/vk_vma.cpp` skipped unified-memory single-heap iGPUs, preventing `imports_give_back()` and Linux memory-pressure eviction from unpinning retired DMA-BUF imports, while per-thread `TLS_OBJECT` fallback slots duplicated `~1.40 GB` across `Presentation Thread` and `host`.
3. **Blinding yellow-white sky/sun screen blowout on Mesa ANV 26.2.4:** After any shader reading `BuiltIn PointCoord` (e.g., simulated-turf shadow) draws in a `VkCommandBuffer`, Mesa ANV leaks `3DSTATE_SBE::PointSpriteTextureCoordinateEnable = 1` onto subsequent pipelines in that same command buffer, overwriting `Location 0` with `PointCoord.x = 0.5` on subsequent `POINT_LIST` draws. In OoT's 1×1 sun visibility pass (`edd4d7c51103645d`, 25 additive points), `G` accumulated to `12.5` instead of `0..1`, blowing out all 11 sun lens-flare draws across the entire viewport.
4. **Interior white bloom blow-out:** Fragment outputs and `OpImageWrite` receiving `±Inf` clamped to `+65000.0` instead of `0.0`, blowing out interior props (e.g. Link's House).

This branch (rebased onto `upstream/main @ 209389fa`, head `79beef54`) fixes all four issues and lifts Courtyard throughput by **`+35.5%`** (`17.2 → 23.3 Base FPS`), cuts `other buffers` by **`-1.04 GiB`** (`1,562 → 493–519 MB`), keeps DMA-BUF imports steady at `~1.1–1.5 GB` (`0` retired still imported), eliminates the sky blowout with zero shader-cache invalidation, and adds low-latency 3D-MV FrameGen (`Auto`/`2x`/`3x`). All Intel-specific heuristics are scoped to `VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA` or runtime switches in `gpu/vk/switches.def` so NVIDIA and AMD paths are preserved.

**Key files to review first:**
- `gpu/vk/vk_draw.h`, `gpu/vk/vk_draw.cpp`, `gpu/vk/vk_replay.cpp` — Mesa ANV `3DSTATE_SBE` `PointCoord` tracking + `split_part_forced()`, shared `FallbackBuffers` pool under `replay_mu`, narrowed MFB/compute barriers, `ahead=1` FrameGen queue.
- `gpu/vk/vk_vma.cpp` — single-heap iGPU accounting in `system_heap_usage()` so `imports_give_back()` / Linux pressure unpins retired DMA-BUF imports.
- `gpu/vk/device_policy.h`, `gpu/vk/vk_texture.cpp`, `gpu/vk/switches.def` — `bc_small_runs` on Mesa ANV, `MFB_BATCH=2` cap on `small_vram_textures`, narrowed compute dispatch barriers.
- `gpu/vk/vk_fsr3.cpp`, `gpu/vk/vk_temporal.cpp`, `gpu/vk/vk_framegen.cpp`, `gpu/vk/shaders/fg_*` — FP16 FSR 3.1 + fused `P_ACCUMULATE_SQ`, 5-tap L1 FP16 RCAS, universal 3D-MV FrameGen & pacing.
- `gpu/vk/eden/shader_recompiler/` — `±Inf`/`NaN` guard (`fng6`), 32-bit guest page-table lookup (`gp35`), `BDA→SSBO` entry-function scoping, warp-32 (`wg_warp_plan.h`), and custom SPIR-V overrides (`vk_custom_spv.h`).

---

## Gameplay Showcase (Live 1080p Capture on Intel Iris Xe)

Recorded live on **Intel Core i5-1235U / Iris Xe 96 EUs (26W PL1, 16 GB LPDDR4x UMA, EndeavourOS Linux Wayland, Mesa ANV 26.2.4)** at `1920×1080` with integrated 2× 3D-MV FrameGen via zero-copy VAAPI (`wf-recorder -c h264_vaapi -d /dev/dri/renderD128 -p qp=20`):

[![Watch 1080p Live Showcase on Intel Iris Xe — 5 Areas](https://raw.githubusercontent.com/bbarrosdavi/Longshot-Intel-Iris-Xe/main/docs/images/iris-xe/08_iris_xe_showcase_thumb.jpg)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/longshot_iris_xe_showcase_1080p.mp4)

- 🎬 **[Full 1080p 5-Area Compilation (55s MP4)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/longshot_iris_xe_showcase_1080p.mp4)**
- 🎬 **[Compact 720p Preview (<10 MB MP4)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/longshot_iris_xe_showcase_720p_under10mb.mp4)**
- Individual 20s 1080p clips:
  1. [Hyrule Market (22–25 FPS)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/01_hyrule_market.mp4)
  2. [Kakariko Village — Clean Sky/Sun (20–21 FPS)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/02_kakariko_village.mp4)
  3. [Hyrule Field (25 FPS)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/03_hyrule_field.mp4)
  4. [Kokiri Forest (20–22 FPS)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/04_kokiri_forest.mp4)
  5. [Zora's Domain (up to 25 FPS)](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/releases/download/v2.0-iris-xe/05_zoras_domain.mp4)

### Before vs. After Visual Comparison

| Before — Unoptimized Upstream (`~9 FPS`) | After — Optimized Build (`23.3 Base / 46+ FG FPS`, Kokiri Forest) |
|:---:|:---:|
| ![Unoptimized upstream 9 FPS](https://raw.githubusercontent.com/bbarrosdavi/Longshot-Intel-Iris-Xe/main/docs/images/iris-xe/01_baseline_unoptimized_kokiri_9fps.jpg) | ![Optimized Kokiri Forest](https://raw.githubusercontent.com/bbarrosdavi/Longshot-Intel-Iris-Xe/main/docs/images/iris-xe/04_optimized_kokiri_clean.jpg) |

| Before — First Functional Baseline (`~13 FPS`) | After — Optimized Build (`Hyrule Castle Path`) |
|:---:|:---:|
| ![First functional baseline 13 FPS](https://raw.githubusercontent.com/bbarrosdavi/Longshot-Intel-Iris-Xe/main/docs/images/iris-xe/02_functional_baseline_deku_13fps.jpg) | ![Optimized Castle Wagon](https://raw.githubusercontent.com/bbarrosdavi/Longshot-Intel-Iris-Xe/main/docs/images/iris-xe/06_optimized_castle_wagon_clean.jpg) |

---

## Testing

**Tested:**
- **Hardware / OS:** EndeavourOS Linux (Wayland / Niri), Intel Core i5-1235U / Iris Xe (Alder Lake GT2, Gen12 / Xe-LP, 96 EUs, 26W PL1, 16 GB LPDDR4x UMA), Mesa ANV 26.2.4, `1920×1080 @ 60 Hz` (`vsync=2` MAILBOX).
- **Live gameplay verification:** Played and recorded across 5 distinct areas (Hyrule Market, Kakariko Village, Hyrule Field, Kokiri Forest, Zora's Domain) — zero crashes, zero GPU hangs, zero sky blowout during 360° camera pans, zero interior bloom spikes, and steady memory footprint (`~1.1–1.5 GB` active imports, `0` retired still imported).
- **ANV `3DSTATE_SBE` `PointCoord` reproduction & verification:** Confirmed the Mesa ANV 26.2.4 `PointSpriteTextureCoordinateEnable` leak with a standalone Vulkan repro + `INTEL_DEBUG=bat`, and verified across a headless 360° Kakariko sky sweep (`edd4d7c51103645d` mean `12.5 → 1.015`, step-10 frame RGB mean `[241.4, 232.9, 210.0] → [118.3, 116.0, 163.9]`, `0` shader-cache invalidation, `0.0 ms` added waits) plus live confirmation.
- **Unit & GPU checks (`RX_RELEASE=1 LONGSHOT_FSR=1`):**
  - Passing: `test_pass_barrier`, `test_mesh_fallback_global_attrs`, `test_mesh_fallback_shared_warp`, `test_present_thread`, `test_linux_pressure`, `test_device_policy`, `test_pcache_load`, `test_vendor_translation`, `test_vendor_fixes`, `test_astc_decoder_gpu.py --gpu`, `test_fallback_buffer_cleanup.py`, `test_mesh_fallback_capacity.py`, `test_quick_pipelines.py`, `test_vram_held_idle.py`.
  - `spirv-val --target-env vulkan1.2` + `INTEL_DEBUG=cs` ISA audit (`SIMD16`, `0:0` spills) on all new/modified compute shaders.
  - `third_party/xess/` verified completely untouched vs `upstream/main`.

**Still to test:**
- **Windows Intel Vulkan ICD (Iris Xe / Arc):** Verify host build (`llvm-mingw`) and runtime paths where Mesa-only guards are bypassed (`@Beanstalk` / by hand).
- **NVIDIA & AMD (Tier 1 GPUs, Windows & Linux):** Verify no regression in default non-Intel codepaths, FSR 3.1, and 3D-MV FrameGen (`@Beanstalk`).
- **Rebase onto latest `upstream/main` (`be663a08+`):** Branch is currently based on `209389fa` and can be split into topic-scoped PRs (e.g., ANV `3DSTATE_SBE` `PointCoord` fix, single-heap iGPU UMA `system_heap_usage()` fix, narrowed fallback barriers, FP16 FSR 3.1, 3D-MV FrameGen) per maintainer preference.

---

## Measurements

**Verdict:** **`+35.5%` Base FPS** in Hyrule Castle Courtyard (`17.2 → 23.3 fps`, **`2.6×`** vs unoptimized `~9 fps`), **`-18.3%`** worst-case frame time in Kakariko (`107.9 → 88.2 ms`), **`-1.04 GiB`** fallback buffer VRAM (`1,562 → 493–519 MB`), **`0`** retired DMA-BUF imports leaked, and **100% elimination** of Mesa ANV `3DSTATE_SBE` sky blowout (`edd4d7c51103645d`: `12.5 → 1.015`).

Setup: Intel Core i5-1235U / Iris Xe 96 EUs (26W), 16 GB LPDDR4x UMA, EndeavourOS Linux, Mesa ANV 26.2.4, `1440×810 → 1080p` (unprofiled 3-way, FSR 3.1, FG Off, uncapped, `PASS_TIMING=0`) and `1787×1005 → 1920×1080` (pass breakdown & progression).

### 1. 3-Way Unprofiled Comparison (`1440×810 → 1080p`, FSR 3.1, FG Off)

| Metric | Before: `upstream/main @ 209389fa` | Intermediate: `v1.0-iris-xe @ d56d9a3a` | After: This Branch (`79beef54`) |
|---|---:|---:|---:|
| **Courtyard (`slot_01`) Steady Base FPS** | `17.2 fps` | `20.8 fps` | **`23.3 fps`** (**`+35.5%`** vs main, **`+12.0%`** vs v1) |
| **Courtyard Slowest Frame (600f)** | `113.7 ms` (13.8 s stall) | `89.6 ms` | **`86.9 ms`** |
| **Courtyard Replay Submit (`queue submit`)** | — | `15.04 ms` (`2.79 ms`) | **`14.04 ms` (`0.00 ms`)** |
| **Courtyard Full Barriers / Frame** | `518.3` | `310.1` | **`406.4`** (`-112` vs main) |
| **Courtyard Draws Skipped** | `140` (black 3D) | `0` | **`0`** |
| **Kakariko Gate (`slot_00`) Steady FPS** | — | `17.4` (`16.85` avg) | **`17.3`** (`16.91` avg) |
| **Kakariko Slowest Frame (600f)** | — | `107.9 ms` | **`88.2 ms`** (**`-18.3%`**) |
| **Fallback `other buffers` VRAM** | `523 MB` (1,562 MB w/ TLS dup) | `4,077 MB` total | **`493–519 MB`** (**`-1.04 GiB`** vs TLS dup) |
| **Active DMA-BUF Imports (16 GB UMA)** | Unbounded growth → OOM crash | — | **`~1.1–1.5 GB`** (`0` retired still imported) |
| **Kakariko Sky Pass `edd4d7c51103645d` Mean** | `12.5` (yellow-white screen wash) | — | **`1.015`** (`step 10` RGB `[241,233,210] → [118,116,164]`) |

### 2. Per-Pass GPU Breakdown (`RX_VK_PASS_TIMING=2`, `1787×1005 → 1920×1080`)

| Pass / Shader | Before | After | Gen12 ISA / Notes |
|---|---:|---:|---|
| `fsr3 accumulate` + `dlss_out.comp` | `3.60 ms` + `~1.00 ms` | **`2.18 ms`** (`dlss_out` removed) | FP16 + 10×10 packed `uvec2` SLM + `kSquareOutput` SpecId 0 (`SIMD8 19,094 cyc → SIMD16 8,899 cyc`, `0:0` spills, `-33.2 MB/frame`) |
| `fsr3 prepare_reactivity` | `1.50 ms` | **`0.96 ms`** (`-36%`) | FP16 + `2 textureGather + 2 texelFetch` (`42 → 25` sends, `SIMD16`, `0:0` spills) |
| `fsr3 prepare_inputs` | `1.00 ms` | **`0.83 ms`** (`-17%`) | FP16 (`31 → 26` sends, `SIMD16`, `0:0` spills) |
| `fsr3 luma_instability` | `0.86 ms` | **`0.69 ms`** (`-20%`) | FP16, removed dead depth/exposure work (`SIMD16`, `0:0` spills) |
| `fg_rcas.comp` (2×–4×/frame @ 1080p) | `12,066 cyc` (SLM + barrier) | **`2,148 cyc`** (**`5.6×`** fewer) | Direct 5-tap L1 FP16 `texelFetch` (`0` loops/barriers, `SIMD16`, `0:0` spills) |
| `26fe0c70ab9fc547` CS (hot compute) | `1.57 ms` | **`0.82 ms`** (**`1.91×`** faster) | Custom SPIR-V (`kCustomSpv_26fe0c70`): `SIMD8 38,364 → SIMD16 15,530 cyc`, `0:0` spills |
| `3de6d86ec7e6720e` (`sim_turf`) | `0.946 ms` | **`<0.33 ms`** | Skip mirror when `target <= stock` (`-15 MB` VRAM, `-13.7 MB/frame`) |
| `m fallback` / `m dispatch` | Baseline | **`-14.7%`** / **`-9.9%`** | Narrowed internal `fallback_record()` barriers (`-40` full barriers/frame) + `RX_VK_GP_FAST32` (`gp35`) |
| Built-in FrameGen (`fg_copy_hud` + `fg_synth2_mv`) | `~7.0 ms` (optical flow) | **`~0.70 ms`** (**`~10×`** faster) | 3D camera + actor MVs + depth + fused 2-frame synthesis, `ahead=1` (`-50 ms` input queue) |

---

## Key Changes

1. **Mesa ANV 26.2.4 `3DSTATE_SBE` `PointCoord` Leak Fix (`79beef54`):** Track `BuiltIn PointCoord` draws per command buffer on `VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA` (`vk_draw.h`/`vk_draw.cpp`) and call `split_part_forced()` in `vk_replay.cpp` before a different program's `NVN_DRAW_PRIMITIVE_POINTS` draw. Eliminates full-screen sun-flare blowout with zero cache invalidation.
2. **16 GB UMA Stability (`vk_vma.cpp`, `vk_draw.cpp`, `device_policy.h`):** Fix single-heap iGPU accounting in `system_heap_usage()` so `imports_give_back()` / Linux pressure unpins retired DMA-BUF imports; share `FallbackBuffers` pool under `replay_mu` + cap `small_vram_textures` to `MFB_BATCH=2` (`-1.04 GiB`).
3. **Barrier Narrowing & Shader Recompiler (`vk_texture.cpp`, `vk_draw.cpp`, `gpu/vk/eden/`):** Narrow compute dispatch and MFB pre/post barriers, enable `bc_small_runs` on Mesa ANV, guard `±Inf`/`NaN` in `FragColor` & `OpImageWrite` (`fng6`), 32-bit guest page-table indexing (`gp35`), `BDA→SSBO` entry-function scoping + `tl_module_prog_key` reset, warp-32 (`wg_warp_plan.h`), and custom Gen12 SPIR-V overrides (`vk_custom_spv.h`).
4. **FP16 FSR 3.1, Fused `P_ACCUMULATE_SQ` & L1 RCAS (`vk_fsr3.cpp`, `vk_temporal.cpp`, `shaders/`):** FP16 rewrite of FSR 3.1 compute passes (`SIMD16`, `0:0` spills), eliminated `dlss_out.comp` via `kSquareOutput`, and 5-tap L1 FP16 RCAS.
5. **Low-Latency 3D-MV FrameGen (`vk_framegen.cpp`, `vk_replay.cpp`):** Exact 3D camera + actor MV synthesis with trajectory consistency gating and exact pre/post-HUD alpha masking, `ahead=1` queue depth on Mesa Intel, and dead-air phase convergence (`0 late`).
6. **Host CPU Efficiency (`host/nvdrv.c`, `vk_memory.cpp`, `vk_texture.cpp`):** AVX2 dirty-page scanning (`collect_in`), `256 KB` range merging, `LOAD_OP_CLEAR` fusion, and `CCS_E` format-list preservation.

Full technical breakdown and documented dead-ends: [`docs/intel_iris_xe_optimization.md`](https://github.com/bbarrosdavi/Longshot-Intel-Iris-Xe/blob/main/docs/intel_iris_xe_optimization.md).

---

## Risks

- **Scope:** This is an umbrella Intel Iris Xe optimization branch (100 files vs `upstream/main @ 209389fa`). If maintainers prefer smaller topic-scoped PRs per `CONTRIBUTING.md`, individual pillars — especially the standalone **Mesa ANV `3DSTATE_SBE` `PointCoord` fix** (`vk_draw.h`/`vk_draw.cpp`/`vk_replay.cpp`) and the **single-heap iGPU `system_heap_usage()` fix** (`vk_vma.cpp`) — are cleanly isolated and trivial to cherry-pick.
- **Cache stamp:** `fng6` and `gp35` update SPIR-V translation stamps relative to stock `main`; players upgrading an existing shader cache will re-translate shaders once on boot. (The ANV `3DSTATE_SBE` fix itself causes zero cache invalidation.)
- **Reversibility:** Every heuristic is scoped to `VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA` or gated behind runtime switches in `gpu/vk/switches.def` (`RX_VK_BDA_SSBO`, `RX_VK_CUSTOM_SPV`, `RX_VK_GP_FAST32`, `RX_VK_MFB_BATCH`, etc.); reverting or toggling switches restores stock behavior with no save/options migration impact.

---

## Checklist

### Ready for review
- [x] Tested on my own PC, in the game across 5 areas, as Testing says.
- [x] Standalone unit & GPU tests pass (`test_pass_barrier`, `test_mesh_fallback_global_attrs`, `test_mesh_fallback_shared_warp`, `test_present_thread`, `test_linux_pressure`, `test_device_policy`, `test_pcache_load`, `test_vendor_translation`, `test_vendor_fixes`).
- [x] Measured against `main @ 209389fa` on identical hardware and settings.
- [x] No game files, build output, or hand-edited generated files; `third_party/xess/` untouched; GPL-3.0-or-later compatible.
- [ ] Rebase onto latest `upstream/main` / split into topic-scoped PRs if requested by maintainers.

### When it applies
- [x] Every new/changed switch has its row in `gpu/vk/switches.def` with measured defaults.
- [x] Checks under `tests/` (`test_device_policy.py`, `test_mesh_fallback_global_attrs.py`, `test_mesh_fallback_shared_warp.py`) updated, not weakened.
- [x] SPIR-V headers regenerated and validated with `spirv-val --target-env vulkan1.2`.
- [x] Before/after screenshots and live 1080p showcase videos included.
- [x] Docs updated (`docs/intel_iris_xe_optimization.md`, `README.md`).
