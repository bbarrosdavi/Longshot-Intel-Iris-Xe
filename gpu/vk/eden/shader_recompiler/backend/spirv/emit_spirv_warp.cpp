// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#include "shader_recompiler/backend/spirv/mma_software.h"

namespace Shader::Backend::SPIRV {
namespace {
Id SubgroupScope(EmitContext& ctx) {
    return ctx.Const(static_cast<u32>(spv::Scope::Subgroup));
}

Id LinearLocalId(EmitContext& ctx) {
    const Id lid{ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id)};
    Id linear{ctx.OpCompositeExtract(ctx.U32[1], lid, 0U)};
    if (ctx.wg_size_y > 1U) {
        const Id y{ctx.OpCompositeExtract(ctx.U32[1], lid, 1U)};
        linear = ctx.OpIAdd(ctx.U32[1], linear, ctx.OpIMul(ctx.U32[1], y, ctx.Const(ctx.wg_size_x)));
    }
    if (ctx.wg_size_z > 1U) {
        const Id z{ctx.OpCompositeExtract(ctx.U32[1], lid, 2U)};
        linear = ctx.OpIAdd(
            ctx.U32[1], linear,
            ctx.OpIMul(ctx.U32[1], z, ctx.Const(ctx.wg_size_x * ctx.wg_size_y)));
    }
    return linear;
}

Id WgWarpId(EmitContext& ctx) {
    return ctx.OpShiftRightLogical(ctx.U32[1], LinearLocalId(ctx), ctx.Const(5U));
}

Id WgTid(EmitContext& ctx) {
    return ctx.OpBitwiseAnd(ctx.U32[1], LinearLocalId(ctx), ctx.Const(31U));
}

Id WgShflPtr(EmitContext& ctx, Id slot, Id lane_in_warp) {
    Id base;
    if (ctx.mesh_wg_warps <= 1U) {
        base = ctx.OpShiftLeftLogical(ctx.U32[1], slot, ctx.Const(5U));
    } else {
        const Id warp_slot{ctx.OpIAdd(ctx.U32[1],
                                      ctx.OpIMul(ctx.U32[1], slot, ctx.Const(ctx.mesh_wg_warps)),
                                      WgWarpId(ctx))};
        base = ctx.OpShiftLeftLogical(ctx.U32[1], warp_slot, ctx.Const(5U));
    }
    return ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_shfl,
                             ctx.OpIAdd(ctx.U32[1], base, lane_in_warp), ctx.u32_zero_value);
}

void WgBarrier(EmitContext& ctx) {
    ctx.wg_shfl_cached_block = {};
    ctx.wg_shfl_cached_value = {};
    ctx.wg_shfl_cached_slot = {};
    const Id scope{ctx.Const(static_cast<u32>(spv::Scope::Workgroup))};
    ctx.OpControlBarrier(scope, scope,
                         ctx.Const(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::WorkgroupMemory)));
}

Id WgBallot(EmitContext& ctx, Id pred) {
    const bool has_exec{Sirit::ValidId(ctx.linear_exec_id)};
    const Id eff_pred{has_exec ? (pred.value == ctx.true_value.value
                                      ? ctx.linear_exec_id
                                      : ctx.OpLogicalAnd(ctx.U1, ctx.linear_exec_id, pred))
                               : pred};
    // Outside linearized divergent Ifs all 32 warp lanes are active, so Ballot(true) is ~0u.
    if (eff_pred.value == ctx.true_value.value) {
        return ctx.Const(0xFFFFFFFFu);
    }
    if (eff_pred.value == ctx.false_value.value) {
        return ctx.u32_zero_value;
    }
    if (Sirit::ValidId(ctx.current_block_label) &&
        ctx.wg_ballot_cached_block.value == ctx.current_block_label.value &&
        ctx.wg_ballot_cached_pred.value == eff_pred.value) {
        return ctx.wg_ballot_cached_result;
    }
    // SIMD8-assisted 1-barrier ping-pong ballot: each 8-lane hardware subgroup evaluates its 8-bit
    // mask natively, lane 0 of each subgroup stores into its disjoint SLM slot, then 1 Workgroup
    // barrier publishes all 4 bytes (0 atomics, 1 barrier instead of 3 barriers + 2 atomics).
    const Id sub_ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), eff_pred)};
    const Id sub_mask{ctx.OpBitwiseAnd(ctx.U32[1], ctx.OpCompositeExtract(ctx.U32[1], sub_ballot, 0U),
                                       ctx.Const(0xFFu))};
    const Id tid{WgTid(ctx)};
    const Id sub_id{ctx.OpShiftRightLogical(ctx.U32[1], tid, ctx.Const(3U))};
    const Id slot{ctx.OpLoad(ctx.U32[1], ctx.wg_ballot_slot)};
    ctx.OpStore(ctx.wg_ballot_slot, ctx.OpBitwiseXor(ctx.U32[1], slot, ctx.Const(1U)));
    const Id base{ctx.mesh_wg_warps <= 1U
                      ? ctx.OpShiftLeftLogical(ctx.U32[1], slot, ctx.Const(2U))
                      : ctx.OpShiftLeftLogical(
                            ctx.U32[1],
                            ctx.OpIAdd(ctx.U32[1],
                                       ctx.OpIMul(ctx.U32[1], slot, ctx.Const(ctx.mesh_wg_warps)),
                                       WgWarpId(ctx)),
                            ctx.Const(2U))};
    const Id is_sub_leader{
        ctx.OpIEqual(ctx.U1, ctx.OpBitwiseAnd(ctx.U32[1], tid, ctx.Const(7U)), ctx.u32_zero_value)};
    const Id then_lbl{ctx.OpLabel()};
    const Id merge_lbl{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_lbl, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_sub_leader, then_lbl, merge_lbl);
    ctx.AddLabel(then_lbl);
    const Id write_ptr{ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_ballot,
                                         ctx.OpIAdd(ctx.U32[1], base, sub_id), ctx.u32_zero_value)};
    ctx.OpStore(write_ptr, sub_mask);
    ctx.OpBranch(merge_lbl);
    ctx.AddLabel(merge_lbl);
    WgBarrier(ctx);
    const Id b0{ctx.OpLoad(
        ctx.U32[1], ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_ballot, base, ctx.u32_zero_value))};
    const Id b1{ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_ballot,
                                                         ctx.OpIAdd(ctx.U32[1], base, ctx.Const(1U)),
                                                         ctx.u32_zero_value))};
    const Id b2{ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_ballot,
                                                         ctx.OpIAdd(ctx.U32[1], base, ctx.Const(2U)),
                                                         ctx.u32_zero_value))};
    const Id b3{ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_wg_ballot,
                                                         ctx.OpIAdd(ctx.U32[1], base, ctx.Const(3U)),
                                                         ctx.u32_zero_value))};
    const Id lo16{ctx.OpBitwiseOr(ctx.U32[1], b0, ctx.OpShiftLeftLogical(ctx.U32[1], b1, ctx.Const(8U)))};
    const Id hi16{ctx.OpBitwiseOr(ctx.U32[1], ctx.OpShiftLeftLogical(ctx.U32[1], b2, ctx.Const(16U)),
                                  ctx.OpShiftLeftLogical(ctx.U32[1], b3, ctx.Const(24U)))};
    const Id res{ctx.OpBitwiseOr(ctx.U32[1], lo16, hi16)};
    ctx.wg_ballot_cached_block = ctx.current_block_label;
    ctx.wg_ballot_cached_pred = eff_pred;
    ctx.wg_ballot_cached_result = res;
    return res;
}

Id WgShuffle(EmitContext& ctx, Id value, Id src_lane) {
    const Id src{ctx.OpBitwiseAnd(ctx.U32[1], src_lane, ctx.Const(31U))};
    if (Sirit::ValidId(ctx.current_block_label) &&
        ctx.wg_shfl_cached_block.value == ctx.current_block_label.value &&
        ctx.wg_shfl_cached_value.value == value.value) {
        return ctx.OpLoad(ctx.U32[1], WgShflPtr(ctx, ctx.wg_shfl_cached_slot, src));
    }
    const Id tid{WgTid(ctx)};
    const Id slot{ctx.OpLoad(ctx.U32[1], ctx.wg_shfl_slot)};
    ctx.OpStore(ctx.wg_shfl_slot, ctx.OpBitwiseXor(ctx.U32[1], slot, ctx.Const(1U)));
    ctx.OpStore(WgShflPtr(ctx, slot, tid), value);
    WgBarrier(ctx);
    ctx.wg_shfl_cached_block = ctx.current_block_label;
    ctx.wg_shfl_cached_value = value;
    ctx.wg_shfl_cached_slot = slot;
    return ctx.OpLoad(ctx.U32[1], WgShflPtr(ctx, slot, src));
}

bool StageSupportsSubgroups(EmitContext& ctx) {
    return ctx.profile.SupportsSubgroupStage(ctx.stage);
}

Id GetThreadId(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgTid(ctx);
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id WarpExtract(EmitContext& ctx, Id value) {
    const Id thread_id{GetThreadId(ctx)};
    const Id local_index{ctx.OpShiftRightArithmetic(ctx.U32[1], thread_id, ctx.Const(5U))};
    if (ctx.profile.has_broken_spirv_subgroup_mask_vector_extract_dynamic) {
        const Id c0_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(0U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 0U), ctx.Const(0U))};
        const Id c1_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(1U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 1U), ctx.Const(0U))};
        const Id c2_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(2U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 2U), ctx.Const(0U))};
        const Id c3_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(3U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 3U), ctx.Const(0U))};
        const Id c0_or_c1{ctx.OpBitwiseOr(ctx.U32[1], c0_sel, c1_sel)};
        const Id c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c2_sel, c3_sel)};
        const Id c0_or_c1_or_c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c0_or_c1, c2_or_c3)};
        return c0_or_c1_or_c2_or_c3;
    } else {
        return ctx.OpVectorExtractDynamic(ctx.U32[1], value, local_index);
    }
}

Id LoadMask(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpLoad(ctx.U32[4], mask)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpCompositeExtract(ctx.U32[1], value, 0U);
    }
    return WarpExtract(ctx, value);
}

void SetInBoundsFlag(IR::Inst* inst, Id result) {
    IR::Inst* const in_bounds{inst->GetAssociatedPseudoOperation(IR::Opcode::GetInBoundsFromOp)};
    if (!in_bounds) {
        return;
    }
    in_bounds->SetDefinition(result);
    in_bounds->Invalidate();
}

Id ComputeMinThreadId(EmitContext& ctx, Id thread_id, Id segmentation_mask) {
    return ctx.OpBitwiseAnd(ctx.U32[1], thread_id, segmentation_mask);
}

Id ComputeMaxThreadId(EmitContext& ctx, Id min_thread_id, Id clamp, Id not_seg_mask) {
    return ctx.OpBitwiseOr(ctx.U32[1], min_thread_id,
                           ctx.OpBitwiseAnd(ctx.U32[1], clamp, not_seg_mask));
}

Id GetMaxThreadId(EmitContext& ctx, Id thread_id, Id clamp, Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    return ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask);
}

Id HostThreadId(EmitContext& ctx, Id thread_id) {
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return thread_id;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    const Id partition_base{ctx.OpShiftLeftLogical(ctx.U32[1], partition_idx, ctx.Const(5u))};
    return ctx.OpIAdd(ctx.U32[1], thread_id, partition_base);
}

Id GuestLane(EmitContext& ctx, Id index) {
    return ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U));
}

bool UseShuffleFallback(EmitContext& ctx) {
    return ctx.profile.has_broken_spirv_subgroup_shuffle ||
           (ctx.profile.has_broken_spirv_fragment_shuffle && ctx.stage == Stage::Fragment);
}

Id ShuffleAbsolute(EmitContext& ctx, Id value, Id src_thread_id, bool broadcast) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgShuffle(ctx, value, src_thread_id);
    }
    if (!broadcast) {
        return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, src_thread_id);
    }
    Id result{ctx.u32_zero_value};
    for (u32 lane = 0; lane < ctx.profile.max_subgroup_size; ++lane) {
        const Id read{
            ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, src_thread_id, ctx.Const(lane))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}

Id ShuffleRelative(EmitContext& ctx, Id value, Id delta, Id src_thread_id, spv::Op op) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgShuffle(ctx, value, src_thread_id);
    }
    const bool broadcast{UseShuffleFallback(ctx)};
    if (!ctx.profile.support_shuffle_relative || broadcast) {
        return ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), broadcast);
    }
    const Id result{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, delta)};
    ctx.opcode_rewrites.emplace_back(result.value, op);
    return result;
}

Id BroadcastLane(EmitContext& ctx, Id value, u32 lane) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgShuffle(ctx, value, ctx.Const(lane));
    }
    Id result{
        ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return result;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    for (u32 base = 32; base < ctx.profile.max_subgroup_size; base += 32) {
        const Id read{ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value,
                                                     ctx.Const(base + lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, partition_idx, ctx.Const(base >> 5))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}
} // Anonymous namespace

Id EmitLaneId(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgTid(ctx);
    }
    const Id id{GetThreadId(ctx)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return id;
    }
    return ctx.OpBitwiseAnd(ctx.U32[1], id, ctx.Const(31U));
}

Id EmitVoteAll(EmitContext& ctx, Id pred) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        if (!Sirit::ValidId(ctx.linear_exec_id) && pred.value == ctx.true_value.value) {
            return ctx.true_value;
        }
        return ctx.OpIEqual(ctx.U1, WgBallot(ctx, ctx.OpLogicalNot(ctx.U1, pred)), ctx.u32_zero_value);
    }
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAll(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
    return ctx.OpIEqual(ctx.U1, lhs, active_mask);
}

Id EmitVoteAny(EmitContext& ctx, Id pred) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        if (!Sirit::ValidId(ctx.linear_exec_id) && pred.value == ctx.true_value.value) {
            return ctx.true_value;
        }
        return ctx.OpINotEqual(ctx.U1, WgBallot(ctx, pred), ctx.u32_zero_value);
    }
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAny(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
    return ctx.OpINotEqual(ctx.U1, lhs, ctx.u32_zero_value);
}

Id EmitVoteEqual(EmitContext& ctx, Id pred) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        const Id b_t{WgBallot(ctx, pred)};
        const Id active{WgBallot(ctx, ctx.true_value)};
        return ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, b_t, ctx.u32_zero_value),
                               ctx.OpIEqual(ctx.U1, b_t, active));
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.true_value;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAllEqual(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseXor(ctx.U32[1], ballot, active_mask)};
    return ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, lhs, ctx.u32_zero_value),
                           ctx.OpIEqual(ctx.U1, lhs, active_mask));
}

Id EmitSubgroupBallot(EmitContext& ctx, Id pred) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return WgBallot(ctx, pred);
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.OpSelect(ctx.U32[1], pred, ctx.Const(1u), ctx.u32_zero_value);
    }
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    }
    return WarpExtract(ctx, ballot);
}

Id EmitSubgroupEqMask(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return ctx.OpShiftLeftLogical(ctx.U32[1], ctx.Const(1U), WgTid(ctx));
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_eq);
}

Id EmitSubgroupLtMask(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return ctx.OpISub(ctx.U32[1], EmitSubgroupEqMask(ctx), ctx.Const(1U));
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return LoadMask(ctx, ctx.subgroup_mask_lt);
}

Id EmitSubgroupLeMask(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        const Id eq{EmitSubgroupEqMask(ctx)};
        return ctx.OpBitwiseOr(ctx.U32[1], ctx.OpISub(ctx.U32[1], eq, ctx.Const(1U)), eq);
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_le);
}

Id EmitSubgroupGtMask(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return ctx.OpNot(ctx.U32[1], EmitSubgroupLeMask(ctx));
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return LoadMask(ctx, ctx.subgroup_mask_gt);
}

Id EmitSubgroupGeMask(EmitContext& ctx) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        return ctx.OpNot(ctx.U32[1], EmitSubgroupLtMask(ctx));
    }
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_ge);
}

Id EmitShuffleIndex(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                    Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    const Id max_thread_id{ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask)};

    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], GuestLane(ctx, index), not_seg_mask)};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], lhs, min_thread_id)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    if (!StageSupportsSubgroups(ctx)) {
        return value;
    }
    const IR::Value lane{inst->Arg(1).Resolve()};
    const IR::Value segment{inst->Arg(3).Resolve()};
    if (ctx.runtime_info.mesh_wg_warp32 && segment.IsImmediate() && (segment.U32() & 24U) == 24U) {
        const Id local_src{ctx.OpBitwiseAnd(ctx.U32[1], src_thread_id, ctx.Const(7U))};
        const Id shuffled{ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, local_src)};
        return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
    }
    if (lane.IsImmediate() && segment.IsImmediate() && segment.U32() == 0) {
        return ctx.OpSelect(ctx.U32[1], in_range, BroadcastLane(ctx, value, lane.U32() & 31),
                            value);
    }
    const Id shuffled{
        ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), UseShuffleFallback(ctx))};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleUp(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                 Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpISub(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSGreaterThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    if (ctx.runtime_info.mesh_wg_warp32) {
        const IR::Value segment{inst->Arg(3).Resolve()};
        if (segment.IsImmediate() && (segment.U32() & 24U) == 24U) {
            const Id local_src{ctx.OpBitwiseAnd(ctx.U32[1], src_thread_id, ctx.Const(7U))};
            const Id shuffled{ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, local_src)};
            return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
        }
    }
    const Id shuffled{ShuffleRelative(ctx, value, delta, src_thread_id,
                                      spv::Op::OpGroupNonUniformShuffleUp)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleDown(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                   Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpIAdd(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    if (ctx.runtime_info.mesh_wg_warp32) {
        const IR::Value segment{inst->Arg(3).Resolve()};
        if (segment.IsImmediate() && (segment.U32() & 24U) == 24U) {
            const Id local_src{ctx.OpBitwiseAnd(ctx.U32[1], src_thread_id, ctx.Const(7U))};
            const Id shuffled{ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, local_src)};
            return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
        }
    }
    const Id shuffled{ShuffleRelative(ctx, value, delta, src_thread_id,
                                      spv::Op::OpGroupNonUniformShuffleDown)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleButterfly(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                        Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id mask{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpBitwiseXor(ctx.U32[1], thread_id, mask)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    // (longshot) through the broadcasts where shuffles are broken (UseShuffleFallback: AMD's fragment stages, Lichii #273),
    // as the absolute and relative shuffles are; #273 left SHFL.BFLY native, and some of Kokiri Forest's plants use it
    if (ctx.runtime_info.mesh_wg_warp32) {
        const IR::Value mask_val{inst->Arg(1).Resolve()};
        if (mask_val.IsImmediate() && (mask_val.U32() & 31U) < 8U) {
            const Id shuffled{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value,
                                                              ctx.Const(mask_val.U32() & 7U))};
            return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
        }
        return ctx.OpSelect(ctx.U32[1], in_range, WgShuffle(ctx, value, src_thread_id), value);
    }
    const Id shuffled{UseShuffleFallback(ctx)
                          ? ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), true)
                          : ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id lane) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        // Every 4-lane quad lies strictly inside its 8-lane SIMD8 hardware subgroup.
        const Id quad_base{ctx.OpBitwiseAnd(ctx.U32[1], WgTid(ctx), ctx.Const(4U))};
        const Id local_lane{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3U))};
        const Id local_src{ctx.OpBitwiseOr(ctx.U32[1], quad_base, local_lane)};
        return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, local_src);
    }
    if (ctx.profile.support_quad_shuffles) {
        return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
    }
    const Id base{ctx.OpBitwiseAnd(ctx.U32[1], GetThreadId(ctx), ctx.Const(~3u))};
    const Id local_lane{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3u))};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], base, local_lane)};
    return ShuffleAbsolute(ctx, value, src_thread_id,
                           ctx.profile.has_broken_spirv_subgroup_shuffle);
}

Id EmitQuadSwap(EmitContext& ctx, Id value, Id direction) {
    if (ctx.runtime_info.mesh_wg_warp32) {
        const Id xor_mask{ctx.OpIAdd(ctx.U32[1], direction, ctx.Const(1u))};
        return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, xor_mask);
    }
    if (ctx.profile.support_quad_shuffles) {
        const Id result{
            ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, direction)};
        ctx.opcode_rewrites.emplace_back(result.value, spv::Op::OpGroupNonUniformQuadSwap);
        return result;
    }
    const Id xor_mask{ctx.OpIAdd(ctx.U32[1], direction, ctx.Const(1u))};
    // (longshot) not through UseShuffleFallback's broadcasts: a program with quad shuffles alone (uses_quad_shuffles) has
    // neither the lane id input nor the ballot capability they need, and the SPIR-V made with them crashed the start
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, xor_mask);
}

Id EmitFSwizzleAdd(EmitContext& ctx, Id op_a, Id op_b, Id swizzle) {
    const Id three{ctx.Const(3U)};
    Id mask{GetThreadId(ctx)};
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);
    mask = ctx.OpShiftLeftLogical(ctx.U32[1], mask, ctx.Const(1U));
    mask = ctx.OpShiftRightLogical(ctx.U32[1], swizzle, mask);
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);

    const Id modifier_a{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_a, mask)};
    const Id modifier_b{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_b, mask)};

    const Id result_a{ctx.OpFMul(ctx.F32[1], op_a, modifier_a)};
    const Id result_b{ctx.OpFMul(ctx.F32[1], op_b, modifier_b)};
    return ctx.OpFAdd(ctx.F32[1], result_a, result_b);
}

Id EmitDPdxFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxFine(ctx.F32[1], op_a);
}

Id EmitDPdyFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyFine(ctx.F32[1], op_a);
}

Id EmitDPdxCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxCoarse(ctx.F32[1], op_a);
}

Id EmitDPdyCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyCoarse(ctx.F32[1], op_a);
}

// (longshot) HMMA in software (mma_software.h): the 32 lanes' fragments through subgroup operations in one helper function,
// accumulated with FMA in f32 and rounded to f16 after every 8 columns of K, as HMMA.F16 does. It needs no cooperative matrices
// (RX_VK_MMA_SOFTWARE=1, vk_coopmat.cpp); the form is the one vk_coopmat.cpp's self-test chose for this device (mma_software_form).
void DefineSoftwareMatrixFunction(EmitContext& ctx) {
    ctx.AddCapability(spv::Capability::GroupNonUniform);
    ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
    if (!ctx.profile.SupportsSubgroupStage(ctx.stage) || ctx.profile.has_broken_spirv_subgroup_shuffle)
        throw NotImplementedException("Software HMMA requires subgroup shuffle support");
    const auto form = static_cast<SoftwareMmaForm>(mma_software_form);
    const bool quad = form != SoftwareMmaForm::Loop && mma_software_quad;
    if (quad) ctx.AddCapability(spv::Capability::GroupNonUniformQuad);
    const SoftwareMmaTypes types{ctx.U32[1], ctx.U32[2], ctx.U32[4], ctx.F32[1], ctx.F32[2], ctx.F16[1], ctx.U1};
    ctx.software_matrix_function = DefineSoftwareMma(ctx, types, ctx.subgroup_local_invocation_id, form, quad);
}


// (longshot) HMMA_nx2: D(16x8) = A(16x16) * B(16x8) + C on the subgroup's cooperative matrices. Each
// lane's fragment registers hold its elements in the console's order (two f16 a word); they go to and come
// from the device's elements in its own order (mma_layout.h, measured by vk_coopmat.cpp).
Id EmitMatrixMulAddF16(EmitContext& ctx, Id a, Id b, Id c) {
    if (mma_software) return ctx.OpFunctionCall(ctx.U32[2], ctx.software_matrix_function, a, b, c);
    const Id f16{ctx.F16[1]};
    const Id scope{ctx.Const(static_cast<u32>(spv::Scope::Subgroup))};
    const auto type = [&](u32 rows, u32 columns, spv::CooperativeMatrixUse use) {
        return ctx.TypeCooperativeMatrixKHR(f16, scope, ctx.Const(rows), ctx.Const(columns),
                                            ctx.Const(static_cast<u32>(use)));
    };
    const Id a_type{type(16, 16, spv::CooperativeMatrixUse::MatrixAKHR)};
    const Id b_type{type(16, 8, spv::CooperativeMatrixUse::MatrixBKHR)};
    const Id c_type{type(16, 8, spv::CooperativeMatrixUse::MatrixAccumulatorKHR)};
    const Id zero{ctx.Constant(f16, 0U)};
    const auto load = [&](Id matrix_type, Id words, u32 count, const auto& order) {
        Id m{ctx.OpCompositeConstruct(matrix_type, zero)};
        for (u32 w = 0; w < count; ++w) {
            const Id word{count == 1 ? words : ctx.OpCompositeExtract(ctx.U32[1], words, w)};
            const Id pair{ctx.OpBitcast(ctx.F16[2], word)};
            for (u32 h = 0; h < 2; ++h) {
                m = ctx.OpCompositeInsert(matrix_type, ctx.OpCompositeExtract(f16, pair, h), m, order[w * 2 + h]);
            }
        }
        return m;
    };
    const Id d{ctx.OpCooperativeMatrixMulAddKHR(c_type, load(a_type, a, 4, mma_order_a), load(b_type, b, 2, mma_order_b),
                                                    load(c_type, c, 2, mma_order_c))};
    Id out[2];
    for (u32 w = 0; w < 2; ++w) {
        const Id pair{ctx.OpCompositeConstruct(ctx.F16[2], ctx.OpCompositeExtract(f16, d, mma_order_c[w * 2]),
                                               ctx.OpCompositeExtract(f16, d, mma_order_c[w * 2 + 1]))};
        out[w] = ctx.OpBitcast(ctx.U32[1], pair);
    }
    return ctx.OpCompositeConstruct(ctx.U32[2], out[0], out[1]);
}

} // namespace Shader::Backend::SPIRV
