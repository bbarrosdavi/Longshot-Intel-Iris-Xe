// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shader_recompiler/frontend/ir/abstract_syntax_list.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::SPIRV {

struct LinearPhiInfo {
    const IR::Block* entry_block{};
    const IR::Block* tail_block{};
    size_t if_node_index{};
};

struct WorkgroupWarpPlan {
    bool eligible{false};
    bool had_early_return{false};
    const char* reject_reason{nullptr};
    std::vector<bool> linearize_if;
    std::unordered_map<const IR::Inst*, LinearPhiInfo> linear_phis;
};

inline bool IsWarpSyncOp(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    switch (op) {
    case Op::VoteAll:
    case Op::VoteAny:
    case Op::VoteEqual:
    case Op::SubgroupBallot:
    case Op::ShuffleIndex:
    case Op::ShuffleUp:
    case Op::ShuffleDown:
    case Op::ShuffleButterfly:
    case Op::QuadBroadcast:
    case Op::QuadSwap:
    case Op::SubgroupBarrier:
        return true;
    default:
        return false;
    }
}

inline bool NeedsExecGuard(const IR::Inst& inst) noexcept {
    using Op = IR::Opcode;
    const Op op = inst.GetOpcode();
    switch (op) {
    case Op::ConditionRef:
    case Op::Reference:
    case Op::PhiMove:
    case Op::Prologue:
    case Op::Epilogue:
    case Op::Join:
    case Op::Barrier:
    case Op::WorkgroupMemoryBarrier:
    case Op::SubgroupBarrier:
    case Op::DeviceMemoryBarrier:
        return false;
    case Op::GetCbufU8:
    case Op::GetCbufS8:
    case Op::GetCbufU16:
    case Op::GetCbufS16:
    case Op::GetCbufU32:
    case Op::GetCbufF32:
    case Op::GetCbufU32x2:
        return !inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate();
    case Op::GetAttribute:
    case Op::GetAttributeU32:
    case Op::GetAttributeIndexed:
    case Op::GetPatch:
    case Op::IsbeRead:
    case Op::IsbeReadOutput:
    case Op::LoadGlobalU8:
    case Op::LoadGlobalS8:
    case Op::LoadGlobalU16:
    case Op::LoadGlobalS16:
    case Op::LoadGlobal32:
    case Op::LoadGlobal64:
    case Op::LoadGlobal128:
    case Op::LoadStorageU8:
    case Op::LoadStorageS8:
    case Op::LoadStorageU16:
    case Op::LoadStorageS16:
    case Op::LoadStorage32:
    case Op::LoadStorage64:
    case Op::LoadStorage128:
    case Op::LoadLocal:
    case Op::LoadSharedU8:
    case Op::LoadSharedS8:
    case Op::LoadSharedU16:
    case Op::LoadSharedS16:
    case Op::LoadSharedU32:
    case Op::LoadSharedU64:
    case Op::LoadSharedU128:
    case Op::BindlessImageSampleImplicitLod:
    case Op::BindlessImageSampleExplicitLod:
    case Op::BindlessImageSampleDrefImplicitLod:
    case Op::BindlessImageSampleDrefExplicitLod:
    case Op::BindlessImageGather:
    case Op::BindlessImageGatherDref:
    case Op::BindlessImageFetch:
    case Op::BindlessImageQueryDimensions:
    case Op::BindlessImageQueryLod:
    case Op::BindlessImageGradient:
    case Op::BindlessImageRead:
    case Op::BoundImageSampleImplicitLod:
    case Op::BoundImageSampleExplicitLod:
    case Op::BoundImageSampleDrefImplicitLod:
    case Op::BoundImageSampleDrefExplicitLod:
    case Op::BoundImageGather:
    case Op::BoundImageGatherDref:
    case Op::BoundImageFetch:
    case Op::BoundImageQueryDimensions:
    case Op::BoundImageQueryLod:
    case Op::BoundImageGradient:
    case Op::BoundImageRead:
    case Op::ImageSampleImplicitLod:
    case Op::ImageSampleExplicitLod:
    case Op::ImageSampleDrefImplicitLod:
    case Op::ImageSampleDrefExplicitLod:
    case Op::ImageGather:
    case Op::ImageGatherDref:
    case Op::ImageFetch:
    case Op::ImageQueryDimensions:
    case Op::ImageQueryLod:
    case Op::ImageGradient:
    case Op::ImageRead:
        return true;
    default:
        return inst.MayHaveSideEffects();
    }
}

namespace detail {

inline bool IsGlobalOrImageWrite(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    return (op >= Op::WriteGlobalU8 && op <= Op::WriteGlobal128) ||
           (op >= Op::GlobalAtomicIAdd32 && op <= Op::GlobalAtomicMaxF32x2) ||
           op == Op::BindlessImageWrite || op == Op::BoundImageWrite || op == Op::ImageWrite ||
           (op >= Op::BindlessImageAtomicIAdd32 && op <= Op::ImageAtomicExchange32);
}

inline bool IsStorageWrite(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    return (op >= Op::WriteStorageU8 && op <= Op::WriteStorage128) ||
           (op >= Op::StorageAtomicIAdd32 && op <= Op::StorageAtomicMaxF32x2);
}

inline bool IsUniformPreservingOp(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    return op == Op::Identity || op == Op::ConditionRef || op == Op::WorkgroupId ||
           (op >= Op::GetCbufU8 && op <= Op::GetCbufU32x2) ||
           (op >= Op::CompositeConstructU32x2 && op <= Op::UnpackDouble2x32) ||
           (op >= Op::GetZeroFromOp && op <= Op::GetOverflowFromOp) ||
           (op >= Op::FPAbs16 && op <= Op::UGreaterThanEqual) ||
           (op >= Op::LogicalOr && op <= Op::LogicalNot) ||
           (op >= Op::ConvertS16F16 && op <= Op::ConvertS32S16);
}

inline bool IsGlobalLoad(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    return op >= Op::LoadGlobalU8 && op <= Op::LoadGlobal128;
}

inline bool IsStorageLoad(IR::Opcode op) noexcept {
    using Op = IR::Opcode;
    return op >= Op::LoadStorageU8 && op <= Op::LoadStorage128;
}

inline bool IsUniformVal(IR::Value v, const std::unordered_set<const IR::Inst*>& uniform_insts) {
    if (const IR::Inst* inst = v.TryInstRecursive()) {
        return uniform_insts.contains(inst);
    }
    const IR::Type t = v.Type();
    return t != IR::Type::Void && t != IR::Type::Reg && t != IR::Type::Pred;
}

inline bool InstArgsUniform(const IR::Inst& inst,
                            const std::unordered_set<const IR::Inst*>& uniform_insts) {
    const size_t n = inst.NumArgs();
    for (size_t i = 0; i < n; ++i) {
        if (!IsUniformVal(inst.Arg(i), uniform_insts)) {
            return false;
        }
    }
    return true;
}

struct AstIfInfo {
    size_t open_idx{};
    const IR::Block* entry_block{};
    const IR::Block* merge_block{};
    const IR::Block* last_block{};
    std::vector<size_t> enclosing_ifs;
    std::vector<size_t> enclosing_loops;
    IR::U1 cond{};
    bool has_early_exit{};
    bool has_warp_sync{};
    bool needs_linearize{};
};

struct AstBreakInfo {
    IR::U1 cond{};
    bool nested{};
};

struct AstLoopInfo {
    size_t open_idx{};
    const IR::Block* header_block{};
    const IR::Block* merge_block{};
    std::vector<size_t> enclosing_ifs;
    std::vector<size_t> enclosing_loops;
    std::vector<AstBreakInfo> breaks;
    IR::U1 repeat_cond{};
    bool has_nested_break{};
    bool has_direct_unreachable{};
    bool header_has_guard{};
    bool has_warp_sync{};
};

struct BlockAstInfo {
    std::vector<size_t> enclosing_ifs;
    std::vector<size_t> enclosing_loops;
    size_t loop_header_id{SIZE_MAX};
    size_t if_merge_id{SIZE_MAX};
};

} // namespace detail

inline WorkgroupWarpPlan AnalyzeWorkgroupWarpPlan(const IR::Program& prog, bool allow_linearize = true) {
    WorkgroupWarpPlan plan{};
    auto fail = [&](const char* why) -> WorkgroupWarpPlan {
        WorkgroupWarpPlan out{};
        out.had_early_return = plan.had_early_return;
        out.reject_reason = why;
        return out;
    };
    const bool is_mesh_or_task = (prog.stage == Stage::Mesh || prog.stage == Stage::Task);
    u32 total_threads = 0;
    if (is_mesh_or_task) {
        if (prog.workgroup_size[0] != 32 || prog.workgroup_size[1] != 1 || prog.workgroup_size[2] != 1) {
            return fail("mesh_wg_not_32");
        }
        total_threads = 32U;
    } else if (prog.stage == Stage::Compute) {
        total_threads = prog.workgroup_size[0] * prog.workgroup_size[1] * prog.workgroup_size[2];
        if (total_threads < 32U || total_threads > 256U || (total_threads % 32U) != 0U ||
            (prog.workgroup_size[0] % 8U) != 0U) {
            return fail("comp_wg_shape");
        }
    } else {
        return fail("stage");
    }
    const u32 num_warps = total_threads / 32U;
    // Ping-pong SLM buffers: 2 * 4 u32/warp for SIMD8 ballot + 2 * 32 u32/warp for shuffle = 288 B/warp.
    const u32 wg_bytes = (32U + 256U) * num_warps;
    if (is_mesh_or_task) {
        if (prog.shared_memory_size != 0U) {
            return fail("shared_1warp");
        }
    } else if (prog.shared_memory_size + wg_bytes > 32768U) {
        return fail("shared_limit");
    }
    if (prog.info.uses_cooperative_matrix || prog.info.uses_fswzadd) {
        return fail("mma_or_fswzadd");
    }

    // Pass 0: structural check, instruction counts, early-return detection, and AST metadata.
    using T = IR::AbstractSyntaxNode::Type;
    using Op = IR::Opcode;
    struct OpenFrame {
        bool is_loop{};
        size_t id{};
    };
    std::vector<OpenFrame> stack;
    std::vector<detail::AstIfInfo> ifs;
    std::vector<detail::AstLoopInfo> loops;
    std::unordered_map<const IR::Block*, detail::BlockAstInfo> block_info;
    std::vector<size_t> node_to_id(prog.syntax_list.size(), SIZE_MAX);

    const IR::Block* current_block{};
    size_t total_ir_insts = 0;
    bool writes_global = false;
    std::unordered_set<u32> written_storage_bindings;
    bool saw_warp_sync = false;
    bool saw_early_return = false;

    auto collect_enclosing = [&](std::vector<size_t>& out_ifs, std::vector<size_t>& out_loops) {
        for (const auto& f : stack) {
            if (f.is_loop) {
                out_loops.push_back(f.id);
            } else {
                out_ifs.push_back(f.id);
            }
        }
    };

    for (size_t i = 0; i < prog.syntax_list.size(); ++i) {
        const auto& node = prog.syntax_list[i];
        switch (node.type) {
        case T::If: {
            const size_t id = ifs.size();
            detail::AstIfInfo info{};
            info.open_idx = i;
            info.entry_block = current_block;
            info.merge_block = node.data.if_node.merge;
            info.cond = node.data.if_node.cond;
            collect_enclosing(info.enclosing_ifs, info.enclosing_loops);
            ifs.push_back(std::move(info));
            node_to_id[i] = id;
            stack.push_back(OpenFrame{false, id});
            current_block = nullptr;
            break;
        }
        case T::Loop: {
            const size_t id = loops.size();
            detail::AstLoopInfo info{};
            info.open_idx = i;
            info.header_block = current_block;
            info.merge_block = node.data.loop.merge;
            collect_enclosing(info.enclosing_ifs, info.enclosing_loops);
            if (current_block) {
                block_info[current_block].loop_header_id = id;
                for (const IR::Inst& inst : *current_block) {
                    if (inst.GetOpcode() != Op::Phi && NeedsExecGuard(inst)) {
                        info.header_has_guard = true;
                    }
                }
            }
            loops.push_back(std::move(info));
            node_to_id[i] = id;
            stack.push_back(OpenFrame{true, id});
            current_block = nullptr;
            break;
        }
        case T::EndIf: {
            if (stack.empty() || stack.back().is_loop) {
                return {};
            }
            const size_t id = stack.back().id;
            stack.pop_back();
            if (ifs[id].merge_block != node.data.end_if.merge) {
                return {};
            }
            if (ifs[id].merge_block) {
                block_info[ifs[id].merge_block].if_merge_id = id;
            }
            current_block = nullptr;
            break;
        }
        case T::Repeat: {
            if (stack.empty() || !stack.back().is_loop) {
                return {};
            }
            const size_t id = stack.back().id;
            stack.pop_back();
            loops[id].repeat_cond = node.data.repeat.cond;
            current_block = nullptr;
            break;
        }
        case T::Break: {
            size_t loop_pos = SIZE_MAX;
            for (size_t k = stack.size(); k > 0; --k) {
                if (stack[k - 1].is_loop) {
                    loop_pos = k - 1;
                    break;
                }
                ifs[stack[k - 1].id].has_early_exit = true;
            }
            if (loop_pos == SIZE_MAX) {
                return {};
            }
            const size_t lid = stack[loop_pos].id;
            const bool nested = (loop_pos + 1 != stack.size());
            if (nested) {
                loops[lid].has_nested_break = true;
            }
            loops[lid].breaks.push_back(detail::AstBreakInfo{node.data.break_node.cond, nested});
            current_block = nullptr;
            break;
        }
        case T::Return:
            if (!stack.empty()) {
                saw_early_return = true;
            }
            current_block = nullptr;
            break;
        case T::Unreachable:
            if (!stack.empty()) {
                if (stack.back().is_loop) {
                    loops[stack.back().id].has_direct_unreachable = true;
                }
                for (size_t k = stack.size(); k > 0 && !stack[k - 1].is_loop; --k) {
                    ifs[stack[k - 1].id].has_early_exit = true;
                }
            }
            current_block = nullptr;
            break;
        case T::Block: {
            current_block = node.data.block;
            if (!stack.empty() && !stack.back().is_loop) {
                ifs[stack.back().id].last_block = current_block;
            }
            detail::BlockAstInfo& binfo = block_info[current_block];
            collect_enclosing(binfo.enclosing_ifs, binfo.enclosing_loops);
            for (const IR::Inst& inst : *current_block) {
                ++total_ir_insts;
                const Op op = inst.GetOpcode();
                if (op == Op::IsbeReadOutput) {
                    return fail("isbe_read_output");
                }
                if (detail::IsGlobalOrImageWrite(op)) {
                    writes_global = true;
                } else if (detail::IsStorageWrite(op)) {
                    if (inst.NumArgs() > 0 && inst.Arg(0).IsImmediate()) {
                        written_storage_bindings.insert(inst.Arg(0).U32());
                    } else {
                        writes_global = true;
                    }
                }
                if (IsWarpSyncOp(op)) {
                    saw_warp_sync = true;
                }
            }
            break;
        }
        default:
            break;
        }
    }
    if (!stack.empty()) {
        return {};
    }

    if (prog.stage == Stage::Compute) {
        const bool uses_subgroups = saw_warp_sync || prog.info.uses_subgroup_vote ||
                                    prog.info.uses_subgroup_mask || prog.info.uses_subgroup_invocation_id ||
                                    prog.info.uses_subgroup_shuffles || prog.info.uses_quad_shuffles;
        if (!uses_subgroups) {
            return fail("no_subgroups");
        }
        if (total_ir_insts < 1600) {
            return fail("small_ir");
        }
    }
    if (saw_early_return) {
        plan.had_early_return = true;
        return fail("early_return");
    }

    // Pass 0.5: Canonicalize trivial/self-referential Phis (e.g. loop-invariant values threaded through inner loops).
    std::unordered_map<const IR::Inst*, IR::Value> trivial_phi_canon;
    auto canon_val = [&](IR::Value v) -> IR::Value {
        v = v.Resolve();
        for (int hops = 0; hops < 64; ++hops) {
            const IR::Inst* p = v.TryInstRecursive();
            if (!p) {
                break;
            }
            const auto it = trivial_phi_canon.find(p);
            if (it == trivial_phi_canon.end()) {
                break;
            }
            v = it->second.Resolve();
        }
        return v;
    };
    for (size_t pass = 0; pass < 16; ++pass) {
        bool canon_changed = false;
        for (const auto& node : prog.syntax_list) {
            if (node.type != T::Block) {
                continue;
            }
            for (const IR::Inst& inst : *node.data.block) {
                if (inst.GetOpcode() != Op::Phi) {
                    break;
                }
                if (trivial_phi_canon.contains(&inst)) {
                    continue;
                }
                const size_t n = inst.NumArgs();
                if (n == 0) {
                    continue;
                }
                bool has_unique = false;
                bool ok = true;
                IR::Value unique{};
                for (size_t k = 0; k < n; ++k) {
                    const IR::Value cv = canon_val(inst.Arg(k));
                    if (cv.TryInstRecursive() == &inst) {
                        continue;
                    }
                    if (!has_unique) {
                        unique = cv;
                        has_unique = true;
                    } else if (cv != unique) {
                        ok = false;
                        break;
                    }
                }
                if (ok && has_unique) {
                    trivial_phi_canon.emplace(&inst, unique);
                    canon_changed = true;
                }
            }
        }
        if (!canon_changed) {
            break;
        }
    }

    // Pass 1: Fixed-point workgroup-uniformity analysis (monotonic shrinking lattice).
    std::vector<bool> uniform_if(ifs.size(), true);
    std::vector<bool> uniform_loop(loops.size(), false);
    for (size_t lid = 0; lid < loops.size(); ++lid) {
        uniform_loop[lid] = !loops[lid].has_nested_break && !loops[lid].has_direct_unreachable;
    }

    auto block_in_non_uniform_cf = [&](const detail::BlockAstInfo& binfo) -> bool {
        for (size_t eid : binfo.enclosing_ifs) {
            if (!uniform_if[eid]) {
                return true;
            }
        }
        for (size_t elid : binfo.enclosing_loops) {
            if (!uniform_loop[elid]) {
                return true;
            }
        }
        return false;
    };

    auto phi_backedge_safe_in_loop = [&](const detail::BlockAstInfo& pred_info, size_t lid) -> bool {
        const size_t min_if_count = loops[lid].enclosing_ifs.size();
        for (size_t k = min_if_count; k < pred_info.enclosing_ifs.size(); ++k) {
            if (!uniform_if[pred_info.enclosing_ifs[k]]) {
                return false;
            }
        }
        const size_t min_loop_count = loops[lid].enclosing_loops.size();
        for (size_t k = min_loop_count; k < pred_info.enclosing_loops.size(); ++k) {
            if (!uniform_loop[pred_info.enclosing_loops[k]]) {
                return false;
            }
        }
        return true;
    };

    std::unordered_set<const IR::Inst*> uniform_insts;
    for (size_t outer = 0; outer < ifs.size() + loops.size() + 4; ++outer) {
        uniform_insts.clear();
        std::unordered_set<const IR::Inst*> loop_header_cand;
        for (const auto& node : prog.syntax_list) {
            if (node.type != T::Block) {
                continue;
            }
            const auto& binfo = block_info[node.data.block];
            if (binfo.loop_header_id != SIZE_MAX && uniform_loop[binfo.loop_header_id]) {
                for (const IR::Inst& inst : *node.data.block) {
                    if (inst.GetOpcode() != Op::Phi) {
                        break;
                    }
                    loop_header_cand.insert(&inst);
                    uniform_insts.insert(&inst);
                }
            }
        }

        for (size_t sweep = 0; sweep < 32; ++sweep) {
            bool changed = false;
            for (const auto& node : prog.syntax_list) {
                if (node.type != T::Block) {
                    continue;
                }
                const IR::Block* b = node.data.block;
                const auto& binfo = block_info[b];
                const bool non_unif_cf = block_in_non_uniform_cf(binfo);
                for (const IR::Inst& inst : *b) {
                    const Op op = inst.GetOpcode();
                    bool ok = false;
                    if (op == Op::Phi) {
                        const auto t_it = trivial_phi_canon.find(&inst);
                        if (t_it != trivial_phi_canon.end()) {
                            ok = detail::IsUniformVal(canon_val(t_it->second), uniform_insts);
                        } else {
                            const size_t n = inst.NumArgs();
                            if (n > 0) {
                                const IR::Value first = canon_val(inst.Arg(0));
                                bool all_same = detail::IsUniformVal(first, uniform_insts);
                                for (size_t k = 1; k < n && all_same; ++k) {
                                    if (canon_val(inst.Arg(k)) != first) {
                                        all_same = false;
                                    }
                                }
                                if (all_same) {
                                    ok = true;
                                } else if (binfo.loop_header_id != SIZE_MAX) {
                                    const size_t lid = binfo.loop_header_id;
                                    if (uniform_loop[lid] && loop_header_cand.contains(&inst)) {
                                        if (sweep == 0) {
                                            ok = true;
                                        } else {
                                            ok = true;
                                            for (size_t k = 0; k < n && ok; ++k) {
                                                const IR::Block* pred = inst.PhiBlock(k);
                                                const auto p_it = block_info.find(pred);
                                                if (p_it == block_info.end() ||
                                                    !detail::IsUniformVal(inst.Arg(k), uniform_insts)) {
                                                    ok = false;
                                                    break;
                                                }
                                                const auto& p_loops = p_it->second.enclosing_loops;
                                                const bool inside_lid =
                                                    std::find(p_loops.begin(), p_loops.end(), lid) != p_loops.end();
                                                if (inside_lid && !phi_backedge_safe_in_loop(p_it->second, lid)) {
                                                    ok = false;
                                                }
                                            }
                                            if (!ok) {
                                                loop_header_cand.erase(&inst);
                                            }
                                        }
                                    }
                                } else if (binfo.if_merge_id != SIZE_MAX) {
                                    const size_t if_id = binfo.if_merge_id;
                                    if (uniform_if[if_id]) {
                                        ok = detail::InstArgsUniform(inst, uniform_insts);
                                    }
                                } else if (!non_unif_cf) {
                                    ok = true;
                                    for (size_t k = 0; k < n && ok; ++k) {
                                        const auto p_it = block_info.find(inst.PhiBlock(k));
                                        if (p_it == block_info.end() || block_in_non_uniform_cf(p_it->second) ||
                                            !detail::IsUniformVal(inst.Arg(k), uniform_insts)) {
                                            ok = false;
                                        }
                                    }
                                }
                            }
                        }
                    } else if (non_unif_cf && NeedsExecGuard(inst)) {
                        ok = false;
                    } else if (op == Op::IsbeRead) {
                        // Task: returns WorkgroupId.x + first_task (always uniform).
                        // Mesh: reads read-only task payload (uniform when address is uniform).
                        ok = (prog.stage == Stage::Task) ||
                             (prog.stage == Stage::Mesh && detail::InstArgsUniform(inst, uniform_insts));
                    } else if (detail::IsUniformPreservingOp(op)) {
                        ok = detail::InstArgsUniform(inst, uniform_insts);
                    } else if (!writes_global && written_storage_bindings.empty() && !non_unif_cf &&
                               detail::IsGlobalLoad(op)) {
                        ok = detail::InstArgsUniform(inst, uniform_insts);
                    } else if (!writes_global && !non_unif_cf && detail::IsStorageLoad(op) &&
                               inst.NumArgs() > 0 && inst.Arg(0).IsImmediate() &&
                               !written_storage_bindings.contains(inst.Arg(0).U32())) {
                        ok = detail::InstArgsUniform(inst, uniform_insts);
                    }
                    const bool was = uniform_insts.contains(&inst);
                    if (ok && !was) {
                        uniform_insts.insert(&inst);
                        changed = true;
                    } else if (!ok && was) {
                        uniform_insts.erase(&inst);
                        changed = true;
                    }
                }
            }
            if (sweep > 0 && !changed) {
                break;
            }
        }

        bool cf_changed = false;
        for (size_t id = 0; id < ifs.size(); ++id) {
            if (!uniform_if[id]) {
                continue;
            }
            bool next_ok = detail::IsUniformVal(ifs[id].cond, uniform_insts);
            for (size_t eid : ifs[id].enclosing_ifs) {
                next_ok = next_ok && uniform_if[eid];
            }
            for (size_t elid : ifs[id].enclosing_loops) {
                next_ok = next_ok && uniform_loop[elid];
            }
            if (!next_ok) {
                uniform_if[id] = false;
                cf_changed = true;
            }
        }
        for (size_t lid = 0; lid < loops.size(); ++lid) {
            if (!uniform_loop[lid]) {
                continue;
            }
            bool next_ok = detail::IsUniformVal(loops[lid].repeat_cond, uniform_insts);
            for (const auto& brk : loops[lid].breaks) {
                next_ok = next_ok && !brk.nested && detail::IsUniformVal(brk.cond, uniform_insts);
            }
            bool inside_non_unif_if = false;
            for (size_t eid : loops[lid].enclosing_ifs) {
                if (!uniform_if[eid]) {
                    inside_non_unif_if = true;
                    break;
                }
            }
            if (inside_non_unif_if && loops[lid].header_has_guard) {
                next_ok = false;
            }
            for (size_t elid : loops[lid].enclosing_loops) {
                if (!uniform_loop[elid]) {
                    next_ok = false;
                    break;
                }
            }
            if (!next_ok) {
                uniform_loop[lid] = false;
                cf_changed = true;
            }
        }
        if (!cf_changed) {
            break;
        }
    }

    // Pass 2A: Determine which non-uniform Ifs need linearization and validate barriers + merge Phis.
    plan.linearize_if.assign(prog.syntax_list.size(), false);
    stack.clear();
    int loop_depth = 0;
    size_t warp_ops_in_loops = 0;

    for (size_t i = 0; i < prog.syntax_list.size(); ++i) {
        const auto& node = prog.syntax_list[i];
        switch (node.type) {
        case T::If:
            stack.push_back(OpenFrame{false, node_to_id[i]});
            break;
        case T::Loop:
            stack.push_back(OpenFrame{true, node_to_id[i]});
            ++loop_depth;
            break;
        case T::EndIf: {
            const size_t id = stack.back().id;
            stack.pop_back();
            const auto& f = ifs[id];
            if (f.needs_linearize) {
                if (!allow_linearize || f.has_early_exit || !f.entry_block || !f.last_block) {
                    return fail(f.has_early_exit ? "lin_early_exit" : "lin_no_blocks");
                }
                for (const IR::Inst& inst : *f.merge_block) {
                    if (inst.GetOpcode() != Op::Phi) {
                        break;
                    }
                    if (inst.NumArgs() != 2) {
                        return fail("lin_phi_nargs");
                    }
                    const IR::Block* b0 = inst.PhiBlock(0);
                    const IR::Block* b1 = inst.PhiBlock(1);
                    if (!((b0 == f.entry_block && b1 == f.last_block) ||
                          (b0 == f.last_block && b1 == f.entry_block))) {
                        return fail("lin_phi_blocks");
                    }
                    plan.linear_phis.emplace(&inst, LinearPhiInfo{f.entry_block, f.last_block, f.open_idx});
                }
                plan.linearize_if[f.open_idx] = true;
            }
            break;
        }
        case T::Repeat:
            stack.pop_back();
            --loop_depth;
            break;
        case T::Block:
            for (const IR::Inst& inst : *node.data.block) {
                const Op op = inst.GetOpcode();
                if (!stack.empty() && op == Op::GetSparseFromOp) {
                    return fail("sparse_in_cf");
                }
                if (op == Op::Barrier || op == Op::WorkgroupMemoryBarrier) {
                    for (const auto& f : stack) {
                        if (f.is_loop ? !uniform_loop[f.id] : !uniform_if[f.id]) {
                            return fail(f.is_loop ? "barrier_nonunif_loop" : "barrier_nonunif_if");
                        }
                    }
                }
                if (IsWarpSyncOp(op)) {
                    if (loop_depth > 0) {
                        ++warp_ops_in_loops;
                    }
                    for (size_t k = stack.size(); k > 0; --k) {
                        const auto& f = stack[k - 1];
                        if (f.is_loop) {
                            loops[f.id].has_warp_sync = true;
                            if (!uniform_loop[f.id]) {
                                return fail("warp_nonunif_loop");
                            }
                        } else {
                            ifs[f.id].has_warp_sync = true;
                            if (!uniform_if[f.id]) {
                                if (!allow_linearize) {
                                    return fail("warp_nonunif_if");
                                }
                                ifs[f.id].needs_linearize = true;
                            }
                        }
                    }
                }
            }
            break;
        default:
            break;
        }
    }

    if (prog.stage == Stage::Compute && warp_ops_in_loops > 16) {
        return fail("loop_warp_gt16");
    }

    // Pass 2B: Ensure every Loop inside a linearized If is either uniform or guarded by a non-linearized If.
    stack.clear();
    for (size_t i = 0; i < prog.syntax_list.size(); ++i) {
        const auto& node = prog.syntax_list[i];
        switch (node.type) {
        case T::If:
            stack.push_back(OpenFrame{false, node_to_id[i]});
            break;
        case T::Loop: {
            const size_t lid = node_to_id[i];
            size_t innermost_lin_pos = SIZE_MAX;
            for (size_t k = stack.size(); k > 0; --k) {
                if (!stack[k - 1].is_loop && plan.linearize_if[ifs[stack[k - 1].id].open_idx]) {
                    innermost_lin_pos = k - 1;
                    break;
                }
            }
            if (innermost_lin_pos != SIZE_MAX) {
                bool has_guard_if = false;
                for (size_t k = innermost_lin_pos + 1; k < stack.size(); ++k) {
                    if (!stack[k].is_loop && !plan.linearize_if[ifs[stack[k].id].open_idx]) {
                        has_guard_if = true;
                        break;
                    }
                }
                if (loops[lid].has_warp_sync) {
                    if (!uniform_loop[lid] || has_guard_if) {
                        return fail("pass2b_warp_loop");
                    }
                } else if (!has_guard_if && !uniform_loop[lid]) {
                    return fail("pass2b_unguarded_loop");
                }
            }
            stack.push_back(OpenFrame{true, lid});
            break;
        }
        case T::EndIf:
        case T::Repeat:
            stack.pop_back();
            break;
        default:
            break;
        }
    }

    plan.eligible = true;
    return plan;
}

} // namespace Shader::Backend::SPIRV
