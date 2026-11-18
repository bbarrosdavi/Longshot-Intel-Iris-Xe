// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <unordered_map>
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

inline WorkgroupWarpPlan AnalyzeWorkgroupWarpPlan(const IR::Program& prog, bool allow_linearize = true) {
    WorkgroupWarpPlan plan{};
    if (prog.stage != Stage::Mesh && prog.stage != Stage::Task) {
        return plan;
    }
    if (prog.workgroup_size[0] != 32 || prog.workgroup_size[1] != 1 || prog.workgroup_size[2] != 1) {
        return plan;
    }
    if (prog.shared_memory_size != 0 || prog.info.uses_cooperative_matrix || prog.info.uses_fswzadd) {
        return plan;
    }
    plan.linearize_if.assign(prog.syntax_list.size(), false);

    struct Frame {
        bool is_loop{};
        size_t open_idx{};
        const IR::Block* entry_block{};
        const IR::Block* merge_block{};
        const IR::Block* last_block{};
        bool has_warp_op{};
        bool has_child_loop{};
        bool has_early_exit{};
    };
    std::vector<Frame> stack;
    const IR::Block* current_block{};
    int loop_depth = 0;

    for (size_t i = 0; i < prog.syntax_list.size(); ++i) {
        const auto& node = prog.syntax_list[i];
        using T = IR::AbstractSyntaxNode::Type;
        switch (node.type) {
        case T::If:
            stack.push_back(Frame{false, i, current_block, node.data.if_node.merge, nullptr, false, false, false});
            current_block = nullptr;
            break;
        case T::Loop:
            for (auto& f : stack) {
                f.has_child_loop = true;
            }
            stack.push_back(Frame{true, i, current_block, node.data.loop.merge, nullptr, false, false, false});
            loop_depth++;
            current_block = nullptr;
            break;
        case T::EndIf: {
            if (stack.empty() || stack.back().is_loop) {
                return {};
            }
            const Frame f = stack.back();
            stack.pop_back();
            if (f.merge_block != node.data.end_if.merge) {
                return {};
            }
            if (f.has_warp_op) {
                if (!allow_linearize || f.has_child_loop || f.has_early_exit || !f.entry_block || !f.last_block) {
                    return {};
                }
                for (const IR::Inst& inst : *f.merge_block) {
                    if (inst.GetOpcode() != IR::Opcode::Phi) {
                        break;
                    }
                    if (inst.NumArgs() != 2) {
                        return {};
                    }
                    const IR::Block* b0 = inst.PhiBlock(0);
                    const IR::Block* b1 = inst.PhiBlock(1);
                    if (!((b0 == f.entry_block && b1 == f.last_block) ||
                          (b0 == f.last_block && b1 == f.entry_block))) {
                        return {};
                    }
                    plan.linear_phis.emplace(&inst, LinearPhiInfo{f.entry_block, f.last_block, f.open_idx});
                }
                plan.linearize_if[f.open_idx] = true;
            }
            current_block = nullptr;
            break;
        }
        case T::Repeat:
            if (stack.empty() || !stack.back().is_loop) {
                return {};
            }
            stack.pop_back();
            loop_depth--;
            current_block = nullptr;
            break;
        case T::Break:
            if (loop_depth <= 0) {
                return {};
            }
            for (size_t k = stack.size(); k > 0 && !stack[k - 1].is_loop; --k) {
                stack[k - 1].has_early_exit = true;
            }
            current_block = nullptr;
            break;
        case T::Return:
        case T::Unreachable:
            if (!stack.empty()) {
                return {};
            }
            current_block = nullptr;
            break;
        case T::Block:
            current_block = node.data.block;
            if (!stack.empty()) {
                stack.back().last_block = current_block;
            }
            for (const IR::Inst& inst : *current_block) {
                using Op = IR::Opcode;
                const Op op = inst.GetOpcode();
                if (op == Op::IsbeReadOutput) {
                    return {};
                }
                if (!stack.empty() &&
                    (op == Op::Barrier || op == Op::WorkgroupMemoryBarrier || op == Op::SubgroupBarrier)) {
                    return {};
                }
                if (!stack.empty() && op == Op::GetSparseFromOp) {
                    return {};
                }
                if (IsWarpSyncOp(op)) {
                    if (loop_depth > 0) {
                        return {};
                    }
                    if (!stack.empty()) {
                        if (!allow_linearize) {
                            return {};
                        }
                        for (auto& f : stack) {
                            f.has_warp_op = true;
                        }
                    }
                }
            }
            break;
        default:
            break;
        }
    }
    if (!stack.empty() || loop_depth != 0) {
        return {};
    }
    plan.eligible = true;
    return plan;
}

} // namespace Shader::Backend::SPIRV
