// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "impl/context/context_id.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

class SubDeviceImpl {
public:
    // Constructors for internal tt_metal/ use
    explicit SubDeviceImpl(
        const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores, ContextId context_id = SILICON_CONTEXT_ID);
    explicit SubDeviceImpl(
        std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores, ContextId context_id = SILICON_CONTEXT_ID);
    explicit SubDeviceImpl(tt::stl::Span<const CoreRangeSet> cores, ContextId context_id = SILICON_CONTEXT_ID);

    // Copy/move semantics
    SubDeviceImpl(const SubDeviceImpl&) = default;
    SubDeviceImpl& operator=(const SubDeviceImpl&) = default;
    SubDeviceImpl(SubDeviceImpl&&) noexcept = default;
    SubDeviceImpl& operator=(SubDeviceImpl&&) noexcept = default;

    // Query methods
    bool has_core_type(HalProgrammableCoreType core_type) const;
    uint32_t num_cores(HalProgrammableCoreType core_type) const;
    const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores() const;
    const CoreRangeSet& cores(HalProgrammableCoreType core_type) const;

private:
    void validate() const;

    ContextId context_id_;
    std::array<CoreRangeSet, NumHalProgrammableCoreTypes> cores_;
};

}  // namespace tt::tt_metal
