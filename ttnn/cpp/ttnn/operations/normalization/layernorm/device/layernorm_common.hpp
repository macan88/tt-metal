// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

// Creates a program config from tensor.
// - If tensor has shard_spec, creates a sharded config derived from it (using tensor's tile shape)
// - Otherwise, returns a default interleaved config
LayerNormProgramConfig create_layernorm_program_config(const Tensor& tensor);

}  // namespace ttnn::prim
