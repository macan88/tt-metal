# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.swin_l.common import (
    DEFAULT_INPUT_H,
    DEFAULT_INPUT_W,
    SWIN_L_EMBED_DIM,
    SWIN_L_DEPTHS,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
    SWIN_L_STAGE_CHANNELS,
)
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_swin_l_backbone_e2e(device, swin_l_ref, swin_l_ckpt_path, reset_seeds):
    from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
    from models.experimental.swin_l.tt.model_preprocessing import (
        load_backbone_weights,
        compute_attn_masks,
    )

    torch_input = torch.rand(1, 3, DEFAULT_INPUT_H, DEFAULT_INPUT_W)
    torch_feats = swin_l_ref.forward_backbone(torch_input)
    assert len(torch_feats) == 4

    parameters = load_backbone_weights(
        swin_l_ckpt_path,
        device,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
    )
    attn_masks = compute_attn_masks(DEFAULT_INPUT_H, DEFAULT_INPUT_W, 4, SWIN_L_WINDOW_SIZE, device)
    ttnn_model = TtSwinLBackbone(
        device,
        parameters,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        attn_masks=attn_masks,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_feats = ttnn_model(ttnn_input)
    assert len(ttnn_feats) == 4

    # --- Compare per-stage ---
    pcc_threshold = 0.97
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_feats, ttnn_feats)):
        ttnn_out = ttnn.to_torch(ttnn.from_device(ttnn_feat))
        ttnn_out = ttnn_out.permute(0, 3, 1, 2).contiguous()
        assert (
            ttnn_out.shape == torch_feat.shape
        ), f"Stage {i} shape mismatch: TTNN {ttnn_out.shape} vs PyTorch {torch_feat.shape}"
        passing, pcc_val = comp_pcc(torch_feat, ttnn_out, pcc_threshold)
        logger.info(
            f"Stage {i}: shape={list(torch_feat.shape)}, "
            f"channels={SWIN_L_STAGE_CHANNELS[i]}, PCC={pcc_val:.6f}, pass={passing}"
        )
        assert passing, f"Stage {i} PCC {pcc_val:.6f} < {pcc_threshold}"
