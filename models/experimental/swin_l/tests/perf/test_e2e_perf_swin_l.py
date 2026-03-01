# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Swin-L Backbone end-to-end performance test (Pipeline API).

Runs Swin-L backbone with:
- Trace enabled
- 2 command queues

Usage:
    pytest models/experimental/swin_l/tests/perf/test_e2e_perf_swin_l.py -v -m models_performance_bare_metal
"""

import os
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.swin_l.common import (
    SWIN_L_DEPTHS,
    SWIN_L_EMBED_DIM,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
)
from models.experimental.swin_l.tt.model_preprocessing import compute_attn_masks, load_backbone_weights
from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)


def _get_swin_l_checkpoint_path():
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"

    candidates = [
        ckpt_dir / "dino_5scale_swin_l.pth",
        ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _create_swin_l_pipeline_model(ttnn_model, batch_size, input_h, padded_input_w, actual_input_w):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor on device"
        reshaped_input = ttnn.reshape(l1_input_tensor, (batch_size, 3, input_h, padded_input_w))
        input_for_model = ttnn.to_memory_config(reshaped_input, ttnn.DRAM_MEMORY_CONFIG)
        if reshaped_input.is_allocated():
            ttnn.deallocate(reshaped_input)
        if l1_input_tensor.is_allocated():
            ttnn.deallocate(l1_input_tensor)
        if padded_input_w != actual_input_w:
            input_for_model = ttnn.slice(
                input_for_model,
                [0, 0, 0, 0],
                [batch_size, 3, input_h, actual_input_w],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        features = ttnn_model(input_for_model)
        # Keep one tensor as model output to minimize host readback cost in perf mode.
        last_feature = features[-1]
        for feature in features[:-1]:
            if feature.is_allocated():
                ttnn.deallocate(feature)
        return last_feature

    return run


def _get_l1_input_memory_config(host_input):
    height_dim = host_input.shape[-2]
    width_dim = host_input.shape[-1]
    input_l1_core_grid = ttnn.CoreGrid(x=8, y=1)
    num_cores = input_l1_core_grid.num_cores
    if height_dim % num_cores != 0:
        input_l1_core_grid = ttnn.CoreGrid(x=4, y=1)
        num_cores = input_l1_core_grid.num_cores

    return ttnn.create_sharded_memory_config(
        shape=(height_dim // num_cores, width_dim),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@run_for_wormhole_b0()
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 32768,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [8])
@pytest.mark.parametrize("batch_size, expected_compile_time, expected_throughput_fps", [(1, 300.0, 1.0)])
def test_swin_l_backbone_e2e_perf_trace_2cq(
    device,
    num_iterations,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
):
    input_h = 800
    actual_input_w = 1333
    padded_input_w = ((actual_input_w + 31) // 32) * 32

    ckpt_path = _get_swin_l_checkpoint_path()
    if not ckpt_path:
        pytest.skip(
            "Checkpoint not found. Set SWIN_L_CKPT or download DINO-5scale Swin-L checkpoint under "
            "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
        )

    logger.info("Loading Swin-L TTNN parameters...")
    parameters = load_backbone_weights(
        ckpt_path,
        device,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
    )
    attn_masks = compute_attn_masks(input_h, actual_input_w, 4, SWIN_L_WINDOW_SIZE, device)
    ttnn_model = TtSwinLBackbone(
        device,
        parameters,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        attn_masks=attn_masks,
    )

    host_input_nchw = ttnn.from_torch(
        torch.rand(batch_size, 3, input_h, padded_input_w, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
    )
    host_input = ttnn.reshape(host_input_nchw, (1, 1, batch_size * 3 * input_h, padded_input_w))
    dram_input_memory_config = get_memory_config_for_persistent_dram_tensor(
        host_input.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    l1_input_memory_config = _get_l1_input_memory_config(host_input)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=_create_swin_l_pipeline_model(ttnn_model, batch_size, input_h, padded_input_w, actual_input_w),
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    logger.info("Compiling Swin-L e2e perf pipeline (trace + 2CQ)...")
    start = time.time()
    pipeline.compile(host_input)
    end = time.time()
    compile_and_first_run_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running Swin-L e2e perf for {num_iterations} iterations...")
    inputs = [host_input] * num_iterations
    start = time.time()
    _ = pipeline.enqueue(inputs).pop_all()
    end = time.time()
    inference_time = (end - start) / num_iterations

    pipeline.cleanup()

    fps = batch_size / inference_time
    logger.info(f"Swin-L average inference time: {inference_time:.4f} s")
    logger.info(f"Swin-L throughput: {fps:.2f} FPS")

    prep_perf_report(
        model_name="ttnn_swin_l_backbone_trace_2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_and_first_run_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"{input_h}x{actual_input_w}_batch{batch_size}",
    )
