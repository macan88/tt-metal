// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LM Head Sampling Unified Kernel: CCL Broadcast + Mcast + Matmul for Vocab Projection
//
// Single .cpp compiled for all three RISC processors (NCRISC, BRISC, TRISC).
// Compile-time role flags (is_input_core, is_mcast_receiver_core, is_matmul_core, skip_ccl)
// enable dead code elimination via `if constexpr`, so each core only runs its assigned path.
//
// Data flow:
//   1. CCL Broadcast (multi-device only): Sender device broadcasts input [1, K] to all
//      devices in the mesh via the fabric interconnect. Skipped when skip_ccl=true.
//   2. Mcast:  Sender core multicasts input [1, K] to all cores in the device grid
//   3. Matmul: Each matmul core computes [1, K] x [K, N_per_core] -> [1, N_per_core]
//   4. Argmax: Fused k=1 sampling across all matmul cores (and optionally across devices)
//   5. MTP Fusion (optional): Embedding lookup, RMSNorm, concat, mcast, EH projection matmul
//
// RISC responsibilities:
//   NCRISC: CCL broadcast writer (fabric multicast to remote devices) + mcast receiver
//           (semaphore wait + CB push) + sharded buffer setup (mcast_src on sender core, weight shards on
//           matmul cores) + argmax reader + MTP token transfer + embedding DRAM fetch + concat assembly
//   BRISC:  CCL broadcast reader + mcast sender + argmax writer + MTP mcast sender
//   TRISC:  RMSNorm compute + Matmul compute + MTP h/e RMSNorm + EH matmul
//
// CB layout (see op.py LMHeadSampling class for index definitions):
//   CB 0  (mcast_src):   Input tensor on sender core (tensor-backed).
//                         In multi-device mode, backed by intermediate_tensor (CCL broadcast
//                         destination). In single-device mode, backed by input_tensor directly.
//   CB 1  (mcast_dst):   Mcast destination / matmul in0 on all cores (intermediate)
//   CB 2  (matmul_in1):  Vocab weights on matmul cores (tensor-backed)
//   CB 9  (matmul_eh):   [MTP] EH projection weights on matmul cores (tensor-backed)
//   CB 10 (embedding):   [MTP] Embedding row (copy from mcast_eh_src_cb) for e_rmsnorm input
//   CB 11 (h_gamma):     [MTP] RMSNorm gamma for hidden states (tensor-backed)
//   CB 12 (e_gamma):     [MTP] RMSNorm gamma for embeddings (tensor-backed)
//   CB 13 (h_norm):      [MTP] h_rmsnorm output (intermediate)
//   CB 14 (e_norm):      [MTP] e_rmsnorm output (intermediate)
//   CB 15 (mcast_eh_src):[MTP] [h_norm|embedding] then [h_norm|e_norm]; embedding written via get_write_ptr
//   CB 16 (matmul_out):  Matmul output on matmul cores (tensor-backed)
//   CB 17 (matmul_eh_out):[MTP] EH matmul output (tensor-backed)
//   CB 18 (mcast_eh_dst):[MTP] Mcast destination for concat (intermediate)
//   CB 20 (mtp_token_bcast):[MTP] Token broadcast CB, tensor-backed by intermediate_tensor
//   CB 30 (bcast_pkt):   CCL broadcast packet buffer (multi-device mode only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/broadcast.hpp"
#include "../../../unified_kernels/argmax.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"

// Per-core role flags set by UnifiedCompileTimeCoreDescriptor in op.py.
// Each flag is specialized per core group at compile time, enabling if constexpr
// to eliminate dead code paths (e.g., sender-only code on receiver cores).
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
    static constexpr bool enable_argmax = get_named_compile_time_arg_val("enable_argmax") == 1;
    static constexpr bool is_argmax_core = is_matmul_core;
    static constexpr bool is_argmax_final_core = get_named_compile_time_arg_val("is_argmax_final_core") == 1;
    static constexpr bool is_argmax_mesh_sender_core =
        get_named_compile_time_arg_val("is_argmax_mesh_sender_core") == 1;
    static constexpr bool is_rmsnorm_core = get_named_compile_time_arg_val("is_rmsnorm_core") == 1;
    static constexpr bool enable_mtp = get_named_compile_time_arg_val("enable_mtp") == 1;
    static constexpr bool is_mtp_token_source_device =
        get_named_compile_time_arg_val("is_mtp_token_source_device") == 1;
    static constexpr bool is_eh_matmul_core = get_named_compile_time_arg_val("is_eh_matmul_core") == 1;
};

void kernel_main() {
// ============================================================================
// Per-RISC compile-time arg setup
// Each RISC receives different named compile-time args from op.py and
// constructs the appropriate Broadcast/Mcast/Matmul arg structs for its role.
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    uint32_t ncrisc_rt_arg_idx = 0;
    // --- NCRISC: CCL broadcast writer + mcast receiver + sharded buffer setup ---
    uint32_t mtp_token_addr = 0;

    // CCL Broadcast CTArgs type alias
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_num_targets_forward_direction"),
        get_named_compile_time_arg_val("bcast_num_targets_backward_direction"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        get_named_compile_time_arg_val("bcast_core_noc_x"),
        get_named_compile_time_arg_val("bcast_core_noc_y"),
        get_named_compile_time_arg_val("bcast_is_secondary_sender"),
        get_named_compile_time_arg_val("bcast_has_secondary_target"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("bcast_range_hops_forward"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("bcast_range_hops_backward")>;

    // CCL Broadcast writer runtime args (only populated when not skip_ccl)
    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{};
    if constexpr (!Core::skip_ccl) {
        bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // tensor_address0
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_bank_addr
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // wait_output_semaphore
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // reset_global_semaphore
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_y
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_wait_value
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_y
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // ring_index
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // secondary_sync_sem
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // num_connections (computed from len(dst_nodes))
        };
    }

    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Setup MTP tensor-backed CBs on sender core
    if constexpr (Core::enable_mtp) {
        if constexpr (Core::is_input_core) {
            constexpr uint32_t h_gamma_cb = get_named_compile_time_arg_val("h_gamma_cb");
            constexpr uint32_t rmsnorm_h_num_tiles = get_named_compile_time_arg_val("rmsnorm_h_num_tiles");
            unified_kernels::setup_sharded_buffer(h_gamma_cb, rmsnorm_h_num_tiles);
            constexpr uint32_t e_gamma_cb = get_named_compile_time_arg_val("e_gamma_cb");
            constexpr uint32_t rmsnorm_e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
            unified_kernels::setup_sharded_buffer(e_gamma_cb, rmsnorm_e_num_tiles);
        }
    }

    // Matmul reader args (NCRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};
    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::ReaderCTArgs<
        get_named_compile_time_arg_val("argmax_num_values"),
        get_named_compile_time_arg_val("argmax_winner_page_bytes"),
        get_named_compile_time_arg_val("argmax_num_senders"),
        get_named_compile_time_arg_val("argmax_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_receiver_semaphore_id"),
        get_named_compile_time_arg_val("argmax_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("argmax_mesh_mode"),
        get_named_compile_time_arg_val("argmax_stage1_sender"),
        get_named_compile_time_arg_val("argmax_stage1_receiver"),
        get_named_compile_time_arg_val("argmax_stage2_sender"),
        get_named_compile_time_arg_val("argmax_stage2_receiver"),
        get_named_compile_time_arg_val("argmax_stage1_slot_base_offset"),
        get_named_compile_time_arg_val("argmax_stage1_num_slots"),
        get_named_compile_time_arg_val("argmax_stage1_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_stage1_local_slot_offset"),
        get_named_compile_time_arg_val("argmax_stage2_slot_base_offset"),
        get_named_compile_time_arg_val("argmax_stage2_num_slots"),
        get_named_compile_time_arg_val("argmax_stage2_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_stage2_local_slot_offset"),
        get_named_compile_time_arg_val("argmax_mesh_local_send_slot_offset"),
        get_named_compile_time_arg_val("argmax_sender_idx"),
        get_named_compile_time_arg_val("argmax_socket_mode"),
        get_named_compile_time_arg_val("argmax_socket_cb"),
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes")>;

    // Matmul cores: register matmul_in1 CB (CB 2) backed by vocab weight shards
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
        // Note: EH projection weights (CB 9) are managed by DRAMStreamingMatmul reader
        // via cb_reserve_back/cb_push_back, so no setup_sharded_buffer needed here.
    }
#elif defined(COMPILE_FOR_BRISC)
    uint32_t brisc_rt_arg_idx = 0;
    // --- BRISC: CCL broadcast reader + mcast sender ---
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender")>;

    // CCL Broadcast reader runtime args (empty payload by design)
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{};

    // Template params: <num_cores, is_sender_in_receiver_grid, loopback>
    // loopback=false because sender does not consume its own multicast data
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_semaphore(get_named_compile_time_arg_val("mcast_data_sender_semaphore")),
        get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        Core::is_input_core ? get_read_ptr(mcast_src_cb) : 0,
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul writer args (BRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};
    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::WriterCTArgs<
        get_named_compile_time_arg_val("argmax_winner_page_bytes"),
        get_named_compile_time_arg_val("argmax_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("argmax_socket_mode"),
        get_named_compile_time_arg_val("argmax_socket_cb"),
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes")>;

#elif defined(COMPILE_FOR_TRISC)
    // --- TRISC: Matmul compute ---
    // CCL Broadcast CTArgs (no-op for TRISC)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};

    // Mcast is a no-op on TRISC (data movement handled by NCRISC/BRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul compute: [1, K] x [K, N_per_core] -> [1, N_per_core]
    // out_w (output tiles per core) is a compile-time template param for loop unrolling
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb")>;
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(numel))
    };

    using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w")>;

    // CB indices and tile count from op.py compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("matmul_in0");  // CB 1: mcast_dst
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");  // CB 2: vocab weights
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("matmul_out");  // CB 16: matmul output
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");

    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        .in0 = in0_cb,
        .in1 = in1_cb,
        .out = out_cb,
        .k_num_tiles = num_tiles_k,
    };
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Phase 0 (multi-device only): CCL Broadcast — replicate input from sender
    // device to all devices in the mesh via the fabric interconnect.
    // Only the input core participates (writer on NCRISC, reader on BRISC).
    // After this phase, every device has the input in its intermediate tensor
    // (which backs CB 0 / mcast_src).
    // ========================================================================
    if constexpr (!Core::skip_ccl) {
        deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
        {
            DeviceZoneScopedN("CCL_BROADCAST");
            bcast(bcast_args);
        }
    }

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers so BRISC/TRISC can access tensor data.
    // Sender core: register RMSNorm input CB backed by input_tensor (skip_ccl)
    // or intermediate_tensor (CCL mode, where broadcast placed the data)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
    }
#endif

    // ========================================================================
    // Phase 0.5: First RMSNorm (TRISC only)
    // When MTP is enabled, don't pop the input so CB 0 data persists for h_rmsnorm reuse.
    // ========================================================================
#if defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, !Core::enable_mtp> rmsnorm;
    {
        DeviceZoneScopedN("RMSNORM");
        rmsnorm(rmsnorm_args);
    }
#endif

    // ========================================================================
    // Phase 1: Mcast — multicast input from sender core to all device cores
    //
    // Template params: <CTArgs, IsSender, IsMcastGridCore, IsReceiverCore, PopSrc>
    //   IsMcastGridCore: participates in semaphore-based sync (all receivers)
    //   IsReceiverCore:  performs CB reserve/push for incoming data (all receivers)
    //   PopSrc:          sender pops mcast_src CB after send (frees tensor-backed buffer)
    // ========================================================================
    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
            mcast;

    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }
    mcast.teardown();

    // ========================================================================
    // Phase 2: Matmul — each matmul core computes local GEMM with its weight shard
    //
    // Template params: <CTArgs, IsActive, PopIn0, PopIn1>
    //   IsActive: only matmul cores execute; others are no-ops
    //   PopIn0:   pop mcast_dst CB (CB 1) after read (frees intermediate buffer)
    //   PopIn1:   pop matmul_in1 CB (CB 2) after read (frees weight buffer)
    // ========================================================================
    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, true> matmul;
    {
        DeviceZoneScopedN("MATMUL");
        matmul(matmul_args);
    }

    // ========================================================================
    // [MTP] h_rmsnorm on TRISC — starts immediately after the LM head matmul,
    // overlapping with argmax on NCRISC/BRISC. CB 0 still has hidden states
    // (first RMSNorm used pop_input=false when MTP is enabled).
    // ========================================================================
#if defined(COMPILE_FOR_TRISC)
    if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
        using HRMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
            get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
            get_named_compile_time_arg_val("rmsnorm_h_num_tiles"),
            get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
            get_named_compile_time_arg_val("rmsnorm_h_input_cb"),
            get_named_compile_time_arg_val("rmsnorm_h_gamma_cb"),
            get_named_compile_time_arg_val("rmsnorm_h_output_cb")>;
        deepseek_b1_ops::RMSNorm::Op<HRMSNormCTArgs, true, true> h_rmsnorm;
        cb_reserve_back(
            get_named_compile_time_arg_val("rmsnorm_h_input_cb"),
            get_named_compile_time_arg_val("rmsnorm_h_num_tiles"));
        cb_push_back(
            get_named_compile_time_arg_val("rmsnorm_h_input_cb"),
            get_named_compile_time_arg_val("rmsnorm_h_num_tiles"));
        {
            DeviceZoneScopedN("MTP_H_RMSNORM");
            h_rmsnorm(rmsnorm_args);
        }
    }
#endif

    // ========================================================================
    // Phase 3: Argmax Sampling (NCRISC + BRISC only)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#if defined(COMPILE_FOR_NCRISC)
        constexpr uint32_t gather_cb = get_named_compile_time_arg_val("argmax_gather_cb");
        uint32_t scores_addr = 0;
        if constexpr (Core::is_matmul_core) {
            // Matmul (TRISC) pushes matmul_out CB; wait before NCRISC consumes scores.
            constexpr uint32_t matmul_out_cb = get_named_compile_time_arg_val("matmul_out");
            constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
            cb_wait_front(matmul_out_cb, out_w);
            scores_addr = get_read_ptr(matmul_out_cb);
        }
        deepseek_b1_ops::Sampling::ReaderArgs sampling_args{
            scores_addr,
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_write_ptr(gather_cb),
        };
#elif defined(COMPILE_FOR_BRISC)
    deepseek_b1_ops::Sampling::WriterArgs sampling_args{
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
    };
#endif
        // k=1 fast path: fused sampling invocation matches micro-op style.
        deepseek_b1_ops::Sampling::
            Op<ArgmaxCTArgs, Core::is_matmul_core, Core::is_argmax_final_core, Core::is_argmax_mesh_sender_core>
                sampling_op;
        {
            DeviceZoneScopedN("ARGMAX");
            sampling_op(sampling_args);
        }

#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_matmul_core) {
            constexpr uint32_t matmul_out_cb = get_named_compile_time_arg_val("matmul_out");
            constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
            cb_pop_front(matmul_out_cb, out_w);
        }

        // ================================================================
        // [MTP] Token transfer: argmax_final_core writes token to input_core
        //
        // All MTP-enabled cores consume 3 args (input_core_noc_x/y, mtp_token_addr)
        // to keep arg indices aligned. Only argmax_final_core on the source device
        // actually performs the NOC write. Only input_core on the source device
        // waits on the semaphore.
        // ================================================================
        if constexpr (Core::enable_mtp) {
            uint32_t input_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
            uint32_t input_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
            mtp_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);

            if constexpr (Core::is_argmax_final_core && Core::is_mtp_token_source_device) {
                // Token is at sampling_args.output_addr (argmax wrote it there).
                // NOC write it to input_core's intermediate buffer (= mtp_token_addr).
                uint64_t dst = get_noc_addr(input_core_noc_x, input_core_noc_y, mtp_token_addr);
                noc_async_write(sampling_args.output_addr, dst, 4);
                noc_async_write_barrier();

                uint64_t sem_addr = get_noc_addr(
                    input_core_noc_x,
                    input_core_noc_y,
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_inc(sem_addr, 1);
            }

            if constexpr (Core::is_input_core && Core::is_mtp_token_source_device) {
                volatile tt_l1_ptr uint32_t* mtp_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_wait(mtp_ready_sem, 1);
                noc_semaphore_set(mtp_ready_sem, 0);
            }
        }
#endif
#endif  // COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC

        // ========================================================================
        // [MTP] CCL Broadcast token from source device to ALL devices.
        // After this phase, every device has the token at mtp_token_addr
        // (= intermediate_tensor.buffer_address()). Skipped in single-device mode.
        // ========================================================================
        if constexpr (Core::enable_mtp && !Core::skip_ccl) {
#if defined(COMPILE_FOR_NCRISC)
            using MtpBcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
                get_named_compile_time_arg_val("mtp_bcast_cb0_id"),
                get_named_compile_time_arg_val("mtp_bcast_num_pages_to_read"),
                get_named_compile_time_arg_val("mtp_bcast_tensor0_page_size"),
                get_named_compile_time_arg_val("mtp_bcast_num_targets_forward_direction"),
                get_named_compile_time_arg_val("mtp_bcast_num_targets_backward_direction"),
                get_named_compile_time_arg_val("mtp_bcast_is_sender"),
                get_named_compile_time_arg_val("mtp_bcast_core_noc_x"),
                get_named_compile_time_arg_val("mtp_bcast_core_noc_y"),
                get_named_compile_time_arg_val("mtp_bcast_is_secondary_sender"),
                get_named_compile_time_arg_val("mtp_bcast_has_secondary_target"),
                get_named_compile_time_arg_val("mtp_bcast_start_distance_in_hops_forward"),
                get_named_compile_time_arg_val("mtp_bcast_range_hops_forward"),
                get_named_compile_time_arg_val("mtp_bcast_start_distance_in_hops_backward"),
                get_named_compile_time_arg_val("mtp_bcast_range_hops_backward")>;

            deepseek_b1_ops::Broadcast::WriterArgs mtp_bcast_args{};
            if constexpr (Core::is_input_core) {
                mtp_bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // tensor_address0 = mtp_token_addr
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_bank_addr
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // wait_output_semaphore
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // reset_global_semaphore
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_x
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_y
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_wait_value
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_x
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_y
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // ring_index
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // secondary_sync_sem
                    get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // num_connections
                };
            }
            deepseek_b1_ops::Broadcast::Op<MtpBcastCTArgs, Core::is_input_core> mtp_bcast;
            {
                DeviceZoneScopedN("MTP_TOKEN_CCL_BCAST");
                mtp_bcast(mtp_bcast_args);
            }
#elif defined(COMPILE_FOR_BRISC)
        using MtpBcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
            get_named_compile_time_arg_val("mtp_bcast_cb0_id"),
            get_named_compile_time_arg_val("mtp_bcast_num_pages_to_read"),
            get_named_compile_time_arg_val("mtp_bcast_is_sender")>;
        deepseek_b1_ops::Broadcast::ReaderArgs mtp_bcast_args{};
        deepseek_b1_ops::Broadcast::Op<MtpBcastCTArgs, Core::is_input_core> mtp_bcast;
        mtp_bcast(mtp_bcast_args);
#elif defined(COMPILE_FOR_TRISC)
        using MtpBcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
        deepseek_b1_ops::Broadcast::ComputeArgs mtp_bcast_args{};
        deepseek_b1_ops::Broadcast::Op<MtpBcastCTArgs, Core::is_input_core> mtp_bcast;
        mtp_bcast(mtp_bcast_args);
#endif
        }

        // ========================================================================
        // [MTP] Embedding lookup (NCRISC on input_core, all devices)
        // After broadcast (or semaphore in single-device), every device has the token
        // at mtp_token_addr. Write embedding directly to the L1 back of mcast_eh_src_cb,
        // then copy that region to embedding_cb so e_rmsnorm can read from front.
        // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_input_core) {
            constexpr uint32_t embedding_size_bytes = get_named_compile_time_arg_val("embedding_size_bytes");
            constexpr uint32_t emb_cb = get_named_compile_time_arg_val("embedding_cb");
            constexpr uint32_t e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");

            uint32_t embedding_base = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);

            // Embedding tensor is ROW_MAJOR_LAYOUT, DRAM interleaved.
            // Each token's embedding is one page of embedding_size_bytes.
            const InterleavedAddrGen<true> embedding_addr_gen = {
                .bank_base_address = embedding_base,
                .page_size = embedding_size_bytes,
            };

            uint32_t token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_token_addr);
            cb_reserve_back(emb_cb, e_num_tiles);
            uint64_t dram_addr = embedding_addr_gen.get_noc_addr(token_id);
            noc_async_read(dram_addr, get_write_ptr(emb_cb), embedding_size_bytes);
            noc_async_read_barrier();
            cb_push_back(emb_cb, e_num_tiles);
        }
#endif

        // ========================================================================
        // [MTP] Second mcast receiver (NCRISC on non-sender cores)
        // All devices receive the concat [h_norm|e_norm] mcast.
        // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && !Core::is_input_core && Core::is_mcast_receiver_core) {
            using McastEhReceiverCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs mcast_eh_recv_args{
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
                get_named_compile_time_arg_val("mcast_eh_dst_cb"),
                get_named_compile_time_arg_val("mcast_eh_dst_num_pages"),
            };
            deepseek_b1_ops::Mcast::
                Op<McastEhReceiverCTArgs, false, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, false>
                    mcast_eh_receiver;
            mcast_eh_receiver(mcast_eh_recv_args);
        }
#endif

        // ========================================================================
        // [MTP] Concat [h_norm|e_norm] (NCRISC on input_core)
        // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_input_core) {
            constexpr uint32_t eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
            constexpr uint32_t h_norm_cb = get_named_compile_time_arg_val("rmsnorm_h_input_cb");
            constexpr uint32_t e_norm_cb = get_named_compile_time_arg_val("rmsnorm_e_output_cb");
            constexpr uint32_t h_tiles = get_named_compile_time_arg_val("rmsnorm_h_num_tiles");
            constexpr uint32_t e_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
            constexpr uint32_t tile_size_bytes = get_named_compile_time_arg_val("mcast_eh_tile_size_bytes");
            constexpr uint32_t num_uint32_per_tile = tile_size_bytes / 4;  // 4 bytes per uint32_t
            constexpr uint32_t total_num_tiles = h_tiles + e_tiles;

            cb_wait_front(e_norm_cb, e_tiles);
            cb_wait_front(h_norm_cb, h_tiles);
            constexpr uint32_t h_size = h_tiles * num_uint32_per_tile;
            constexpr uint32_t e_size = e_tiles * num_uint32_per_tile;
            volatile tt_l1_ptr uint32_t* h_norm_cb_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(h_norm_cb));
            volatile tt_l1_ptr uint32_t* e_norm_cb_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(e_norm_cb));
            cb_reserve_back(eh_src_cb, total_num_tiles);
            volatile tt_l1_ptr uint32_t* eh_src_cb_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eh_src_cb));
            for (uint32_t i = 0; i < h_size; ++i) {
                eh_src_cb_ptr[i] = h_norm_cb_ptr[i];
            }
            for (uint32_t i = 0; i < e_size; ++i) {
                eh_src_cb_ptr[h_size + i] = e_norm_cb_ptr[i];
            }
            cb_push_back(eh_src_cb, total_num_tiles);
        }

#endif

        // ========================================================================
        // [MTP] Second mcast sender (BRISC on input_core)
        // Wait for NCRISC concat signal, then mcast to all matmul cores.
        // ========================================================================
#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::enable_mtp && Core::is_input_core) {
            using McastEhSenderCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
                get_named_compile_time_arg_val("mcast_eh_num_cores"),
                get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
                false>;
            constexpr uint32_t eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
            constexpr uint32_t eh_dst_cb = get_named_compile_time_arg_val("mcast_eh_dst_cb");
            deepseek_b1_ops::Mcast::SenderArgs mcast_eh_send_args{
                get_named_compile_time_arg_val("mcast_eh_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_end_y"),
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_sender_semaphore")),
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
                get_named_compile_time_arg_val("mcast_eh_data_size_bytes"),
                eh_src_cb,
                get_named_compile_time_arg_val("mcast_eh_src_num_pages"),
                Core::is_input_core ? get_read_ptr(eh_src_cb) : 0,
                get_write_ptr(eh_dst_cb),
            };
            cb_wait_front(eh_src_cb, get_named_compile_time_arg_val("mcast_eh_src_num_pages"));
            deepseek_b1_ops::Mcast::
                Op<McastEhSenderCTArgs, true, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
                    mcast_eh_sender;
            mcast_eh_sender.init(mcast_eh_send_args);
            mcast_eh_sender(mcast_eh_send_args);
            mcast_eh_sender.teardown();
        }
#endif

        // ========================================================================
        // [MTP] e_rmsnorm on TRISC (after embedding arrives in CB)
        // ========================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
            using ERMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
                get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
                get_named_compile_time_arg_val("rmsnorm_e_num_tiles"),
                get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
                get_named_compile_time_arg_val("rmsnorm_e_input_cb"),
                get_named_compile_time_arg_val("rmsnorm_e_gamma_cb"),
                get_named_compile_time_arg_val("rmsnorm_e_output_cb")>;
            deepseek_b1_ops::RMSNorm::ComputeArgs e_rmsnorm_args{
                get_common_arg_val<uint32_t>(0),  // epsilon (same)
                get_common_arg_val<float>(2),     // e_scalar = 1/sqrt(embedding_dim)
            };
            deepseek_b1_ops::RMSNorm::Op<ERMSNormCTArgs, Core::is_rmsnorm_core, true> e_rmsnorm;
            {
                DeviceZoneScopedN("MTP_E_RMSNORM");
                e_rmsnorm(e_rmsnorm_args);
            }
        }
#endif

        // ========================================================================
        // [MTP] EH matmul on EH_MATMUL cores using DRAM streaming
        // TRISC: compute (uses matmul_eh_subblock_w etc. — TRISC-only CT args).
        // NCRISC: DRAM reader + between-iteration cb_wait/cb_pop/setup_sharded_buffer.
        // BRISC: no-op (excluded via preprocessor).
        // ========================================================================
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_eh_matmul_core) {
            constexpr uint32_t eh_in0_cb = get_named_compile_time_arg_val("matmul_eh_in0");
            constexpr uint32_t eh_in1_cb = get_named_compile_time_arg_val("matmul_eh_in1");
            constexpr uint32_t eh_out_cb = get_named_compile_time_arg_val("matmul_eh_out");
            constexpr uint32_t eh_num_tiles_k = get_named_compile_time_arg_val("matmul_eh_k_num_tiles");
            constexpr uint32_t eh_out_w = get_named_compile_time_arg_val("matmul_eh_out_w");
            constexpr uint32_t eh_num_loop_iters = get_named_compile_time_arg_val("eh_matmul_num_loop_iters");
            constexpr uint32_t eh_cb_in1_buf_addr = get_named_compile_time_arg_val("eh_matmul_cb_in1_buf_addr");

#if defined(COMPILE_FOR_TRISC)
            using EHDRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
                eh_in0_cb,
                eh_in1_cb,
                eh_out_cb,
                get_named_compile_time_arg_val("matmul_eh_subblock_k"),
                eh_out_w,
                get_named_compile_time_arg_val("matmul_eh_subblock_w"),
                get_named_compile_time_arg_val("matmul_eh_num_subblocks_k"),
                1,
                0,
                0>;
#elif defined(COMPILE_FOR_NCRISC)
            using EHDRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
                eh_in1_cb,
                eh_out_cb,
                get_named_compile_time_arg_val("matmul_eh_dram_in1_tensor_addr"),
                get_named_compile_time_arg_val("matmul_eh_dram_in1_page_size"),
                get_named_compile_time_arg_val("matmul_eh_dram_in1_num_pages"),
                get_named_compile_time_arg_val("matmul_eh_subblock_k"),
                eh_out_w,
                get_named_compile_time_arg_val("matmul_eh_dram_in1_block_size_bytes"),
                get_named_compile_time_arg_val("matmul_eh_out_num_tiles"),
                get_named_compile_time_arg_val("matmul_eh_num_subblocks_k"),
                get_named_compile_time_arg_val("matmul_eh_bank_id"),
                get_named_compile_time_arg_val("matmul_eh_vc")>;
#endif

            // matmul_eh_dram_in1_block_size_bytes: 15232 bytes
            for (uint32_t iter = 0; iter < eh_num_loop_iters; ++iter) {
                deepseek_b1_ops::DRAMStreamingMatmul::
                    Op<EHDRAMMMCTArgs, true, true, (eh_num_loop_iters > 1), eh_cb_in1_buf_addr>
                        eh_matmul;
                {
                    DeviceZoneScopedN("MTP_EH_DRAM_MATMUL");
                    eh_matmul();
                }

#if defined(COMPILE_FOR_NCRISC)
                // Between iterations: wait for compute to finish, pop output, re-setup in0
                if constexpr (eh_num_loop_iters > 1) {
                    if (iter < eh_num_loop_iters - 1) {
                        constexpr uint32_t out_num_tiles = get_named_compile_time_arg_val("matmul_eh_out_num_tiles");
                        cb_wait_front(eh_out_cb, out_num_tiles);
                        cb_pop_front(eh_out_cb, out_num_tiles);
                        unified_kernels::setup_sharded_buffer(eh_in0_cb, eh_num_tiles_k);
                    }
                }
#endif
            }
        }
#endif
}
