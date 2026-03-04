// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <umd/device/types/arch.hpp>
#include "device/mock_device_util.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metalium_env.hpp"
#include "mesh_config.hpp"
#include "mesh_device.hpp"
#include "system_mesh.hpp"

namespace tt::tt_metal {

TEST(MetalContextIntegrationTest, CoexistingMockAndSiliconDevice) {
    // Create silicon mesh
    auto silicon_env = std::make_shared<MetaliumEnv>();
    ContextId silicon_context_id = tt::tt_metal::MetalContext::create_instance(silicon_env);
    log_info(tt::LogTest, "MetaliumEnv (silicon) created with context id {}", silicon_context_id);
    EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

    auto mesh_shape = tt::tt_metal::MetalContext::instance(silicon_context_id).get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create(silicon_context_id, mesh_device_config);
    log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

    // Create mock mesh device with 1 blackhole chip
    auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
        MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)));
    ContextId mock_context_id_bh_1 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_1);
    log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_1);
    EXPECT_NE(mock_context_id_bh_1, SILICON_CONTEXT_ID);

    auto mock_mesh_shape_bh_1 = tt::tt_metal::MetalContext::instance(mock_context_id_bh_1).get_system_mesh().shape();
    auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
    std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
        distributed::MeshDevice::create(mock_context_id_bh_1, mock_mesh_device_config_bh_1);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

    // Create mock mesh device with 2 blackhole chips
    auto mock_env_bh_2 = std::make_shared<MetaliumEnv>(
        MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
    ContextId mock_context_id_bh_2 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_2);
    log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_2);
    EXPECT_NE(mock_context_id_bh_2, SILICON_CONTEXT_ID);

    auto mock_mesh_shape_bh_2 = tt::tt_metal::MetalContext::instance(mock_context_id_bh_2).get_system_mesh().shape();
    auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
    std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_2 =
        distributed::MeshDevice::create(mock_context_id_bh_2, mock_mesh_device_config_bh_2);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

    EXPECT_NE(mock_context_id_bh_1, mock_context_id_bh_2);
    ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
    ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
}

}  // namespace tt::tt_metal
