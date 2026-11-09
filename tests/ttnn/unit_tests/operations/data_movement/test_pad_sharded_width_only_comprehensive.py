# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for ttnn.pad sharded width-only factory bugs (#64834).

Tests cover:
- Bug 1: Missing grid/orientation/shard_height validation in can_use_sharded_optimized_factory
- Bug 2: BFLOAT16 2-byte stack buffer over-read and UINT16 incorrect packing
- Bug 3: Unchecked shard_spec() dereference on ND_SHARDED inputs
"""

import pytest
import torch
import ttnn


class TestPadShardedWidthOnlyBug1:
    """Bug 1: Missing validation checks cause silent data corruption when resharding."""

    def test_height_resharding_with_grid_change(self, device):
        """Test that height resharding with different core grids is rejected."""
        torch_input = torch.arange(1, 128 * 64 + 1, dtype=torch.bfloat16).reshape(1, 1, 128, 64)
        in_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        out_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        in_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(in_grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(out_grid, [64, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem_config)
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 64)], 0.0, memory_config=out_mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 64), value=0.0)
        assert torch.equal(torch_out, expected)

    def test_height_resharding_with_orientation_change(self, device):
        """Test that height resharding with different orientations is rejected."""
        torch_input = torch.arange(1, 64 * 32 + 1, dtype=torch.bfloat16).reshape(1, 1, 64, 32)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        in_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.COL_MAJOR),
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem_config)
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 32)], 0.0, memory_config=out_mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 32), value=0.0)
        assert torch.equal(torch_out, expected)

    def test_height_resharding_with_shard_height_change(self, device):
        """Test that height resharding with different shard heights is rejected."""
        torch_input = torch.arange(1, 64 * 32 + 1, dtype=torch.bfloat16).reshape(1, 1, 64, 32)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        in_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [64, 64], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem_config)
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 32)], 0.0, memory_config=out_mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 32), value=0.0)
        assert torch.equal(torch_out, expected)


class TestPadShardedWidthOnlyBug2:
    """Bug 2: BFLOAT16 stack buffer over-read and UINT16 incorrect packing."""

    @pytest.mark.parametrize("pad_value", [0.0, 1.5, -3.14, float("inf")])
    def test_bfloat16_padding_values(self, device, pad_value):
        """Test BFLOAT16 padding with various values."""
        torch_input = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_config)
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 32)], pad_value, memory_config=mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 32), value=pad_value)
        assert torch.allclose(torch_out, expected, rtol=0.01, atol=0.01)

    def test_uint16_padding_nonzero_value(self, device):
        """Test UINT16 padding with non-zero value."""
        torch_input = torch.ones((1, 1, 32, 18), dtype=torch.uint16)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_config)
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 14)], 7, memory_config=mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 14), value=7)
        assert torch.equal(torch_out, expected)


class TestPadShardedWidthOnlyBug3:
    """Bug 3: Unchecked shard_spec() dereference on ND_SHARDED inputs."""

    def test_nd_sharded_output_mem_config(self, device):
        """Test that ND_SHARDED output_mem_config doesn't crash."""
        torch_input = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        in_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # ND_SHARDED has no shard_spec
        out_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.ND_SHARDED,
            ttnn.BufferType.L1,
        )
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem_config)
        # Should not crash, should fall back to default factory
        tt_out = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 32)], 0.0, memory_config=out_mem_config)
        torch_out = ttnn.to_torch(tt_out)
        expected = torch.nn.functional.pad(torch_input, (0, 32), value=0.0)
        assert torch.equal(torch_out, expected)
