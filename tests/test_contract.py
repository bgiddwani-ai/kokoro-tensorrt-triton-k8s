import os
import unittest
from unittest.mock import patch

import numpy as np

from kokoro_triton.cache import cache_key
from kokoro_triton.contract import expansion_indices, validate_inputs, warm_inputs
from kokoro_triton.settings import Settings


class ContractTests(unittest.TestCase):
    def test_valid_boundaries(self):
        for size in (4, 500):
            validate_inputs(warm_inputs(size))

    def test_reject_invalid_requests_before_gpu_execution(self):
        changes = [
            ("tokens", np.array([[0, 178, 1, 0]], np.int64)),
            ("tokens", np.array([[0, -1, 1, 0]], np.int64)),
            ("tokens", np.zeros((2, 4), np.int64)),
            ("tokens", np.zeros((1, 501), np.int64)),
            ("tokens", np.ones((1, 4), np.int64)),
            ("style", np.zeros((256,), np.float32)),
            ("style", np.full((1, 256), np.nan, np.float32)),
            ("speed", np.array([0.0], np.float32)),
            ("speed", np.array([np.inf], np.float32)),
        ]
        for name, value in changes:
            with self.subTest(name=name, shape=value.shape):
                inputs = warm_inputs()
                inputs[name] = value
                with self.assertRaises(ValueError):
                    validate_inputs(inputs)

    def test_expansion_preserves_token_order_and_checks_allocation_limit(self):
        np.testing.assert_array_equal(
            expansion_indices(np.array([1, 3, 2])), [0, 1, 1, 1, 2, 2]
        )
        for durations in (
            [],
            [0, 1],
            [-1, 3],
            [2001],
            [1500, 1500],
            [2**63 - 1, 2**63 - 1],
        ):
            with self.subTest(durations=durations), self.assertRaises(ValueError):
                expansion_indices(np.asarray(durations, np.int64))

    def test_cache_changes_for_hardware_runtime_and_shape_range(self):
        identity = {"gpu_name": "example", "runtime": "v1", "max_frames": 2000}
        self.assertEqual(
            cache_key(identity), cache_key(dict(reversed(list(identity.items()))))
        )
        for field, value in (
            ("gpu_name", "other"),
            ("runtime", "v2"),
            ("max_frames", 1000),
        ):
            self.assertNotEqual(
                cache_key(identity), cache_key({**identity, field: value})
            )

    def test_invalid_configuration(self):
        for name, value in (
            ("KOKORO_WORKERS", "0"),
            ("KOKORO_MAX_TOKENS", "501"),
            ("KOKORO_MAX_FRAMES", "2001"),
            ("KOKORO_WORKSPACE_MIB", "128"),
        ):
            with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                Settings.from_env()
