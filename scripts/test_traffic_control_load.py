#!/usr/bin/env python3
"""Deterministic tests for the live dual-engine acceptance oracle."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import traffic_control_load as load


class TrafficControlLoadOracleTests(unittest.TestCase):
    def test_locus_metrics_accept_only_bounded_documented_labels(self) -> None:
        parsed = load.parse_locus_metrics(
            'locus_admission_requests_total{class="latency",tenant="alpha",outcome="admitted"} 2\n'
            'locus_traffic_controller_draining 0\n',
            max_samples=10,
        )
        self.assertEqual(parsed, {"samples": 2, "series": 2})

        with self.assertRaisesRegex(AssertionError, "undocumented labels"):
            load.parse_locus_metrics(
                'locus_admission_requests_total{request_id="unbounded"} 1\n',
                max_samples=10,
            )

    def test_engine_counter_delta_fails_closed_on_reset(self) -> None:
        engine = load.Engine("sglang", "http://engine-a", "model-a")
        before = {
            "prompt_tokens": {"value": 10.0},
            "generation_tokens": {"value": 20.0},
        }
        after = {
            "prompt_tokens": {"value": 13.0},
            "generation_tokens": {"value": 25.0},
        }
        self.assertEqual(
            load.engine_counter_deltas(engine, before, after, "normal load"),
            {"prompt_tokens": 3.0, "generation_tokens": 5.0},
        )
        after["prompt_tokens"]["value"] = 9.0
        with self.assertRaisesRegex(AssertionError, "counter reset"):
            load.engine_counter_deltas(engine, before, after, "normal load")

    def test_tenant_name_is_a_bounded_metric_identity_and_secret_stays_in_env(self) -> None:
        with patch.dict(os.environ, {"TEST_TENANT_KEY": "secret-value"}, clear=False):
            tenant = load.parse_tenant("alpha=TEST_TENANT_KEY")
        self.assertEqual(tenant.name, "alpha")
        self.assertEqual(tenant.api_key, "secret-value")

        with self.assertRaises(argparse.ArgumentTypeError):
            load.parse_tenant("bad tenant=TEST_TENANT_KEY")


if __name__ == "__main__":
    unittest.main()
