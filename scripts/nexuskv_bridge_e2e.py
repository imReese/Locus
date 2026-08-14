#!/usr/bin/env python3
"""Run the Locus handshake against a real, separate NexusKV bridge process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


LOCUS_ROOT = Path(__file__).resolve().parents[1]
LOCUS_FIXTURE = (
    LOCUS_ROOT / "crates" / "store" / "nexuskv" / "tests" / "fixtures" / "conformance.json"
)
EXTERNAL_TEST = "real_nexuskv_process_completes_locus_plan_and_import_handshake"


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_native_planner(nexuskv_root: Path) -> None:
    command = ["cargo", "rustc", "-p", "bindings-py", "--crate-type", "cdylib"]
    if sys.platform == "darwin":
        command.extend(["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"])
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    run(command, cwd=nexuskv_root / "rust", env=env)


def wait_for_bridge(process: subprocess.Popen[str], ready_file: Path) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if ready_file.exists():
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "locus.nexuskv-bridge.v1":
                raise RuntimeError(f"unexpected bridge readiness payload: {payload}")
            if payload.get("evidence_level") != "protocol":
                raise RuntimeError(f"bridge did not declare protocol evidence: {payload}")
            return str(payload["base_url"])
        return_code = process.poll()
        if return_code is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"bridge exited with {return_code} before readiness:\n{output}")
        time.sleep(0.05)
    raise TimeoutError("NexusKV bridge did not become ready within 30 seconds")


def terminate(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate(timeout=5)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Locus against a separate NexusKV bridge process"
    )
    parser.add_argument(
        "--nexuskv-root",
        type=Path,
        default=LOCUS_ROOT.parent / "NexusKV",
        help="path to a NexusKV checkout (default: sibling ../NexusKV)",
    )
    parser.add_argument("--skip-native-build", action="store_true")
    args = parser.parse_args()

    nexuskv_root = args.nexuskv_root.resolve()
    nexus_fixture = nexuskv_root / "tests" / "fixtures" / "locus_bridge" / "conformance.json"
    if LOCUS_FIXTURE.read_bytes() != nexus_fixture.read_bytes():
        raise RuntimeError("Locus and NexusKV bridge conformance fixtures are not byte-identical")

    fixture = json.loads(LOCUS_FIXTURE.read_text(encoding="utf-8"))
    expectations = fixture["expectations"]
    if expectations != {
        "matched_state": "nexus-state-1",
        "locality": "local",
        "receipt_namespace": "nexuskv.protocol-transfer-receipt.v1",
        "bytes_transferred": 0,
        "evidence_level": "protocol",
        "physical_transfer_verified": False,
    }:
        raise RuntimeError(f"unexpected conformance claim boundary: {expectations}")

    if not args.skip_native_build:
        build_native_planner(nexuskv_root)

    with tempfile.TemporaryDirectory(prefix="locus-nexuskv-bridge-") as temp_dir:
        ready_file = Path(temp_dir) / "ready.json"
        env = os.environ.copy()
        python_path = str(nexuskv_root / "python")
        if env.get("PYTHONPATH"):
            python_path = os.pathsep.join([python_path, env["PYTHONPATH"]])
        env["PYTHONPATH"] = python_path
        command = [
            sys.executable,
            "-m",
            "nexuskv.integrations.locus_bridge",
            "--listen",
            "127.0.0.1:0",
            "--fixture",
            str(nexus_fixture),
            "--ready-file",
            str(ready_file),
        ]
        print(f"+ {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=nexuskv_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            bridge_url = wait_for_bridge(process, ready_file)
            test_env = os.environ.copy()
            test_env["LOCUS_NEXUSKV_BRIDGE_URL"] = bridge_url
            run(
                [
                    "cargo",
                    "test",
                    "-p",
                    "locus-store-nexuskv",
                    "--test",
                    "external_bridge",
                    EXTERNAL_TEST,
                    "--",
                    "--ignored",
                    "--exact",
                    "--nocapture",
                ],
                cwd=LOCUS_ROOT,
                env=test_env,
            )
        finally:
            server_output = terminate(process)
            if server_output:
                print(server_output, end="", flush=True)

    print(
        json.dumps(
            {
                "schema_version": "locus.nexuskv-bridge.e2e-result.v1",
                "bridge_schema": "locus.nexuskv-bridge.v1",
                "external_process": True,
                "rust_matcher": True,
                "locus_plan_prepare_materialize_commit": True,
                "evidence_level": "protocol",
                "physical_transfer_verified": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
