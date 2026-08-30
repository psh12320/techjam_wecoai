"""Runtime boundary tests for generated candidate code."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aide.interpreter import ExecutionPolicy, Interpreter


def _policy(workspace: Path) -> ExecutionPolicy:
    working = workspace / "working"
    input_dir = workspace / "input"
    working.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    return ExecutionPolicy(
        read_roots=(str(workspace), str(working), sys.prefix, sys.base_prefix),
        write_roots=(str(working),),
        write_files=(str(workspace / "runfile.py"),),
        environment={
            "AIDE_SEED": "7",
            "PATH": os.environ.get("PATH", ""),
            "SystemRoot": os.environ.get("SystemRoot", ""),
            "TEMP": str(working),
            "TMP": str(working),
        },
    )


def test_candidate_environment_is_scrubbed_and_working_write_is_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-child")
    interpreter = Interpreter(tmp_path, timeout=30, execution_policy=_policy(tmp_path))
    try:
        result = interpreter.run(
            "import os\n"
            "from pathlib import Path\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            "assert os.environ['AIDE_SEED'] == '7'\n"
            "Path('working/ok.txt').write_text('ok')\n"
            "print('secure')\n"
        )
    finally:
        interpreter.cleanup_session()
    assert result.exc_type is None
    assert (tmp_path / "working" / "ok.txt").read_text() == "ok"


def test_candidate_parent_read_network_and_subprocess_are_blocked(
    tmp_path: Path,
) -> None:
    (tmp_path.parent / "secret.txt").write_text("secret")
    cases = (
        "open('../secret.txt').read()",
        "import socket; socket.socket()",
        "import subprocess; subprocess.run(['cmd', '/c', 'echo', 'bad'])",
    )
    for code in cases:
        interpreter = Interpreter(
            tmp_path, timeout=30, execution_policy=_policy(tmp_path)
        )
        try:
            result = interpreter.run(code)
        finally:
            interpreter.cleanup_session()
        assert result.exc_type == "PermissionError"


def test_candidate_can_import_advertised_ml_runtime(tmp_path: Path) -> None:
    interpreter = Interpreter(
        tmp_path, timeout=60, execution_policy=_policy(tmp_path)
    )
    try:
        result = interpreter.run(
            "import numpy, pandas, torch, catboost\n"
            "torch.set_num_threads(1)\n"
            "print(numpy.__version__, pandas.__version__, torch.__version__, catboost.__version__)\n"
        )
    finally:
        interpreter.cleanup_session()
    assert result.exc_type is None
    assert "Execution time:" in "".join(result.term_out)
