"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


@dataclass(frozen=True)
class ExecutionPolicy:
    """Process-local restrictions for untrusted generated Python."""

    read_roots: tuple[str, ...] = ()
    write_roots: tuple[str, ...] = ()
    write_files: tuple[str, ...] = ()
    environment: dict[str, str] | None = None
    deny_network: bool = True
    deny_process_creation: bool = True


def _path_within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _resolve_audit_path(value: Any, working_dir: Path) -> Path | None:
    if isinstance(value, int) or value is None:
        return None
    try:
        return Path(os.fsdecode(value)).resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return None


def _install_execution_audit_hook(policy: ExecutionPolicy, working_dir: Path) -> None:
    read_roots = tuple(Path(value).resolve() for value in policy.read_roots)
    write_roots = tuple(Path(value).resolve() for value in policy.write_roots)
    write_files = {Path(value).resolve() for value in policy.write_files}
    file_read_events = {"open", "os.chdir", "os.listdir", "os.scandir"}
    file_write_events = {
        "os.chmod",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.truncate",
        "os.unlink",
        "os.utime",
    }

    def ensure_read_allowed(value: Any) -> None:
        path = _resolve_audit_path(value, working_dir)
        if path is not None and not _path_within(path, read_roots + write_roots):
            raise PermissionError(f"candidate read blocked outside sandbox: {path}")

    def ensure_write_allowed(value: Any) -> None:
        path = _resolve_audit_path(value, working_dir)
        if (
            path is not None
            and path not in write_files
            and not _path_within(path, write_roots)
        ):
            raise PermissionError(f"candidate write blocked outside sandbox: {path}")

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if policy.deny_network and event.startswith("socket."):
            raise PermissionError(f"candidate network operation blocked: {event}")
        if policy.deny_process_creation and (
            event == "subprocess.Popen"
            or event == "os.system"
            or event.startswith("os.spawn")
            or event.startswith("os.posix_spawn")
            or event in {"os.fork", "os.forkpty"}
        ):
            raise PermissionError(f"candidate process creation blocked: {event}")
        if event == "open" and args:
            mode_or_flags = args[1] if len(args) > 1 else "r"
            writing = False
            if isinstance(mode_or_flags, str):
                writing = any(token in mode_or_flags for token in "wax+")
            elif isinstance(mode_or_flags, int):
                writing = bool(
                    mode_or_flags
                    & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
                )
            if writing:
                ensure_write_allowed(args[0])
            else:
                ensure_read_allowed(args[0])
        elif event in file_read_events and args:
            ensure_read_allowed(args[0])
        elif event in file_write_events and args:
            ensure_write_allowed(args[0])
            if event in {"os.rename", "os.replace"} and len(args) > 1:
                ensure_write_allowed(args[1])

    sys.addaudithook(hook)


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue, timeout=5):
        self.queue = queue
        self.timeout = timeout

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logger.warning("Queue write timed out")

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        execution_policy: ExecutionPolicy | None = None,
        max_memory_bytes: int | None = None,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.execution_policy = execution_policy
        self.max_memory_bytes = max_memory_bytes
        self.process: Process = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        if self.execution_policy is not None:
            if self.execution_policy.environment is not None:
                os.environ.clear()
                os.environ.update(self.execution_policy.environment)
            _install_execution_audit_hook(self.execution_policy, self.working_dir)

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        # `exec` with an empty globals dict leaves `__name__` to resolve through
        # builtins, where it is "builtins" -- so the standard
        # `if __name__ == "__main__":` guard is never true and the script's body
        # silently never runs (clean exit, no output, no exception). Mirror the
        # semantics of running the file directly. See #62.
        global_scope: dict = {
            "__name__": "__main__",
            "__file__": self.agent_file_name,
            "__builtins__": __builtins__,
        }
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name)

            # put EOF marker to indicate that we're done
            result_outq.put(None)

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self):
        if self.process is None:
            return
        try:
            descendants = []
            try:
                import psutil
            except ImportError:
                descendants = []
            else:
                try:
                    descendants = psutil.Process(self.process.pid).children(
                        recursive=True
                    )
                    for child in descendants:
                        child.terminate()
                except psutil.Error:
                    descendants = []
            self.process.terminate()
            self.process.join(timeout=2.0)

            if descendants:
                try:
                    _, alive = psutil.wait_procs(descendants, timeout=2.0)
                    for child in alive:
                        child.kill()
                    psutil.wait_procs(alive, timeout=2.0)
                except psutil.Error:
                    logger.warning("Could not fully reap candidate descendants")

            if self.process.exitcode is None:
                logger.warning("Process failed to terminate, killing immediately")
                self.process.kill()
                self.process.join(timeout=3.0)

                if self.process.exitcode is None:
                    if os.name == "nt":
                        logger.error("Process refuses to die, using Windows taskkill")
                        subprocess.run(
                            [
                                "taskkill",
                                "/PID",
                                str(self.process.pid),
                                "/T",
                                "/F",
                            ],
                            check=False,
                            capture_output=True,
                            timeout=10,
                        )
                    else:
                        logger.error("Process refuses to die, using SIGKILL")
                        os.kill(self.process.pid, signal.SIGKILL)
                    self.process.join(timeout=5.0)
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None and self.process.exitcode is not None:
                self.process.close()
                self.process = None
            elif self.process is not None:
                logger.error(
                    "Timed-out child is still alive; leaving handle open safely"
                )

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            # Windows spawn plus antivirus scanning can exceed ten seconds even
            # when the child is healthy. A longer startup grace period prevents
            # false failures before the per-execution timeout begins.
            state = self.event_outq.get(timeout=60)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    running_time = time.time() - start_time
                else:
                    running_time = time.time() - start_time
                if self.max_memory_bytes is not None:
                    try:
                        import psutil
                    except ImportError:
                        resident = 0
                    else:
                        try:
                            process = psutil.Process(self.process.pid)
                            resident = process.memory_info().rss + sum(
                                child.memory_info().rss
                                for child in process.children(recursive=True)
                            )
                        except psutil.Error:
                            resident = 0
                    if resident > self.max_memory_bytes:
                        logger.warning(
                            "Execution exceeded memory limit of %s bytes",
                            self.max_memory_bytes,
                        )
                        self.cleanup_session()
                        state = (
                            None,
                            "MemoryLimitError",
                            {"resident_bytes": resident},
                            [],
                        )
                        exec_time = running_time
                        break
                if self.timeout is None:
                    continue
                if running_time > self.timeout:
                    logger.warning(f"Execution exceeded timeout of {self.timeout}s")
                    if os.name == "nt":
                        # Windows does not implement POSIX SIGINT delivery for a
                        # spawned multiprocessing child.  os.kill(..., SIGINT)
                        # can raise WinError 5 and abort the whole AIDE campaign,
                        # so use the existing bounded terminate/kill cleanup and
                        # return a normal TimeoutError node for autonomous repair.
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    if running_time > self.timeout + 5:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        start_collect = time.time()
        while True:
            try:
                # Add 5-second timeout for output collection
                if time.time() - start_collect > 5:
                    logger.warning("Output collection timed out")
                    break
                message = self.result_outq.get(timeout=1)
            except queue.Empty:
                continue
            if message is None:
                break
            output.append(message)

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        elif e_cls_name == "MemoryLimitError":
            output.append(
                f"MemoryLimitError: Execution exceeded the configured memory limit of {self.max_memory_bytes} bytes"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
