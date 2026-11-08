"""Bounded thread orchestration of independent child processes, never model threads."""
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class ProcessRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = {}
        self.cancelled = threading.Event()

    def start(self, command, **kwargs):
        with self.lock:
            if self.cancelled.is_set():
                raise RuntimeError("Queue interrupted before child launch")
            process = subprocess.Popen(command, **kwargs)
            self.processes[process.pid] = process
            return process

    def unregister(self, process):
        with self.lock:
            self.processes.pop(process.pid, None)

    def terminate_all(self):
        # Under the same launch lock, prohibit races with not-yet-started workers.
        with self.lock:
            self.cancelled.set()
            running = list(self.processes.values())
        for process in running:
            if process.poll() is None:
                process.terminate()
        for process in running:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def schedule(items, worker, max_parallel, on_state, registry):
    """items: unique (name, payload) pairs; on_state runs only in caller thread.

    On a failure, launch no additional work and let already submitted workers
    finish. CTRL-C terminates only this invocation's registered subprocesses.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    if len({name for name, _ in items}) != len(items):
        raise ValueError("Duplicate scheduler job names")
    pending = iter(items)
    futures, completed, failed = {}, [], []
    pool = ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="fit-process")

    def report(status):
        on_state({"status": status, "active_jobs": sorted(futures.values()),
                  "completed_jobs": list(completed), "failed_jobs": list(failed)})

    def fill():
        while len(futures) < max_parallel:
            item = next(pending, None)
            if item is None:
                break
            name, payload = item
            futures[pool.submit(worker, name, payload)] = name

    try:
        fill()
        report("training")
        while futures:
            done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                name = futures.pop(future)
                try:
                    future.result()
                    completed.append(name)
                except Exception as error:
                    failed.append({"job": name, "error": str(error)})
            if not failed:
                fill()
            report("draining_after_failure" if failed and futures else "failed" if failed else "training")
    except KeyboardInterrupt:
        registry.terminate_all()
        report("interrupted")
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    report("failed" if failed else "finished")
    return {"completed": completed, "failed": failed}
