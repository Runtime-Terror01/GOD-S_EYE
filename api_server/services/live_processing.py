import threading
import time
import uuid
import logging
from typing import Any, Dict, Optional

import torch

from execution import PromptExecutor


class LiveFrameBuffer:

    def __init__(self):
        self._mutex = threading.RLock()
        self._frame: Optional[torch.Tensor] = None
        self._updated_event = threading.Event()

    def set_frame(self, frame: torch.Tensor):
        with self._mutex:
            self._frame = frame
            self._updated_event.set()

    def get_frame(self) -> Optional[torch.Tensor]:
        with self._mutex:
            return self._frame

    def wait_for_update(self, timeout: Optional[float] = None) -> bool:
        signaled = self._updated_event.wait(timeout)
        if signaled:
            self._updated_event.clear()
        return signaled


class LiveProcessingManager:

    def __init__(self, server):
        self.server = server
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running_lock = threading.RLock()
        self._is_running = False
        self._executor: Optional[PromptExecutor] = None
        self._prompt: Optional[Dict[str, Any]] = None
        self._extra_data: Dict[str, Any] = {}
        self._execute_outputs: list[str] = []
        self.frame_buffer = LiveFrameBuffer()

    def start(self, prompt: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None, execute_outputs: Optional[list[str]] = None):
        with self._running_lock:
            if self._is_running:
                raise RuntimeError("Live processing already running")
            self._stop_event.clear()
            self._prompt = prompt
            self._extra_data = extra_data or {}
            self._execute_outputs = list(execute_outputs or [])
            self._executor = PromptExecutor(self.server)
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._is_running = True
            self._thread.start()

    def stop(self):
        with self._running_lock:
            if not self._is_running:
                return
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._running_lock:
            self._is_running = False
            self._thread = None
            self._executor = None

    def is_running(self) -> bool:
        with self._running_lock:
            return self._is_running

    def _run_loop(self):
        assert self._executor is not None and self._prompt is not None
        logging.info("[Live] Starting live processing loop")
        while not self._stop_event.is_set():
            prompt_id = str(uuid.uuid4())
            try:
                self._executor.execute(self._prompt, prompt_id, self._extra_data, self._execute_outputs)
            except Exception as e:
                logging.exception("[Live] Error during live execution: %s", e)
                time.sleep(0.05)
            if self._stop_event.is_set():
                break
        logging.info("[Live] Live processing loop stopped")


# Singleton accessor, importable by custom nodes
_live_manager: Optional[LiveProcessingManager] = None


def get_live_manager(server=None) -> LiveProcessingManager:
    global _live_manager
    if _live_manager is None:
        if server is None:
            raise RuntimeError("LiveProcessingManager not initialized yet")
        _live_manager = LiveProcessingManager(server)
    return _live_manager



