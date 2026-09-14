"""Asynchronous latest-frame debug image writer."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np
from PIL import Image

_LOGGER = logging.getLogger(__name__)

SaveImage = Callable[[np.ndarray, Path], None]
ImageBatch = Dict[str, np.ndarray]


class AsyncDebugImageWriter:
    """Save debug images off-thread while retaining only the latest pending batch."""

    def __init__(
        self,
        debug_dir: str | Path,
        *,
        save_image: SaveImage | None = None,
    ) -> None:
        self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._save_image = save_image or self._save_with_pillow
        self._queue: queue.Queue[ImageBatch | None] = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="debug-image-writer", daemon=True
        )
        self._thread.start()

    def submit(self, images: Mapping[str, np.ndarray]) -> None:
        """Copy and enqueue an image batch without waiting for image encoding or disk I/O."""
        try:
            batch = {name: image.copy() for name, image in images.items()}
            with self._state_lock:
                if self._closed:
                    return
                self._replace_pending(batch)
        except Exception:
            _LOGGER.exception("Failed to submit debug images")

    def close(self, timeout: float = 0.2) -> None:
        """Stop accepting batches and wait at most ``timeout`` seconds for the worker."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._replace_pending(None)
        self._thread.join(timeout=max(0.0, timeout))

    def _replace_pending(self, item: ImageBatch | None) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            for name, image in batch.items():
                try:
                    self._save_image(image, self._debug_dir / f"{name}.jpg")
                except Exception:
                    _LOGGER.exception("Failed to save debug image %s", name)

    @staticmethod
    def _save_with_pillow(image: np.ndarray, path: Path) -> None:
        Image.fromarray(image).save(path)
