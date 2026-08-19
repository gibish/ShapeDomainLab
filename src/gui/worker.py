from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.core.config import DatasetConfig
from src.core.generator import generate_dataset
from src.core.quality import QualityReport, validate_dataset


class GenerationCancelled(Exception):
    pass


class GenerationWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self._config = config
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            output_dir = generate_dataset(self._config, self._on_progress)
            report = validate_dataset(output_dir, self._config)
            self.completed.emit(output_dir, report)
        except GenerationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if self._cancel_requested:
            raise GenerationCancelled()
        self.progress.emit(done, total, message)
