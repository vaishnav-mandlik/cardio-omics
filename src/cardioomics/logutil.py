from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "step": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}) or {})
        return json.dumps(payload, default=str)


def get_logger(step: str, logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(f"cardioomics.{step}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_JsonFormatter())
        logger.addHandler(stream)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w")
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)
    return logger


def log_kv(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})
