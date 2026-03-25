from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

from adapt.data_loader import load_settings


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json_signal(filename: str, payload: dict, settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    signals_dir = _ensure_dir(settings["paths"]["signals_dir"])
    outpath = signals_dir / filename

    payload = dict(payload)
    payload["written_at"] = datetime.now().isoformat()

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return outpath


def archive_json_signal(prefix: str, payload: dict, signal_date: str, settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    archive_dir = _ensure_dir(Path(settings["paths"]["signals_dir"]) / "archive")
    outpath = archive_dir / f"{signal_date}_{prefix}.json"

    payload = dict(payload)
    payload["written_at"] = datetime.now().isoformat()

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return outpath


def write_daily_summary(signal_date: str, summary_text: str, settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    reports_dir = _ensure_dir(settings["paths"]["reports_dir"])
    outpath = reports_dir / f"daily_summary_{signal_date}.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return outpath
