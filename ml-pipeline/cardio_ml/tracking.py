"""Traceability: every data step, figure, table, metric and model is recorded.

The RunTracker writes, per tier, a folder like::

    outputs/clinical/
        figures/      fig_01_class_balance.png, ...
        tables/       csv tables used in the report
        models/       joblib files + sha256 checksums
        run_manifest.json   machine-readable record of the whole run
        TRACEABILITY.md     human-readable summary for the report
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from . import config


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LineageStep:
    step: str
    rows_before: int
    rows_after: int
    detail: str

    @property
    def removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class RunTracker:
    tier: str
    root: Path = field(default=None)
    lineage: list[LineageStep] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    models: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    inputs: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root or config.OUTPUTS / self.tier)
        for sub in ("figures", "tables", "models"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.started = datetime.now().isoformat(timespec="seconds")

    # ---------------------------------------------------------------- records
    def log_input(self, path: Path, description: str) -> None:
        path = Path(path)
        entry = {"path": str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path),
                 "description": description}
        if path.is_file():
            entry["sha256"] = file_sha256(path)
            entry["bytes"] = path.stat().st_size
        self.inputs.append(entry)

    def log_step(self, step: str, rows_before: int, rows_after: int, detail: str = "") -> None:
        self.lineage.append(LineageStep(step, rows_before, rows_after, detail))
        print(f"[{self.tier}] {step}: {rows_before:,} -> {rows_after:,} rows. {detail}")

    def log_param(self, key: str, value) -> None:
        self.params[key] = value

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"[{self.tier}] NOTE: {text}")

    def log_metrics(self, name: str, values: dict) -> None:
        self.metrics[name] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in values.items()}

    # ---------------------------------------------------------------- outputs
    def save_figure(self, fig, slug: str, caption: str) -> Path:
        """Save a numbered figure and record its caption and plain-language reading guide."""
        from .figure_guide import guide_for
        number = len(self.figures) + 1
        path = self.root / "figures" / f"fig_{number:02d}_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        entry = {"number": number, "file": path.name, "slug": slug, "caption": caption}
        guide = guide_for(slug, self.tier.removesuffix("_fast"))
        if guide:
            entry["guide"] = guide
        self.figures.append(entry)
        return path

    def save_table(self, df: pd.DataFrame, slug: str, caption: str, index: bool = False) -> Path:
        path = self.root / "tables" / f"{slug}.csv"
        df.to_csv(path, index=index)
        self.tables.append({"file": path.name, "caption": caption, "rows": len(df)})
        return path

    def save_model(self, obj, slug: str, description: str) -> Path:
        path = self.root / "models" / f"{slug}.joblib"
        joblib.dump(obj, path, compress=3)
        self.models.append({"file": path.name, "description": description,
                            "sha256": file_sha256(path), "bytes": path.stat().st_size})
        return path

    # ---------------------------------------------------------------- summary
    def lineage_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{**asdict(s), "removed": s.removed} for s in self.lineage])

    def finalize(self) -> Path:
        import numpy, scipy, sklearn, matplotlib, seaborn
        manifest = {
            "tier": self.tier,
            "started": self.started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "seed": config.SEED,
            "environment": {
                "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": numpy.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                "seaborn": seaborn.__version__,
            },
            "inputs": self.inputs,
            "params": self.params,
            "lineage": [{**asdict(s), "removed": s.removed} for s in self.lineage],
            "metrics": self.metrics,
            "figures": self.figures,
            "tables": self.tables,
            "models": self.models,
            "notes": self.notes,
        }
        path = self.root / "run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        (self.root / "TRACEABILITY.md").write_text(self._markdown(manifest))
        return path

    def _markdown(self, m: dict) -> str:
        out = [f"# Traceability report — {self.tier} tier", "",
               f"Run: {m['started']} → {m['finished']} · seed {m['seed']} · "
               f"Python {m['environment']['python']} · scikit-learn {m['environment']['scikit-learn']}", ""]
        if m["inputs"]:
            out += ["## Inputs", "", "| File | Description | SHA-256 |", "|---|---|---|"]
            out += [f"| `{i['path']}` | {i['description']} | `{i.get('sha256', '-')[:16]}…` |" for i in m["inputs"]]
            out.append("")
        out += ["## Data lineage", "", "| # | Step | Rows before | Rows after | Removed | Detail |",
                "|---|---|---|---|---|---|"]
        out += [f"| {i} | {s['step']} | {s['rows_before']:,} | {s['rows_after']:,} | {s['removed']:,} | {s['detail']} |"
                for i, s in enumerate(m["lineage"], 1)]
        out.append("")
        if m["params"]:
            out += ["## Parameters", "", "```json", json.dumps(m["params"], indent=2, default=str), "```", ""]
        if m["metrics"]:
            metric_df = pd.DataFrame(m["metrics"]).T
            out += ["## Metrics (held-out test set)", "", metric_df.round(4).to_markdown(), ""]
        out += ["## Figures", "", "Each figure with its reading guide (what it shows · how to read it).", ""]
        for f in m["figures"]:
            out.append(f"{f['number']}. `{f['file']}` — **{f['caption']}**")
            if "guide" in f:
                out.append(f"   - *What it shows:* {f['guide']['what']['en']}")
                out.append(f"   - *How to read it:* {f['guide']['read']['en']}")
        out.append("")
        if m["tables"]:
            out += ["## Tables", ""] + [f"- `{t['file']}` — {t['caption']}" for t in m["tables"]] + [""]
        if m["models"]:
            out += ["## Exported models", "", "| File | Description | SHA-256 |", "|---|---|---|"]
            out += [f"| `{x['file']}` | {x['description']} | `{x['sha256'][:16]}…` |" for x in m["models"]]
            out.append("")
        if m["notes"]:
            out += ["## Notes", ""] + [f"- {n}" for n in m["notes"]] + [""]
        return "\n".join(out)
