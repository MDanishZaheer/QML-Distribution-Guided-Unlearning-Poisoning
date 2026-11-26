"""Execute only the notebook's guarded saved-results mode and record this review.

No training flags are enabled or edited. Read verification, tables and figures
use existing numeric artifacts. The scientific training records stay immutable.
"""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def review(root):
    root = Path(root).resolve()
    target = root / "main_project.ipynb"
    notebook = nbformat.read(target, as_version=4)
    settings = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        try:
                            settings[name.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
    flags = ("DOWNLOAD_MNIST", "RUN_PRIMARY_STUDY", "RUN_ABLATIONS", "RUN_FIXED_WEIGHT_NOISE", "RUN_NOISY_TRAINING",
             "FORCE_RETRAIN", "FORCE_ABLATIONS", "FORCE_NOISE_EVALUATION", "FORCE_NOISY_TRAINING", "USE_IBM_HARDWARE")
    if settings.get("SAVED_RESULTS_ONLY") is not True or any(settings.get(name) is not False for name in flags):
        raise ValueError("The notebook must explicitly disable all execution and force flags before saved review.")
    run_id = datetime.now(timezone.utc).strftime("execution_%Y%m%dT%H%M%SZ")
    directory = root / "results/reviews" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    def started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Saved review cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    client = NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(root)}},
                            on_cell_start=started, record_timing=True)
    with client.setup_kernel(cwd=str(root), env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"}):
        client.reset_execution_trackers()
        for index, cell in enumerate(notebook.cells):
            client.execute_cell(cell, index)
        probe = nbformat.v4.new_code_cell("""
import sys
assert SAVED_RESULTS_ONLY and not any(execution_flags.values())
assert 'torch' not in sys.modules
assert workflow.protect_saved_review is not None
assert noisy_results['coverage'].completed.all()
assert len(artifacts.load_json(noisy_index)['supervised']) == 100
assert len(artifacts.load_json(noisy_index)['unlearned']) == 130
print('GUARDED_SAVED_REVIEW_PASSED')
""")
        # This audit cell is executed in the same guarded kernel but is stored
        # in the review record rather than added to the research narrative.
        notebook.cells.append(probe)
        try:
            client.execute_cell(probe, len(notebook.cells) - 1)
        finally:
            notebook.cells.pop()
    record = {"saved_results_only": True, "flags": {name: settings[name] for name in flags},
              "python": sys.executable, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "research_training_executed": False, "quantum_predictions_generated": False,
              "active_pytorch_imported": False, "before_notebook_sha256": before,
              "audit_outputs": list(probe.outputs), "cells": len(notebook.cells)}
    (directory / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    notebook.metadata["qml_saved_review"] = {"completed_utc": record["completed_utc"],
        "execution_record": (directory / "execution.json").relative_to(root).as_posix(),
        "saved_results_only": True, "all_noisy_review_cells_executed": True}
    nbformat.write(notebook, target)
    print("Saved review completed:", directory.relative_to(root), flush=True)


if __name__ == "__main__":
    review(Path(__file__).resolve().parents[1])
