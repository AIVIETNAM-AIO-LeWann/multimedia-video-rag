"""Check shot sampling with the worker's detector, without loading GPU models."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np


def test_sampling_uses_explicit_worker_detector():
    path = Path(__file__).resolve().parents[1] / "notebooks/ingestion/extract-kf-transnetv2.ipynb"
    notebook = json.loads(path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def select_candidates(" in "".join(cell["source"])
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "select_candidates"
    )
    namespace = {"np": np, "CUT_THRESHOLD": 0.5, "SHORT_SHOT_SECONDS": 4.0, "GAP_SECONDS": 2.0}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    predictions = np.zeros(80)
    detector = Mock()
    # At 10 FPS: 0.4-second, 3-second and 4.6-second shots.
    detector.predictions_to_scenes.return_value = [(0, 3), (4, 33), (34, 79)]
    candidates = namespace["select_candidates"](predictions, 10, detector)
    assert candidates == [(1, 0), (8, 1), (18, 1), (29, 1), (39, 2), (59, 2), (79, 2)]
    detector.predictions_to_scenes.assert_called_once()
    assert detector.predictions_to_scenes.call_args.kwargs == {"threshold": 0.5}
