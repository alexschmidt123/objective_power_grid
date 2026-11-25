import importlib.util
import json
import math

import numpy as np
import pytest
import torch

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.evaluate import ARCHIVE, PROTOCOL, require_originals, score_cases
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import LocationMoE, METHODS


def test_alternate_bank_rejected(tmp_path):
    lock = json.loads(PROTOCOL.read_text())["evaluation"]
    (tmp_path / "development_validation_bank.npz").write_bytes(b"different bank")
    with pytest.raises(ValueError, match="hash mismatch.*development_validation_bank"):
        require_originals(tmp_path, lock)


@pytest.mark.parametrize("method", METHODS)
def test_full_horizon_adapter(method):
    spec = importlib.util.spec_from_file_location("reference", ARCHIVE / "source/frozen_evaluator.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(101)
    model = LocationMoE(method, 30)
    bench = LocationFinding({"location_finding": {"design_bounds": [-4, 4]}})
    truth, noise, contrasts = [torch.randn(shape, dtype=torch.float64) for shape in ((17, 4), (17, 30), (9, 4))]
    actual = score_cases(model, bench, truth, noise, contrasts, reference.contrast_scores, 4)
    expected = scores(bench, model, truth, contrasts.expand(len(truth), -1, -1), noise).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
    assert np.all(actual <= math.log1p(len(contrasts)) + 1e-12)
