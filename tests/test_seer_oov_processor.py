"""Synthetic tests for previously unseen SEER categories."""

import torch
from pyhealth.tasks.seer_survival import (
    SEEROOVMultiHotProcessor,
    SEERSurvivalPrediction,
)


def test_unseen_token_does_not_crash():
    processor = SEEROOVMultiHotProcessor()
    processor.fit(
        [
            {"features": ["tumor_size=050", "sex=Female"]},
            {"features": ["tumor_size=070", "sex=Male"]},
        ],
        "features",
    )
    output = processor.process(["tumor_size=075", "sex=Female"])
    assert isinstance(output, torch.Tensor)
    assert output.shape == (processor.size(),)
    assert output[processor.label_vocab["sex=Female"]].item() == 1.0
    assert "tumor_size=075" not in processor.label_vocab
    assert output.sum().item() == 1.0


def test_known_tokens_unchanged():
    processor = SEEROOVMultiHotProcessor()
    processor.fit([{"features": ["sex=Female", "tumor_size=050"]}], "features")
    output = processor.process(["sex=Female", "tumor_size=050"])
    assert output.sum().item() == 2.0


def test_task_registers_safe_multihot_processor():
    assert SEERSurvivalPrediction.input_schema["features"] is SEEROOVMultiHotProcessor
