"""JudgeRegistry: the frozen judge is rebuilt exactly, and stale rubrics are refused."""

import pytest

from prompt_opt.judge.calibrator import CalibrationReport
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.judge.registry import (
    JudgeMismatchError,
    JudgeNotFoundError,
    JudgeRegistry,
    rubric_fingerprint,
)
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC, Rubric, RubricDimension


def _passing_report():
    return CalibrationReport(judge_version="auto", rubric_version="v1.0", n_items=10, passed=True)


def test_load_judge_rebuilds_the_frozen_model_and_rubric(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    frozen = reg.freeze("frozen/model-a", DEFAULT_RUBRIC.version, _passing_report())

    judge, version = JudgeRegistry(base_path=str(tmp_path)).load_judge()  # fresh load from disk

    assert isinstance(judge, LLMJudge)
    assert judge.model == "frozen/model-a"  # not settings.judge_model
    assert judge.rubric is DEFAULT_RUBRIC
    assert version.version_id == frozen.version_id == "judge_v1"
    assert version.rubric_fingerprint == rubric_fingerprint(DEFAULT_RUBRIC)


def test_latest_and_specific_versions(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    reg.freeze("model-1", DEFAULT_RUBRIC.version, _passing_report())
    reg.freeze("model-2", DEFAULT_RUBRIC.version, _passing_report())

    assert reg.load_judge()[0].model == "model-2"
    assert reg.load_judge("judge_v1")[0].model == "model-1"


def test_nothing_frozen_or_unknown_version_is_not_found(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    with pytest.raises(JudgeNotFoundError, match="No calibrated judge"):
        reg.load_judge()

    reg.freeze("model-1", DEFAULT_RUBRIC.version, _passing_report())
    with pytest.raises(JudgeNotFoundError, match="judge_v9"):
        reg.load_judge("judge_v9")


def test_a_rubric_edited_after_calibration_is_refused(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    reg.freeze("model-1", DEFAULT_RUBRIC.version, _passing_report())

    # Simulate the rubric definition changing without a version bump
    reg._versions["judge_v1"]["rubric_fingerprint"] = "0123456789abcdef"

    with pytest.raises(JudgeMismatchError, match="re-run `calibrate-judge`"):
        reg.load_judge()


def test_unknown_rubric_version_cannot_be_frozen_or_loaded(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    with pytest.raises(JudgeMismatchError):
        reg.freeze("model-1", "v9.9", _passing_report())


def test_content_hash_covers_the_rubric_definition(tmp_path):
    other = Rubric(
        version=DEFAULT_RUBRIC.version,  # same version string, different content
        dimensions=[RubricDimension(name="only", description="d")],
    )
    reg = JudgeRegistry(base_path=str(tmp_path))
    a = reg.freeze("m", DEFAULT_RUBRIC.version, _passing_report())
    b = reg.freeze("m", DEFAULT_RUBRIC.version, _passing_report(), rubric=other)

    assert a.rubric_fingerprint != b.rubric_fingerprint
    assert a.content_hash != b.content_hash


def test_legacy_entries_without_a_fingerprint_still_load(tmp_path):
    reg = JudgeRegistry(base_path=str(tmp_path))
    reg.freeze("model-1", DEFAULT_RUBRIC.version, _passing_report())
    del reg._versions["judge_v1"]["rubric_fingerprint"]

    judge, version = reg.load_judge()
    assert judge.model == "model-1"
    assert version.rubric_fingerprint == ""
