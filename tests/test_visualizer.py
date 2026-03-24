"""Tests for spore.visualizer — GeometricActivationVisualizer."""

from spore.visualizer import GeometricActivationVisualizer, VERBATIM_MIGRATION_BRIEF


def test_migration_brief_is_exact_verbatim_text():
    viz = GeometricActivationVisualizer()
    assert viz.migration_brief() == VERBATIM_MIGRATION_BRIEF
    assert "Replace the existing ball-and-web" in viz.migration_brief()
    assert "unified Streamlit multi-tab dashboard" in viz.migration_brief()


def test_module_plan_has_exactly_ten_modules():
    viz = GeometricActivationVisualizer()
    plan = viz.module_plan()
    assert len(plan) == 10
    assert plan[0].name == "activation_hooks"
    assert plan[-1].name == "attention_logit_lens_view"


def test_readiness_reports_empty_but_ready(tmp_path):
    viz = GeometricActivationVisualizer()
    state = viz.readiness(tmp_path / "artifacts")
    assert state["dashboard_only"] is True
    assert state["empty_but_ready"] is True
    assert state["entrypoint"] == "streamlit_app.py"
