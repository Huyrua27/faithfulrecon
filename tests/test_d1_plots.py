"""Synthetic numeric and image checks for D1 batch figures."""

from pathlib import Path
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest
from sklearn.metrics import auc, average_precision_score, roc_auc_score

from frecon.eval import plots
from frecon.eval.d1_metrics import all_metrics


def make_plot_run(score, label, *, seed=0, signal="opacity", mi_available=False):
    label = np.array(label, dtype=bool)
    return SimpleNamespace(
        seed=seed,
        scores={signal: np.array(score, dtype=float)},
        labels={
            "f_i": label,
            "m_i": label.copy() if mi_available else np.zeros(len(label), dtype=bool),
            "distance": np.linspace(0, 1, len(label)),
        },
        meta={"labels": {"m_i_available": mi_available}},
    )


@pytest.fixture
def captured_figures(monkeypatch):
    figures = {}

    def capture(fig, path, **kwargs):
        figures[Path(path).name] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    yield figures
    plt.close("all")


def signal_line(fig, signal="opacity"):
    return next(
        line
        for line in fig.axes[0].lines
        if line.get_label().startswith(f"{signal} (n=")
    )


@pytest.mark.parametrize("score", [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
def test_roc_area_matches_perfect_inverted_and_tied_reference(
    score, captured_figures, tmp_path
):
    run = make_plot_run(score, [True, False, True, False])
    plots.plot_roc([run], ["opacity"], tmp_path / "roc.png")

    line = signal_line(captured_figures["roc.png"])
    x, y = line.get_data()
    assert auc(x, y) == pytest.approx(roc_auc_score(run.labels["f_i"], score))
    assert (x[0], y[0]) == (0.0, 0.0)
    assert (x[-1], y[-1]) == (1.0, 1.0)


def test_pr_does_not_turn_an_inverted_ranking_into_a_linear_ramp(
    captured_figures, tmp_path
):
    run = make_plot_run([0.1, 0.9, 0.8, 0.7], [True, False, False, False])
    plots.plot_pr([run], ["opacity"], tmp_path / "pr.png")

    line = signal_line(captured_figures["pr.png"])
    recall, precision = line.get_data()
    assert line.get_drawstyle() == "steps-pre"
    assert np.sum(np.diff(recall) * precision[1:]) == pytest.approx(0.25)


def test_mean_curves_have_mean_seed_auc_and_ap(captured_figures, tmp_path):
    runs = [
        make_plot_run([0.1, 0.9, 0.8, 0.7], [True, False, False, False]),
        make_plot_run([0.8, 0.7, 0.7, 0.1], [True, False, True, False], seed=1),
    ]
    plots.plot_roc(runs, ["opacity"], tmp_path / "roc.png")
    plots.plot_pr(runs, ["opacity"], tmp_path / "pr.png")

    fpr, tpr = signal_line(captured_figures["roc.png"]).get_data()
    recall, precision = signal_line(captured_figures["pr.png"]).get_data()
    expected_auc = np.mean(
        [roc_auc_score(run.labels["f_i"], run.scores["opacity"]) for run in runs]
    )
    expected_ap = np.mean(
        [
            average_precision_score(run.labels["f_i"], run.scores["opacity"])
            for run in runs
        ]
    )
    assert auc(fpr, tpr) == pytest.approx(expected_auc)
    assert np.sum(np.diff(recall) * precision[1:]) == pytest.approx(expected_ap)


def test_pr_baseline_matches_the_seeds_available_for_each_signal(
    captured_figures, tmp_path
):
    runs = [
        make_plot_run([0.1, 0.9, 0.8, 0.7], [True, False, False, False]),
        make_plot_run(
            [0.8, 0.7, 0.7, 0.1], [True, True, True, False], seed=1, signal="gradmag"
        ),
    ]
    plots.plot_pr(runs, ["opacity", "gradmag"], tmp_path / "pr.png")

    baselines = {
        line.get_label(): line.get_ydata()[0]
        for line in captured_figures["pr.png"].axes[0].lines
        if line.get_label().startswith("Random for ")
    }
    assert baselines["Random for opacity (0.25)"] == 0.25
    assert baselines["Random for gradmag (0.75)"] == 0.75


@pytest.mark.parametrize("label_value", [False, True])
def test_undefined_classification_plots_are_explicit_and_warning_free(
    label_value, captured_figures, tmp_path
):
    run = make_plot_run([0.1, 0.2, 0.3, 0.4], [label_value] * 4)
    row = {
        "signal": "opacity",
        "label": "f_i",
        **all_metrics(run.scores["opacity"], run.labels["f_i"]),
    }
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        plots.plot_roc([run], ["opacity"], tmp_path / "roc.png")
        plots.plot_pr([run], ["opacity"], tmp_path / "pr.png")
        plots.plot_enrichment([row], tmp_path / "enrichment.png")

    assert not emitted
    for fig in captured_figures.values():
        assert any("No defined" in text.get_text() for text in fig.axes[0].texts)


def test_mi_curves_skip_unavailable_placeholders(captured_figures, tmp_path):
    unavailable = make_plot_run([0.1, 0.2, 0.3, 0.4], [True, False, True, False])
    available = make_plot_run(
        [0.9, 0.1, 0.8, 0.2], [True, False, True, False], seed=1, mi_available=True
    )
    plots.plot_roc(
        [unavailable, available], ["opacity"], tmp_path / "mi.png", label_name="m_i"
    )

    line = signal_line(captured_figures["mi.png"])
    assert line.get_label() == "opacity (n=1)"
    assert auc(*line.get_data()) == pytest.approx(1.0)
    assert "m_i" in captured_figures["mi.png"].axes[0].get_title()


def test_four_functions_write_readable_png_and_release_figures(tmp_path):
    run = make_plot_run([-0.1, -0.2, -0.2, -0.9], [True, False, True, False])
    row = {
        "signal": "opacity",
        "label": "f_i",
        **all_metrics(run.scores["opacity"], run.labels["f_i"]),
    }
    before = set(plt.get_fignums())

    plots.plot_roc([run], ["opacity"], tmp_path / "roc.png")
    plots.plot_pr([run], ["opacity"], tmp_path / "pr.png")
    plots.plot_enrichment([row], tmp_path / "enrichment.png")
    plots.plot_score_vs_error([run], ["opacity"], tmp_path / "score.png")

    for filename in ("roc.png", "pr.png", "enrichment.png", "score.png"):
        with Image.open(tmp_path / filename) as image:
            assert image.format == "PNG"
            assert image.width > 500 and image.height > 500
            image.verify()
    assert set(plt.get_fignums()) == before
