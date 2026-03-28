"""Tests for the hyperparameter search module."""

import optuna

from src.training.hyperparameter_search import load_study, save_study

# ---------------------------------------------------------------------------
# Tests — save / load study
# ---------------------------------------------------------------------------


class TestSaveLoadStudy:
    def test_save_and_load(self, tmp_path):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        path = str(tmp_path / "study.pkl")
        save_study(study, path)

        loaded = load_study(path)
        assert len(loaded.trials) == 3
        assert loaded.best_value == study.best_value

    def test_save_creates_file(self, tmp_path):
        from pathlib import Path

        study = optuna.create_study()
        path = str(tmp_path / "study2.pkl")
        save_study(study, path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Tests — objective function structure
# ---------------------------------------------------------------------------


class TestObjectiveStructure:
    """Test that objective function is importable and has expected signature."""

    def test_import(self):
        from src.training.hyperparameter_search import objective

        assert callable(objective)

    def test_run_hyperparameter_search_import(self):
        from src.training.hyperparameter_search import run_hyperparameter_search

        assert callable(run_hyperparameter_search)


# ---------------------------------------------------------------------------
# Tests — plot functions (graceful error handling)
# ---------------------------------------------------------------------------


class TestPlotFunctions:
    def test_plot_optimization_history_no_trials(self):
        """Should handle gracefully when no completed trials."""
        from src.training.hyperparameter_search import plot_optimization_history

        study = optuna.create_study()
        # Should not raise
        plot_optimization_history(study)

    def test_plot_param_importances_no_trials(self):
        from src.training.hyperparameter_search import plot_param_importances

        study = optuna.create_study()
        # Should not raise
        plot_param_importances(study)
