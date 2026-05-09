from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, make_scorer

def evaluate_model(model, param_grid, X, y):

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "log_loss": "neg_log_loss",
        "brier": make_scorer(brier_score_loss, needs_proba=True)
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_index = grid.best_index_

    results = {
        "best_params": grid.best_params_,
        "roc_auc": grid.cv_results_["mean_test_roc_auc"][best_index],
        "accuracy": grid.cv_results_["mean_test_accuracy"][best_index],
        "log_loss": -grid.cv_results_["mean_test_log_loss"][best_index],
        "brier_score": grid.cv_results_["mean_test_brier"][best_index]
    }

    return results

def print_results(results):

    print("Best parameters:", results["best_params"])
    print(f"Best ROC AUC: {results['roc_auc']}")
    print(f"Best Accuracy: {results['accuracy']}")
    print(f"Best Log Loss: {results['log_loss']}")
    print(f"Best Brier Score: {results['brier_score']}")