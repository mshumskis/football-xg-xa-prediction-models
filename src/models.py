from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, make_scorer, log_loss, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
import pandas as pd

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

    return grid, results

def print_results(results):

    print("Best parameters:", results["best_params"])
    print(f"Best ROC AUC: {results['roc_auc']}")
    print(f"Best Accuracy: {results['accuracy']}")
    print(f"Best Log Loss: {results['log_loss']}")
    print(f"Best Brier Score: {results['brier_score']}")

def print_test_metrics(y_true, pred_xg, statsbomb_xg):

    roc_auc_pred = roc_auc_score(y_true, pred_xg)
    accuracy_pred = accuracy_score(y_true, (pred_xg > 0.5).astype(int))
    logloss_pred = log_loss(y_true, pred_xg)
    brier_pred = brier_score_loss(y_true, pred_xg)

    roc_auc_sb = roc_auc_score(y_true, statsbomb_xg)
    accuracy_sb = accuracy_score(y_true, (statsbomb_xg > 0.5).astype(int))
    logloss_sb = log_loss(y_true, statsbomb_xg)
    brier_sb = brier_score_loss(y_true, statsbomb_xg)

    print("\n=== Model ===")
    print(f"ROC AUC:     {roc_auc_pred:.3f}")
    print(f"Accuracy:    {accuracy_pred:.3f}")
    print(f"Log Loss:    {logloss_pred:.3f}")
    print(f"Brier Score: {brier_pred:.3f}")

    print("\n=== StatsBomb xG ===")
    print(f"ROC AUC:     {roc_auc_sb:.3f}")
    print(f"Accuracy:    {accuracy_sb:.3f}")
    print(f"Log Loss:    {logloss_sb:.3f}")
    print(f"Brier Score: {brier_sb:.3f}")

def get_permutation_importance(model, X_test, y_test):

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        scoring="roc_auc", random_state=42, n_jobs=-1
    )

    importances_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean": result.importances_mean,
        "Std": result.importances_std
    }).sort_values(by="Mean", ascending=True)

    return importances_df