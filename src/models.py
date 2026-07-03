import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, make_scorer, log_loss, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from scipy.stats import norm

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

def summarize_predictions(y_true, predictions, statsbomb_xg=None):

    total_pred_xg = predictions.sum()
    total_goals = y_true.sum()

    print(f"Total Predicted xG: {total_pred_xg:.2f}")

    if statsbomb_xg is not None:
        total_statsbomb_xg = statsbomb_xg.sum()
        print(f"Total StatsBomb xG: {total_statsbomb_xg:.2f}")

    print(f"Actual Goals: {total_goals}")

def print_test_metrics(y_true, pred_xg, statsbomb_xg=None):

    roc_auc_pred = roc_auc_score(y_true, pred_xg)
    accuracy_pred = accuracy_score(y_true, (pred_xg > 0.5).astype(int))
    logloss_pred = log_loss(y_true, pred_xg)
    brier_pred = brier_score_loss(y_true, pred_xg)

    print("\n=== Model ===")
    print(f"ROC AUC:     {roc_auc_pred:.3f}")
    print(f"Accuracy:    {accuracy_pred:.3f}")
    print(f"Log Loss:    {logloss_pred:.3f}")
    print(f"Brier Score: {brier_pred:.3f}")

    if statsbomb_xg is not None:
        roc_auc_sb = roc_auc_score(y_true, statsbomb_xg)
        accuracy_sb = accuracy_score(y_true, (statsbomb_xg > 0.5).astype(int))
        logloss_sb = log_loss(y_true, statsbomb_xg)
        brier_sb = brier_score_loss(y_true, statsbomb_xg)

        print("\n=== StatsBomb xG ===")
        print(f"ROC AUC:     {roc_auc_sb:.3f}")
        print(f"Accuracy:    {accuracy_sb:.3f}")
        print(f"Log Loss:    {logloss_sb:.3f}")
        print(f"Brier Score: {brier_sb:.3f}")

def print_correlation_stats(predictions, statsbomb_xg):

    correlation = np.corrcoef(statsbomb_xg, predictions)[0, 1]
    mae = np.mean(np.abs(statsbomb_xg - predictions))

    print(f"\nCorrelation (Model xG vs StatsBomb xG): {correlation:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

def aggregate_weekly_xg(df, prediction_column):

    return df.groupby("week").agg({
        prediction_column: "sum",
        "statsbomb_xg": "sum",
        "goal/no goal": "sum"
    }).reset_index()

def get_permutation_importance(model, feature_names, X_test, y_test):

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        scoring="roc_auc", random_state=42, n_jobs=-1
    )

    importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean": result.importances_mean,
        "Std": result.importances_std
    }).sort_values(by="Mean", ascending=True)

    return importances_df

def print_coefficients(feature_names, coefficients):
    print("Coefficients:")
    for feature, weight in zip(feature_names, coefficients):
        print(feature, weight)

def bootstrap_logistic_inference(model, X_train, y_train, feature_names):
    n_boot = 2000
    n = len(y_train)
    boot_coefs = np.zeros((n_boot, len(feature_names) + 1))

    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        X_b = X_train[idx]
        if isinstance(y_train, pd.Series):
            y_b = y_train.iloc[idx].values
        else:
            y_b = y_train[idx]

        boot_model = clone(model)
        boot_model.random_state = i
        boot_model.fit(X_b, y_b)
        boot_coefs[i, 0] = boot_model.intercept_[0]
        boot_coefs[i, 1:] = boot_model.coef_[0]

    coef_original = np.concatenate([[model.intercept_[0]], model.coef_[0]])

    boot_se = boot_coefs.std(axis=0, ddof=1)
    z_values = coef_original / boot_se
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    ci_high = np.percentile(boot_coefs, 97.5, axis=0)

    results = pd.DataFrame({
        "Feature": ["intercept"] + list(feature_names),
        "Coefficient": np.round(coef_original, 6),
        "Std. Error": np.round(boot_se, 6),
        "z": np.round(z_values, 3),
        "P>|z|": np.round(p_values, 4),
        "[0.025": np.round(ci_low, 4),
        "0.975]": np.round(ci_high, 4)
    })

    results[" "] = results["P>|z|"].apply(
        lambda p: "**" if p < 0.01
        else ("*" if p < 0.05 else "")
    )

    return results
