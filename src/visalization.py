import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

def plot_correlation(statsbomb_xg, pred_xg):
    plt.figure(figsize=(6,6))
    plt.scatter(statsbomb_xg, pred_xg, alpha=0.6, edgecolor='k')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
    plt.xlabel("StatsBomb xG")
    plt.ylabel("Model xG")
    plt.title("Correlation plot: Model vs StatsBomb xG")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_calibration(y_test, pred_xg, statsbomb_xg):
    prob_true_model, prob_pred_model = calibration_curve(y_test, pred_xg, n_bins=10)
    prob_true_sb, prob_pred_sb = calibration_curve(y_test, statsbomb_xg, n_bins=10)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred_model, prob_true_model, marker='o', label='Model', color='blue')
    plt.plot(prob_pred_sb, prob_true_sb, marker='s', label='StatsBomb xG', color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed goal frequency')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weekly_xg(week_xg):
    plt.figure(figsize=(10, 6))
    plt.plot(week_xg['week'], week_xg['predicted_xg'], marker='o', label='Model xG', color='blue')
    plt.plot(week_xg['week'], week_xg['statsbomb_xg'], marker='s', label='StatsBomb xG', color='orange')
    plt.plot(week_xg['week'], week_xg['goal/no goal'], marker='*', label='Actual goals', color='green')
    plt.xlabel('Match Week')
    plt.ylabel('Total xG (per match)')
    plt.title('Barcelona xG per Match Week (2019/2020)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, pred_xg, statsbomb_xg):
    fpr_model, tpr_model, _ = roc_curve(y_test, pred_xg)
    roc_auc_model = auc(fpr_model, tpr_model)
    fpr_sb, tpr_sb, _ = roc_curve(y_test, statsbomb_xg)
    roc_auc_sb = auc(fpr_sb, tpr_sb)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_model, tpr_model, color="blue", lw=2, label=f"Model (AUC = {roc_auc_model:.3f})")
    plt.plot(fpr_sb, tpr_sb, color="orange", lw=2, label=f"StatsBomb xG (AUC = {roc_auc_sb:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: Model vs StatsBomb xG")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_feature_importances(importances_df, model_name):
    plt.figure(figsize=(8, 5))
    plt.barh(
        importances_df["Feature"],
        importances_df["Mean"],
        xerr=importances_df["Std"],
        color="skyblue",
        edgecolor="black"
    )
    plt.xlabel("Mean Decrease in ROC AUC")
    plt.title(
        f"Permutation Feature Importance - {model_name}"
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    for i, row in importances_df.iterrows():
        mean = row["Mean"]
        std = row["Std"]
        plt.text(
            mean + std + 0.001,
            list(importances_df.index).index(i),
            f"{mean:.3f} ± {std:.3f}",
            va='center',
            fontsize=9
        )

    plt.show()

def plot_l1_paths(C_values, coefs, feature_names):
    plt.figure(figsize=(10, 6))

    for i, feature in enumerate(feature_names):
        plt.plot(C_values, coefs[:, i], label=feature)

    plt.xscale("log")
    plt.xlabel("C (inverse regularization strength)")
    plt.ylabel("Coefficient value")
    plt.title("L1 Logistic Regression — Coefficient Path")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()