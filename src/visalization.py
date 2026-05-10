import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

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