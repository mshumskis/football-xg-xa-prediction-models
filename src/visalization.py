import matplotlib.pyplot as plt

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