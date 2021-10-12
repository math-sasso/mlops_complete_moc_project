import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def get_regression_metrics(y_ref,y_pred):
    mse = mean_squared_error(y_ref, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_ref, y_pred)

    metrics = {
        "mse":mse,
        "rmse":rmse,
        "r2":r2,    
    }

    return metrics


def plot_comparing(y_ref,y_pred,outpath:str):
    bins = np.linspace(-10, 10, 100)
    plt.hist(y_ref, bins, alpha=0.5, label='y_ref')
    plt.hist(y_pred, bins, alpha=0.5, label='y_pred')
    plt.legend(loc='upper right')
    plt.savefig(outpath)


    
