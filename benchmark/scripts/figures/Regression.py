# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# Mean absolute Error
def MAE(y_test, y_pred):
    return mean_absolute_error(y_test,y_pred)

# Mean Squared Error
def MSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

# Root Mean Square Error
def RMSE(y_test, y_pred):
    return np.sqrt(MSE(y_test, y_pred))

# Root Mean Squared Error
def RMSSE(y_ture, y_pred, y_test): 
    n = len(y_test)
    numerator = np.mean(np.sum(np.square(y_true - y_pred)))
    denominator = 1/(n-1)*np.sum(np.square((y_test[1:] - y_test[:-1])))
    msse = numerator/denominator
    return np.sqrt(msse)

# Mean Squared Log Error
def MSLE(y_test, y_pred):
    return mean_squared_log_error(y_test, y_pred)

# Root Mean Squared Log Error
def RMSLE(y_test, y_pred):
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

# Mean Percentage Error
def MPE(y_test, y_pred):
	return np.mean((y_test, y_pred) / y_test * 100)

# Mean Absolute Percentage Error
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Symmetric Mean Absolute Percentage Error
def SMAPE(y_test, y_pred):
	return np.mean((np.abs(y_test-y_pred))/(np.abs(y_test)+np.abs(y_pred)))*100
