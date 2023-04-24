from solved_time_regression import XGBRegression

xgb_regression = XGBRegression(dummy=True, adjusted=True)
xgb_regression.sweep()