from solved_time_regression import XGBRegression

xgb_regression = XGBRegression(dummy=True, adjusted=False)
xgb_regression.sweep()