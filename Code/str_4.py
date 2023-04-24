from solved_time_regression import XGBRegression

xgb_regression = XGBRegression(dummy=False, adjusted=True)
xgb_regression.sweep()