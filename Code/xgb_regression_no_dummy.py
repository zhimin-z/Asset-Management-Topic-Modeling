import os
import wandb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

os.environ["WANDB_API_KEY"] = '9963fa73f81aa361bdbaf545857e1230fc74094c'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "100"

config_defaults = {
    'objective': 'reg:squaredlogerror',
    'tree_method': 'gpu_hist',
    'max_depth': 5,
    'cv': 10,
}

config_sweep = {
    "method": "bayes",
    "metric": {
        'name': 'root mean square log error',
        'goal': 'maximize'
    },
    "parameters": {
        'n_estimators': {
            'values': list(range(500, 1001, 10))
        },
        'eta': {
            'min': 0.01, 'max': 0.3
        },
    },
}

count = 500
wandb_project = 'challenge-solution-regression-modeling'
df = pd.read_json(os.path.join('Result', 'Solution', 'solved_imputed.json'))
wandb.login()


def _train():
    with wandb.init() as run:
        run.config.setdefaults(config_defaults)
        regressor = xgb.XGBRegressor(tree_method = run.config.tree_method, objective=run.config.objective, max_depth=run.config.max_depth, n_estimators=wandb.config.n_estimators, eta=wandb.config.eta)
        scores = cross_val_score(regressor, X, y, cv=run.config.cv)
        wandb.log({'root mean square log error': scores.mean()})


df_original = df[df['Challenge_solved_time'].notna()]
y = df_original['Challenge_solved_time']
df_original.drop(['Challenge_solved_time', 'Challenge_adjusted_solved_time', 'Challenge_link', 'Challenge_topic_macro', 'Solution_topic_macro', 'Tool', 'Platform'], axis=1, inplace=True)
X = df_original
config_sweep['name'] = 'XGB Regression (original)'
sweep_id = wandb.sweep(config_sweep, project=wandb_project)
wandb.agent(sweep_id, function=_train, count=count)

df_adjusted = df[df['Challenge_adjusted_solved_time'].notna()]
y = df_adjusted['Challenge_adjusted_solved_time']
df_adjusted.drop(['Challenge_solved_time', 'Challenge_adjusted_solved_time', 'Challenge_link', 'Challenge_topic_macro', 'Solution_topic_macro', 'Tool', 'Platform'], axis=1, inplace=True)
X = df_adjusted
config_sweep['name'] = 'XGB Regression (adjusted)'
sweep_id = wandb.sweep(config_sweep, project=wandb_project)
wandb.agent(sweep_id, function=_train, count=count)
