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
        'name': 'RMSLE',
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
wandb_project = 'challenge-solved-time-regression-modeling'


class XGBRegression:
    def __init__(self, dummy, adjusted):
        gpu_id = 0
        if dummy:
            df = pd.read_json(os.path.join(
                'Result', 'Solution', 'solved_dummy.json'))
            gpu_id += 1
        else:
            df = pd.read_json(os.path.join(
                'Result', 'Solution', 'solved_imputed.json'))
            df = df.drop(['Tool', 'Platform'], axis=1)
        if adjusted:
            df = df[df['Challenge_adjusted_solved_time'].notna()]
            self.y = df['Challenge_adjusted_solved_time']
            gpu_id += 2
        else:
            df = df[df['Challenge_solved_time'].notna()]
            self.y = df['Challenge_solved_time']
        
        config_defaults['gpu_id'] = gpu_id
        self.X = df.drop(['Challenge_solved_time', 'Challenge_adjusted_solved_time',
                         'Challenge_link', 'Challenge_topic_macro', 'Solution_topic_macro'], axis=1)
        config_sweep['name'] = f'XGB Regression: dummy: {dummy}, adjusted: {adjusted}'

    def __train(self):
        with wandb.init() as run:
            run.config.setdefaults(config_defaults)
            regressor = xgb.XGBRegressor(tree_method=run.config.tree_method, objective=run.config.objective, gpu_id=run.config.gpu_id,
                                         max_depth=run.config.max_depth, n_estimators=wandb.config.n_estimators, eta=wandb.config.eta)
            scores = cross_val_score(regressor, self.X, self.y, cv=run.config.cv)
            wandb.log({'RMSLE': scores.mean()})

    def sweep(self):
        wandb.login()
        sweep_id = wandb.sweep(config_sweep, project=wandb_project)
        wandb.agent(sweep_id, function=self.__train, count=count)
