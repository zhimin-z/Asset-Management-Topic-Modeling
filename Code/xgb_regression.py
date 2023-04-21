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
    'max_depth': 10,
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
            'min': 0.0001, 'max': 0.3
        },
    },
}

path_dataset = 'Dataset'
path_result = 'Result'
path_general = os.path.join(path_result, 'General')

count = 500
wandb_project = 'challenge-solved-time-regression-modeling'
wandb.login()


def _train():
    with wandb.init() as run:
        run.config.setdefaults(config_defaults)
        regressor = xgb.XGBRegressor(objective=run.config.objective, max_depth=run.config.max_depth,
                                     n_estimators=wandb.config.n_estimators, eta=wandb.config.eta)
        scores = cross_val_score(regressor, df[X], df[y], cv=run.config.cv)
        wandb.log({'root mean square log error': scores.mean()})


X = ['Challenge_answer_count', 'Challenge_comment_count', 'Challenge_participation_count', 'Challenge_information_entropy', 'Challenge_link_count', 'Challenge_readability', 'Challenge_score', 'Challenge_sentence_count', 'Challenge_unique_word_count',
     'Challenge_view_count', 'Challenge_word_count', 'Solution_comment_count', 'Solution_information_entropy', 'Solution_link_count', 'Solution_readability', 'Solution_score', 'Solution_sentence_count', 'Solution_unique_word_count', 'Solution_word_count']

y = 'Challenge_solved_time'
config_sweep['name'] = 'XGB Regression (original)'
df = pd.read_json(os.path.join(path_general, 'solved_imputed_original.json'))
sweep_id = wandb.sweep(config_sweep, project=wandb_project)
wandb.agent(sweep_id, function=_train, count=count)

y = 'Challenge_adjusted_solved_time'
config_sweep['name'] = 'XGB Regression (adjusted)'
df = pd.read_json(os.path.join(path_general, 'solved_imputed_adjusted.json'))
sweep_id = wandb.sweep(config_sweep, project=wandb_project)
wandb.agent(sweep_id, function=_train, count=count)
