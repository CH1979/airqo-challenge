from pathlib import Path

MAIN_PATH = Path('./')
TRAIN = MAIN_PATH / 'data' / 'train.csv'
TEST = MAIN_PATH / 'data' / 'test.csv'
BASELINE_SUBMISIION = MAIN_PATH / 'data' / 'submission.csv'

N_FOLDS = 4
RANDOM_STATE = 1

TARGET = 'ref_pm2_5'
FEATURES = [
    'year',
    'month',
    'day',
    'hour',
    # 'minute',
    'weekday',
    'pm2_5',
    'pm10',
    's2_pm2_5',
    's2_pm10',
    'humidity',
    'temp',
    # 'lat',
    # 'long',
    # 'altitude',
    # 'greenness',
    # 'landform_90m',
    # 'landform_270m',
    'population',
    'dist_major_road'
]

LGBM_PARAMS = {
    'random_state': RANDOM_STATE,
    'num_leaves': 15,
    'max_depth': -1,
    'learning_rate': .01,
    'metric': 'rmse',
    'n_estimators': 5000
}
