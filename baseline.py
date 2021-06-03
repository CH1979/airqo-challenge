import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import settings


def get_data():
    train_df = pd.read_csv(
        settings.TRAIN,
        parse_dates=['created_at']
    )
    test_df = pd.read_csv(
        settings.TEST,
        parse_dates=['created_at']
    )
    sub_df = pd.read_csv(
        settings.BASELINE_SUBMISIION
    )
    sub_df['ref_pm2_5'] = 0
    return train_df, test_df, sub_df


def create_features(df):
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    df['hour'] = df['created_at'].dt.hour
    df['minute'] = df['created_at'].dt.minute
    df['weekday'] = df['created_at'].dt.weekday
    return df


def create_submission(train_df, test_df, sub_df, clf):
    folds = StratifiedKFold(
        n_splits=settings.N_FOLDS
    )
    scores = []

    for i, (trn_idx, val_idx) in enumerate(folds.split(
        train_df[settings.FEATURES],
        train_df['site']
    )):
        print('-' * 30)
        print(f'Fold # {i}')
        clf.fit(
            X=train_df.loc[trn_idx, settings.FEATURES],
            y=train_df.loc[trn_idx, settings.TARGET],
            eval_set=[(
                train_df.loc[val_idx, settings.FEATURES],
                train_df.loc[val_idx, settings.TARGET]
            )],
            eval_metric='rmse',
            verbose=1000
        )
        pred = clf.predict(train_df.loc[val_idx, settings.FEATURES])
        score = np.sqrt(mean_squared_error(
            train_df.loc[val_idx, settings.TARGET],
            pred
        ))
        scores.append(score)
        sub_df['ref_pm2_5'] += clf.predict(
            test_df[settings.FEATURES]
        ) / settings.N_FOLDS
    mean_score = np.mean(scores)
    return sub_df, mean_score


def main():
    train_df, test_df, sub_df = get_data()

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    params = settings.LGBM_PARAMS
    clf = lgb.LGBMRegressor(**params)
    sub_df, mean_score = create_submission(train_df, test_df, sub_df, clf)
    print(f'Mean score on {settings.N_FOLDS}-folds:{mean_score}')

    sub_df.to_csv(
        settings.MAIN_PATH / 'subs' / f'lgb-{mean_score:2.8}.csv',
        index=False
    )


if __name__ == '__main__':
    main()
