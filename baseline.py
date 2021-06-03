import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

import settings
import utils


def main():
    train_df, test_df, sub_df = utils.get_data()

    train_df = utils.create_features(train_df)
    test_df = utils.create_features(test_df)

    # params = settings.LGBM_PARAMS
    # clf = lgb.LGBMRegressor(**params)
    # sub_df, mean_score = utils.create_submission(
    #     train_df,
    #     test_df,
    #     sub_df,
    #     clf
    # )
    params = {'C': 0.0001, 'max_iter': 10000}
    # clf = LinearRegression()
    clf = LinearSVR(**params)
    sub_df, mean_score = utils.create_submission(
        train_df,
        test_df,
        sub_df,
        clf
    )
    print(f'Mean score on {settings.N_FOLDS}-folds:{mean_score}')

    # sub_df.to_csv(
    #     settings.MAIN_PATH / 'subs' / f'lgb-{mean_score:2.8}.csv',
    #     index=False
    # )


if __name__ == '__main__':
    main()
