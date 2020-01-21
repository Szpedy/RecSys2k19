import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRanker
import itertools as it
from sklearn.neural_network import MLPClassifier

PARAMS = {
    "boosting_type": "dart",
    "learning_rate": 0.2,
    "num_leaves": 64,
    "min_child_samples": 5,
    "n_estimators": 5000,
    "drop_rate": 0.015,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "n_jobs": -2,
}


def list_to_string(preds):
    preds = [str(i) for i in preds]
    preds = " ".join(preds)
    return preds


def group_lengths(group_ids):
    return np.array([sum(1 for _ in i) for k, i in it.groupby(group_ids)])


def train_model():
    train = pd.read_csv('data/train_new_features.csv')
    train = train.drop(columns=['city', 'reference', 'action_type', 'hotel_cat'])
    y_train = train[['label']]
    x_train = train.drop(columns=['label'])
    groups = group_lengths(x_train["session_id"].values)
    x_train = x_train.drop(columns=['user_id', 'session_id'])
    ranker = LGBMRanker(PARAMS)
    ranker.fit(x_train,
               y_train.values.ravel(),
               group=groups,
               verbose=1)

    # feature_importances = pd.DataFrame([ranker.feature_importances_], columns=x_train.columns)
    # feature_importances = feature_importances.transpose().reset_index()
    # feature_importances.columns = ['feature', 'importance']
    # feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    # feature_importances.importance /= feature_importances.importance.sum()
    # feature_importances.to_csv('data/features_importances.csv')
    # joblib.dump(ranker, 'data/ranker_new2.pkl')


def train_moodel_nn():
    train = pd.read_csv('data/train_new_features.csv')
    train = train.drop(columns=['city', 'reference', 'action_type', 'hotel_cat'])
    y_train = train['label']
    x_train = train.drop(columns=['label'])
    x_train = x_train.drop(columns=['user_id', 'session_id'])
    x_train.fillna(-1, inplace=True)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(75, 50, 25), max_iter=100, activation='relu', random_state=1)
    clf.fit(x_train, y_train)
    joblib.dump(clf, 'data/NN.joblib')


def get_submission_nn():
    clf = joblib.load('data/NN.joblib')
    test = pd.read_csv('data/test_new_features.csv')
    x_test = test.drop(columns=['action_type', 'city', 'reference', 'hotel_cat'])
    x_test = x_test.drop(columns=['user_id', 'session_id'])
    x_test.fillna(-1, inplace=True)
    prediction = clf.predict(x_test)
    test['prediction'] = pd.Series(prediction)
    recom = test[pd.isna(test.reference)]
    recom = recom[['user_id', 'session_id', 'timestamp', 'step', 'impressions', 'prediction']]
    recom = recom.sort_values(['prediction'], ascending=False)
    recom = recom.drop(columns=['prediction'])
    recom = recom.groupby(['user_id', 'session_id', 'timestamp', 'step'], sort=False)['impressions'] \
            .apply(list) \
            .reset_index(name='item_recommendations')
    recom.item_recommendations = recom.item_recommendations.apply(list_to_string)
    recom.to_csv('data/subNN.csv', index=False)


def get_submission():
    ranker = joblib.load('../data/ranker.pkl')
    test = pd.read_csv('../data/test_new_features.csv')
    x_test = test.drop(columns=['action_type', 'city', 'reference', 'hotel_cat'])
    groups = group_lengths(x_test["session_id"].values)
    x_test = x_test.drop(columns=['user_id', 'session_id'])
    prediction = ranker.predict(x_test, group=groups)
    print(prediction)
    test['prediction'] = pd.Series(prediction)

    recom = test[pd.isna(test.reference)]
    recom = recom[['user_id', 'session_id', 'timestamp', 'step', 'impressions', 'prediction']]
    recom = recom.sort_values(['prediction'], ascending=False)
    recom = recom.drop(columns=['prediction'])
    recom = recom.groupby(['user_id', 'session_id', 'timestamp', 'step'], sort=False)['impressions'] \
        .apply(list) \
        .reset_index(name='item_recommendations')
    recom.item_recommendations = recom.item_recommendations.apply(list_to_string)
    recom.to_csv('../data/subLGBM.csv', index=False)


if __name__ == '__main__':
    # train_moodel_nn()
    get_submission()
    # get_submission_nn()

    # ValueError: DataFrame.dtypes for data must be int, float or bool.
    # Did not expect the data types in the following fields: platform, device
