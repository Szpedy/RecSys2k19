import pandas as pd
import numpy as np
import itertools as it


def split_df_to_train_test(df, train_frac):
    """Divides the data into a training and test set, in order."""
    sessions = df.session_id.unique()
    sess_size = len(sessions)
    sess_size_train = int(sess_size * train_frac)
    sessions = sessions[:sess_size_train]
    train = df[df.session_id.isin(sessions)]
    test = df[~df.session_id.isin(sessions)]
    del df
    print(f'Splitted result: train {len(train)} lines, test {len(test)} lines')
    print(f'Splitted result: train {sess_size_train} sessions, test {sess_size - sess_size_train} sessions')
    return train, test


def get_only_clickout_items(df):
    mask = df["action_type"] == "clickout item"
    df_out = df[mask]

    return df_out


# https://github.com/recsyschallenge/2019/blob/master/src/baseline_algorithm/functions.py#L18
def get_popularity(df):
    df_item_clicks = (
        df
            .groupby("reference")
            .size()
            .reset_index(name="n_clicks")
            .transform(lambda x: x.astype(int))
            .sort_values('n_clicks', ascending=False)
    )
    return df_item_clicks


if __name__ == '__main__':
    input_frac = 1
    train_frac = 0.8
    train_csv = '../recSysData/train.csv'
    df = pd.read_csv(train_csv)
    df = df.tail(int(len(df)*0.4)).head(int(len(df)*0.1))
    print(df)
    train, test = split_df_to_train_test(df, train_frac)
    print(train)
    train.to_csv("../data/train.csv", index=False)

    # bool mask
    bool_last_session = ~test.session_id.duplicated(keep='last')
    bool_last_user_id = ~test.user_id.duplicated(keep='last')
    bool_clickout_item = test.action_type == 'clickout item'
    bool_mask = (bool_last_session) & (bool_last_user_id) & (bool_clickout_item)

    test[['user_id', 'session_id', 'timestamp', 'step', 'reference', 'impressions', 'prices']].to_csv('../data/gt.csv', index=False)
    test.loc[bool_mask, 'reference'] = np.NAN
    test.to_csv('../data/test.csv', index=False)
