import pandas as pd
import numpy as np


def remove_duplicates(df):
    df_first_step_for_sessions = df[df.step == 1]
    df_user_sessions = df_first_step_for_sessions[['session_id', 'user_id']]
    unique_sessions = df_user_sessions['session_id'].nunique()
    all_sessions = len(df_user_sessions)
    duplicated_sessions = []
    if unique_sessions != all_sessions:
        df_user_sessions = df_user_sessions.groupby(['session_id', 'user_id']).size().reset_index(name='countCol')
        df_duplicated_sessions = df_user_sessions[df_user_sessions.countCol > 1]
        duplicated_sessions = df_duplicated_sessions['session_id'].to_list()
    return df[~df['session_id'].isin(duplicated_sessions)]


def is_session_id_duplicated(df):
    df_first_step_for_sessions = df[df.step == 1]
    df_user_sessions = df_first_step_for_sessions[['session_id', 'user_id']].compute()
    unique_sessions = df_user_sessions['session_id'].nunique()
    all_sessions = len(df_user_sessions)
    return unique_sessions == all_sessions


def split_data(df, ratio):
    sorted_session_ids = df[df.step == 1].sort_values('timestamp')['session_id']
    sliced = sorted_session_ids.head(int(len(sorted_session_ids) * ratio))
    df_train = df.loc[df['session_id'].isin(sliced)]
    df_gt = df.loc[~df['session_id'].isin(sliced)]
    df_test = nullify_last_click_out(df_gt)
    assert df_gt['step'].iloc[0] == 1
    return df_train, df_gt, df_test


def split_data_by_idx(df, ratio):
    sessions = df.session_id.unique()
    last_train_session_id = sessions[int(ratio * len(sessions))]
    split_index = df[df.session_id == last_train_session_id]['timestamp'].idxmax()
    df_train = df[:split_index + 1]
    df_gt = df[split_index + 1:]
    df_test = nullify_last_click_out(df_gt)
    assert df_gt['step'].iloc[0] == 1
    return df_train, df_test, df_gt


def nullify_last_click_out(df_gt):
    df_test = df_gt.copy()
    indices = df_test[df_test.action_type == 'clickout item'].groupby(['session_id', 'user_id'], as_index=False, sort=False)['timestamp'].idxmax()
    df_test.loc[indices, 'reference'] = np.nan
    return df_test


if __name__ == '__main__':
    df = pd.read_csv("recSysData/train.csv")
    df = remove_duplicates(df)
    df_train, df_gt, df_test = split_data(df, 0.8)
    df_train.to_csv("data/train.csv")
    df_gt.to_csv("data/gt.csv")
    df_test.to_csv("data/test.csv")





