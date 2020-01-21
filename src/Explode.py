import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data_splitter2 import get_only_clickout_items

# https://drive.google.com/file/d/1SOoO0vBYXEpE6-1MY0MYNBvCQnQRjp5_/view


def string_to_array(s):
    """Convert pipe separated string to array."""
    import math
    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df):
    df.impressions = df.impressions.apply(string_to_array)
    df.prices = df.prices.apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df.impressions.str.len())
         for col in df.columns.drop('impressions')}
    )
    df_out.loc[:, 'impressions'] = np.concatenate(df.impressions.values)
    df_out.loc[:, 'prices'] = np.concatenate(df.prices.values)

    df_out.loc[:, 'impressions'] = df_out['impressions'].apply(int)
    df_out.loc[:, 'prices'] = df_out['prices'].apply(int)

    return df_out


def apply_features(df):
    df['session_timestamp'] = df['timestamp'] - df.groupby(by='session_id')['timestamp'].transform('first')
    df['clickout_step_rev'] = df.groupby(by='session_id')['step'].transform('last') - df['step']
    df['clickout_max_step'] = df.groupby(by='session_id')['step'].transform('last')
    df['clickout_item_item_last_timestamp'] = df.groupby(by='session_id')['timestamp'] \
        .transform(lambda x: x.diff()) \
        .fillna(value=-1.)
    df['clickout_item_item_last_step'] = df.groupby(by='session_id')['step'] \
        .transform(lambda x: x.diff()) \
        .fillna(value=-1.)
    df = get_only_clickout_items(df)
    return df


if __name__ == '__main__':
    dfTrain = pd.read_csv('../data/train.csv')
    dfTest = pd.read_csv('../data/test.csv')

    # Timestamp features
    dfTrain = apply_features(dfTrain)
    dfTest = apply_features(dfTest)

    # Train explode
    dfTrain = dfTrain.drop(columns=['current_filters'])
    dfTrain.reference = dfTrain.reference.apply(str)
    dfTrainExploded = explode(dfTrain)
    dfTrainExploded.impressions = dfTrainExploded.impressions.apply(int)
    dfTrainExploded.reference = dfTrainExploded.reference.apply(int)
    dfTrainExploded['label'] = 0
    dfTrainExploded.label = dfTrainExploded.label.mask(dfTrainExploded.reference == dfTrainExploded.impressions, 1)
    # dfTrainExploded.label.value_counts()

    # Test explode
    dfTest = dfTest.drop(columns=['current_filters'])
    dfTest.reference = dfTest.reference.apply(str)
    dfTestExploded = explode(dfTest)
    dfTestExploded.impressions = dfTestExploded.impressions.apply(int)

    # Store dataframes
    dfTestExploded.to_csv('../data/test_exploded.csv', index=False)
    dfTrainExploded.to_csv('../data/train_exploded.csv', index=False)

