
import pandas as pd
from sklearn.utils import resample
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    data_unbalanced = pd.read_parquet(
        f'{input_filepath}/beer_data_clean.parquet')
    data_unbalanced = create_dummies_variables(data_unbalanced)
    data_unbalanced = encode_label(data_unbalanced)

    data_balanced = resample_classes(data_unbalanced)
    data_balanced.to_parquet(
        f'{output_filepath}/beer_data_balanced.parquet')
    logger.info(f'Data saved {len(data_balanced)}')


def encode_label(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to encode the label
    '''
    label_encoder = LabelEncoder()
    data['Style'] = label_encoder.fit_transform(
        data['Style'])
    return data


def create_dummies_variables(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to create the dummies variables for the categoric columns
    '''
    categoric_keys = ['BrewMethod', 'SugarScale', 'Style']
    return pd.get_dummies(data, columns=categoric_keys[:2])


def resample_classes(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to resample the classes with less than 100 elements on it
    '''
    count_df = data.groupby(['Style'])['Style'].count()

    styles_unbalanced = [k for k, v in count_df.items() if v <= 100]

    df_sampled = pd.DataFrame()

    for j in styles_unbalanced:

        df_minority_j = data[data.Style == j]
        if df_minority_j.shape[0] != 0:
            df_minority_upsampled = resample(df_minority_j,
                                             replace=True,
                                             n_samples=400,
                                             stratify=df_minority_j,
                                             random_state=123)
            df_sampled = pd.concat([df_sampled, df_minority_upsampled])
    return pd.concat([data, df_sampled])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    input_url = f'{project_dir}\data\interim'
    output_url = f'{project_dir}\data\processed'
    main(input_url, output_url)
