import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import logging
from pathlib import Path


def main(input_filepath, output_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('making a ML model from processed data')

    data = pd.read_parquet(f"{input_filepath}/beer_data_balanced.parquet")
    X, y = get_x_and_y(data)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X, y.values.ravel())

    dump(random_forest, f'{output_filepath}/random_forest_final_model.joblib')


def get_x_and_y(data: pd.DataFrame):
    label = {'Style'}
    columns_set = set(data.columns.values)
    X = data[list(columns_set-label)]
    y = data[list(label)]
    return X, y


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed', f'{project_dir}/models')
