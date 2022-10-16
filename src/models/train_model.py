import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
import logging
from pathlib import Path


def main(input_filepath, output_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('making a ML model from processed data')

    x = pd.read_csv(f"{input_filepath}/x_train_model_input.csv")
    y = pd.read_csv(f"{input_filepath}/y_train_model_input.csv")

    # Model
    #NB_pipeline = make_pipeline(StandardScaler(), GaussianNB())
    #NB_pipeline.fit(x, y.values.ravel())

    #dump(NB_pipeline, f'{output_filepath}/NB_final_model.joblib')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed', f'{project_dir}/models')
