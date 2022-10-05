import pandas as pd
import pytest
from src.data.make_preprocessing import set_nan_for_numeric_outliers


@pytest.fixture
def data():
    data_test = pd.read_csv('data/raw/cervezaDS.csv')
    return data_test


def test_set_nan_for_numeric_outliers(data):
    rows = len(data)
    assert rows > 1
