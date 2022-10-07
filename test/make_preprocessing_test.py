import pandas as pd
import pytest
# src
from src.data.make_preprocessing import clean_categoric_columns, clean_numeric_columns, drop_gargbage_columns, drop_nan, drop_outliers, drop_unuseful_columns, set_nan_for_numeric_outliers


@pytest.fixture
def data():
    data_test = pd.read_csv('data/raw/cervezaDS.csv')
    return data_test


def test_set_nan_for_numeric_outliers(data):
    # Given
    data_scenario = clean_numeric_columns(data)
    description_before = data_scenario['Efficiency'].describe()
    # When
    set_nan_for_numeric_outliers('Efficiency', data_scenario)
    description_after = data_scenario['Efficiency'].describe()

    # Then
    assert description_before[7] is not description_after[7]
    assert description_before[3] is not description_after[3]


def test_drop_nan(data):
    # Given
    len_before = len(data)
    # When
    data_scenario = drop_nan(data)
    len_after = len(data_scenario)

    # Then
    assert len_before is not len_after


def test_drop_outliers(data):
    # Given
    data_scenario = drop_gargbage_columns(data)
    data_scenario = drop_unuseful_columns(data_scenario)
    data_scenario = clean_numeric_columns(data_scenario)
    data_scenario = clean_categoric_columns(data_scenario)
    # When
    result = drop_outliers(data_scenario)
    # Then
    assert len(data_scenario) > len(result)
