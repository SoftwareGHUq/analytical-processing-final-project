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
    data_scenario = drop_gargbage_columns(data)
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


def test_clean_categoric_columns(data):
    # Given
    style_unique_categories_before = len(data['Style'].unique())
    brew_method_categories_before = len(data['BrewMethod'].unique())
    sugar_scale_categories_before = len(data['SugarScale'].unique())

    # When
    data_scenario = clean_categoric_columns(data)

    # Then
    assert style_unique_categories_before != len(
        data_scenario['Style'].unique())
    assert brew_method_categories_before != len(
        data_scenario['BrewMethod'].unique())
    assert sugar_scale_categories_before != len(
        data_scenario['SugarScale'].unique())


def test_clean_numeric_columns(data):

    # Given
    efficiency_dirty_length = len(data['Efficiency'])
    boil_gravity_dirty_length = len(data['BoilGravity'])
    fg_dirty_length = len(data['FG'])
    primary_temp_dirty_length = len(data['PrimaryTemp'])
    color_dirty_length = len(data['Color'])
    ibu_dirty_length = len(data['IBU'])
    abv_dirty_length = len(data['ABV'])
    boil_time_dirty_length = len(data['BoilTime'])

    data_scenario = drop_gargbage_columns(data)
    # When
    data_scenario = clean_numeric_columns(data)
    # Then
    assert data_scenario['Efficiency'].dtype == 'float'
    assert data_scenario['BoilGravity'].dtype == 'float'
    assert data_scenario['FG'].dtype == 'float'
    assert data_scenario['PrimaryTemp'].dtype == 'float'
    assert data_scenario['Color'].dtype == 'float'
    assert data_scenario['IBU'].dtype == 'float'
    assert data_scenario['ABV'].dtype == 'float'
    assert data_scenario['BoilTime'].dtype == 'float'

    assert efficiency_dirty_length > len(data_scenario['Efficiency'])
    assert boil_gravity_dirty_length > len(data_scenario['BoilGravity'].dtype)
    assert fg_dirty_length > len(data_scenario['FG'].dtype)
    assert primary_temp_dirty_length > len(data_scenario['PrimaryTemp'].dtype)
    assert color_dirty_length > len(data_scenario['Color'].dtype)
    assert ibu_dirty_length > len(data_scenario['IBU'].dtype)
    assert abv_dirty_length > len(data_scenario['ABV'].dtype)
    assert boil_gravity_dirty_length > len(data_scenario['BoilTime'].dtype)


def test_drop_unuseful_columns(data):
    # Given
    unuse_columns = {'PitchRate', 'MashThickness', 'BeerID', 'StyleID',
                     'Name', 'PrimingAmount', 'PrimingMethod', 'UserId', 'Size(L)'}

    # When
    data_scenario = drop_unuseful_columns(data)
    data_scenario_columns = set(data_scenario.columns.values)
    # Then
    assert data_scenario_columns - unuse_columns == data_scenario_columns


def test_drop_garbage_columns(data):
    # Given
    garbage_columns = {'BoilSize', 'ugtft', 'index',
                       'nhbhgv', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 0'}
    # When
    data_scenario = drop_gargbage_columns(data)
    data_scenario_columns = set(data_scenario.columns.values)
    # Then
    assert data_scenario_columns - garbage_columns == data_scenario_columns
