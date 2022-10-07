import pandas as pd
import numpy as np


def drop_exact_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to delete all the exact duplicated rows in the dataset
    '''
    return data_frame.drop_duplicates()


def drop_gargbage_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to drop all the garbage columns created by the teacher Jose
    '''
    keys = ['BeerID', 'Name', 'StyleID', 'Size(L)', 'OG', 'FG', 'ABV', 'IBU',
            'Color', 'Boil', 'BoilTime', 'BoilGravity', 'Efficiency', 'MashThickness',
            'SugarScale', 'BrewMethod', 'PitchRate', 'PrimaryTemp', 'PrimingMethod',
            'PrimingAmount', 'UserId', 'Style']

    columns_frame = set(data_frame.T.index)
    keys_set = set(keys)
    columns_not_in_info_set = columns_frame - keys_set
    return data_frame.drop(list(columns_not_in_info_set), axis=1)


def drop_unuseful_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to drop the unuseful columns in the dataset
    '''
    unuseful_columns = ['PitchRate', 'MashThickness', 'BeerID', 'StyleID',
                        'Name', 'PrimingAmount', 'PrimingMethod', 'UserId', 'Size(L)']
    return data_frame.drop(unuseful_columns, axis=1)


def clean_numeric_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to set all the garbage data as nan and then remove it 
    '''

    data_frame.replace('????', np.nan, inplace=True)
    data_frame.replace('jnkhbgvf 98uihj', np.nan, inplace=True)
    data_frame['Efficiency'].replace('fad', np.nan, inplace=True)
    data_frame['Efficiency'].replace('sf', np.nan, inplace=True)
    data_frame['Efficiency'] = data_frame['Efficiency'].astype('float')

    data_frame['OG'].replace('twyey', np.nan, inplace=True)
    data_frame['OG'] = data_frame['OG'].astype('float')

    #data_frame['Size(L)'] = data_frame['Size(L)'].astype('float')

    data_frame['BoilGravity'].replace('afds', np.nan, inplace=True)
    data_frame['BoilGravity'].replace('dafsdgh', np.nan, inplace=True)
    data_frame['BoilGravity'].replace('wyet', np.nan, inplace=True)
    data_frame['BoilGravity'] = data_frame['BoilGravity'].astype('float')

    data_frame['FG'] = data_frame['FG'].astype('float')

    data_frame['PrimaryTemp'] = data_frame['PrimaryTemp'].astype('float')

    data_frame['Color'].replace('tewtye', np.nan, inplace=True)
    data_frame['Color'] = data_frame['Color'].astype('float')

    data_frame['IBU'].replace('wtyeywte', np.nan, inplace=True)
    data_frame['IBU'] = data_frame['IBU'].astype('float')

    data_frame['ABV'].replace('ywet', np.nan, inplace=True)
    data_frame['ABV'] = data_frame['ABV'].astype('float')

    data_frame['BoilTime'].replace('adsf', np.nan, inplace=True)
    data_frame['BoilTime'].replace('ytew', np.nan, inplace=True)
    data_frame['BoilTime'].replace('wyet', np.nan, inplace=True)
    data_frame['BoilTime'].replace('adf', np.nan, inplace=True)
    data_frame['BoilTime'] = data_frame['BoilTime'].astype('float')
    return drop_nan(data_frame)


def clean_categoric_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to clean all the categoric columns
    '''
    data_frame['Style'].replace(
        '/homebrew/recipe/view/249256/black-ipa-an-ode-to-vegard', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/354071/summer-s-w-h-eat-', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/297910/doonkel-', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/435411/dipa-mango-smoothie', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/257731/blond-ale-call-me', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/476147/-', np.nan, inplace=True)
    data_frame['Style'].replace(
        '/homebrew/recipe/view/476147/-', np.nan, inplace=True)
    data_frame['Style'].replace('856842215413', np.nan, inplace=True)
    data_frame['Style'].replace('-354128741541', np.nan, inplace=True)
    data_frame['Style'].replace('4566543', np.nan, inplace=True)
    data_frame['Style'] = data_frame['Style'].astype('category')

    unique_brewmethod_unique_no_weird_values_list = list(
        filter(lambda x: str(x).isdigit(), list(data_frame['BrewMethod'].unique())))
    unique_brewmethod_unique_no_weird_values_list.append('uyrter')
    unique_brewmethod_unique_no_weird_values_list.append('-354128741541')
    data_frame['BrewMethod'].replace(
        unique_brewmethod_unique_no_weird_values_list, np.nan, inplace=True)
    data_frame['BrewMethod'] = data_frame['BrewMethod'].astype('category')

    unique_sugar_scale_set = set(data_frame['SugarScale'].unique())
    unique_sugar_scale_set.add('23542')
    real_sugar_scale_values_set = {'Specific Gravity', 'Plato'}
    weird_sugar_scale_values = list(
        unique_sugar_scale_set - real_sugar_scale_values_set)
    data_frame['SugarScale'].replace(
        weird_sugar_scale_values, np.nan, inplace=True)
    data_frame['SugarScale'] = data_frame['SugarScale'].astype('category')

    return drop_nan(data_frame)


def drop_outliers(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to drop the numeric ouliters, for this dataset we drop the max and the min
    for the numeric columns  
    '''
    types = data_frame.dtypes
    columns = [
        column for column in data_frame.columns.values if types[column] != 'category']
    # Only for numeric columns
    for column in columns:
        set_nan_for_numeric_outliers(column, data_frame)
    return drop_nan(data_frame)


def set_nan_for_numeric_outliers(column_name: str, data_frame: pd.DataFrame) -> None:
    '''
    This function allows to set nan to the values that are equals or higher than the max and the min values 
    get by the description
    '''
    description = data_frame[column_name].describe()
    max = description[7]
    min = description[3]
    data_frame.loc[((data_frame[column_name] <= min) | (
        data_frame[column_name] >= max)), column_name] = np.nan


def drop_nan(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to remove any row that has a nan from the dataset
    '''
    return data_frame.dropna()
