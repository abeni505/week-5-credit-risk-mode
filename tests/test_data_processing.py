import pytest
import pandas as pd
from datetime import datetime
# This import assumes your project structure allows src to be found.
# If you run into an ImportError, you may need to adjust your PYTHONPATH.
from src.data_processing import extract_time_features

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    data = {
        'TransactionStartTime': [
            '2023-01-15T14:30:00Z',
            '2023-03-20T08:00:00Z',
            '2023-11-01T23:59:59Z'
        ]
    }
    return pd.DataFrame(data)

def test_extract_time_features_columns_exist(sample_dataframe):
    """
    Test 1: Check if the function correctly adds all the new time-based columns.
    """
    processed_df = extract_time_features(sample_dataframe)
    expected_columns = [
        'TransactionHour', 
        'TransactionDay', 
        'TransactionMonth', 
        'TransactionYear', 
        'TransactionDayOfWeek'
    ]
    for col in expected_columns:
        assert col in processed_df.columns, f"Column '{col}' was not created."

def test_extract_time_features_correct_values(sample_dataframe):
    """
    Test 2: Check if the values in the new columns are correct for a sample row.
    """
    processed_df = extract_time_features(sample_dataframe)
    
    # Check values for the first row: 2023-01-15T14:30:00Z (a Sunday)
    assert processed_df.loc[0, 'TransactionHour'] == 14
    assert processed_df.loc[0, 'TransactionDay'] == 15
    assert processed_df.loc[0, 'TransactionMonth'] == 1
    assert processed_df.loc[0, 'TransactionYear'] == 2023
    assert processed_df.loc[0, 'TransactionDayOfWeek'] == 6 # Monday=0, Sunday=6

    # Check values for the second row: 2023-03-20T08:00:00Z (a Monday)
    assert processed_df.loc[1, 'TransactionHour'] == 8
    assert processed_df.loc[1, 'TransactionDayOfWeek'] == 0 # Monday
