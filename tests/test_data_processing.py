import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path to be able to import from src
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from master_dataset_builder import engineer_features_for_chunk

def test_engineer_features_for_chunk():
    # Create a sample DataFrame
    data = {
        'datetime': pd.to_datetime(['2024-01-01 08:00:00', '2024-01-01 16:00:00', '2024-01-06 12:00:00']),
        'route_id': [1, 1, 2],
        'trip_id': [101, 101, 202],
        'stop_sequence': [1, 2, 1]
    }
    df = pd.DataFrame(data)

    # Add a dummy 'boardings' column as it's used in the function
    df['boardings'] = 0

    # Engineer features
    featured_df = engineer_features_for_chunk(df)

    # Assert that the expected columns have been created
    expected_cols = [
        'hour_of_day',
        'day_of_week',
        'day_of_year',
        'month',
        'is_weekend',
        'is_rush_hour',
        'is_first_stop',
        'is_last_stop',
        'stop_sequence_normalized'
    ]

    for col in expected_cols:
        assert col in featured_df.columns

    # Assert that the dtypes are correct
    assert featured_df['hour_of_day'].dtype == 'int8'
    assert featured_df['day_of_week'].dtype == 'int8'
    assert featured_df['day_of_year'].dtype == 'int16'
    assert featured_df['month'].dtype == 'int8'
    assert featured_df['is_weekend'].dtype == 'bool'
    assert featured_df['is_rush_hour'].dtype == 'bool'
    assert featured_df['is_first_stop'].dtype == 'bool'
    assert featured_df['is_last_stop'].dtype == 'bool'
    assert featured_df['stop_sequence_normalized'].dtype == 'float32'
