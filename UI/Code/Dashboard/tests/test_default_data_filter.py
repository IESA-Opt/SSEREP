import pandas as pd

from Code.Dashboard.utils import apply_default_data_filter


def test_apply_default_data_filter_filters_outliers():
    df = pd.DataFrame(
        {
            "variant": ["v1", "v2", "v3", "v4"],
            "CO2_Price": [1999, 2001, 100, 100],
            "totalCosts": [1000, 1000, 70001, 1000],
            "Undispatched": [0.0, 0.0, 0.0, 1.1],
        }
    )

    filtered, removed = apply_default_data_filter(df, enable_filter=True)

    assert removed == 3
    assert list(filtered["variant"]) == ["v1"]


def test_apply_default_data_filter_handles_alt_column_names():
    df = pd.DataFrame(
        {
            "variant": ["ok", "bad"],
            "CO2 Price 2050": [2000, 2000],
            "total system costs": [70000, 70000],
            "Undispatched Electricity (VOLL) - Power NL techUse": [1.0, 2.0],
        }
    )

    filtered, removed = apply_default_data_filter(df, enable_filter=True)

    assert removed == 1
    assert list(filtered["variant"]) == ["ok"]
