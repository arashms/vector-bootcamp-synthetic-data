"""Small raw/processed Berka fixtures for relational quality checks."""
import pandas as pd
import pytest


@pytest.fixture
def raw_tables():
    """All eight raw tables with shared clients and extra client districts."""
    ids = [1, 2, 3, 4]
    return {
        "account": pd.DataFrame({"account_id": ids, "district_id": [1, 2, 1, 2],
                                 "frequency": ["MONTHLY", "WEEKLY", "MONTHLY", "WEEKLY"],
                                 "date": [930101, 930102, 930103, 930104]}),
        "client": pd.DataFrame({"client_id": [1, 2, 3], "district_id": [3, 2, 1],
                                "birth_number": [700101, 715202, 720303]}),
        "disp": pd.DataFrame({"disp_id": ids, "account_id": ids, "client_id": [1, 2, 3, 1],
                              "type": ["OWNER", "OTHER", "OWNER", "OTHER"]}),
        "card": pd.DataFrame({"card_id": ids, "disp_id": ids, "type": ["CLASSIC", "GOLD", "CLASSIC", "GOLD"],
                              "issued": ["930105 00:00:00", "930106", "930107", "930108"]}),
        "loan": pd.DataFrame({"loan_id": ids, "account_id": ids, "date": [930201, 930202, 930203, 930204],
                              "amount": [100, 200, 300, 400], "duration": [12, 24, 12, 24],
                              "payments": [10, 20, 30, 40], "status": ["A", "B", "A", "B"]}),
        "order": pd.DataFrame({"order_id": ids, "account_id": ids, "bank_to": ["AB", "CD", "AB", "CD"],
                               "account_to": [101, 102, 103, 104], "amount": [10, 20, 30, 40],
                               "k_symbol": ["UROK", "SLUZBY", "UROK", "SLUZBY"]}),
        "trans": pd.DataFrame({"trans_id": list(range(1, 9)), "account_id": [1, 1, 2, 2, 3, 3, 4, 4],
                               "date": [930201, 930202] * 4, "type": ["PRIJEM", "VYDAJ"] * 4,
                               "operation": ["VKLAD", "VYBER"] * 4, "amount": [10, 20] * 4,
                               "balance": [100, -10] * 4, "k_symbol": ["UROK", "SLUZBY"] * 4,
                               "bank": ["AB", "CD"] * 4, "account": [1111, 2222] * 4}),
        "district": pd.DataFrame({"A1": [1, 2, 3], "A2": ["D1", "D2", "D3"], "A3": ["R1", "R2", "R1"],
                                  **{f"A{i}": [10+i, 20+i, 30+i] for i in range(4, 17)}}),
    }


@pytest.fixture
def processed_tables(raw_tables):
    """Use existing transforms to create the metric-test reference."""
    from implementations.tabular_data.multi_table.data_preprocessing import (  # noqa: PLC0415
        pre_process_berka_all_tabels as preprocessing,
    )

    return {name: preprocessing.preprocess_table(name, frame, None)[0] for name, frame in raw_tables.items()}


