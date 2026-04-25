import pandas as pd

from src.data.validation import validate_inventory_data


def test_valid_inventory_data_passes_validation():
    df = pd.DataFrame(
        {
            "date": ["2025-10-01"],
            "store_id": ["S001"],
            "product_id": ["P001"],
            "product_name": ["Veggie Burger"],
            "category": ["Food"],
            "opening_stock": [100],
            "purchased_quantity": [20],
            "sold_quantity": [30],
            "wasted_quantity": [5],
            "closing_stock": [85],
            "unit_price": [5.99],
        }
    )

    result = validate_inventory_data(df)

    assert result.is_valid is True
    assert result.data_quality_score == 100
    assert len(result.errors) == 0


def test_missing_required_column_fails_validation():
    df = pd.DataFrame(
        {
            "date": ["2025-10-01"],
            "store_id": ["S001"],
            "product_id": ["P001"],
            "product_name": ["Veggie Burger"],
        }
    )

    result = validate_inventory_data(df)

    assert result.is_valid is False
    assert len(result.errors) > 0
    assert "Missing required columns" in result.errors[0]


def test_negative_stock_fails_validation():
    df = pd.DataFrame(
        {
            "date": ["2025-10-01"],
            "store_id": ["S001"],
            "product_id": ["P001"],
            "product_name": ["Veggie Burger"],
            "category": ["Food"],
            "opening_stock": [-100],
            "purchased_quantity": [20],
            "sold_quantity": [30],
            "wasted_quantity": [5],
            "closing_stock": [-115],
            "unit_price": [5.99],
        }
    )

    result = validate_inventory_data(df)

    assert result.is_valid is False
    assert any("negative values" in error for error in result.errors)


def test_inventory_equation_mismatch_fails_validation():
    df = pd.DataFrame(
        {
            "date": ["2025-10-01"],
            "store_id": ["S001"],
            "product_id": ["P001"],
            "product_name": ["Veggie Burger"],
            "category": ["Food"],
            "opening_stock": [100],
            "purchased_quantity": [20],
            "sold_quantity": [30],
            "wasted_quantity": [5],
            "closing_stock": [999],
            "unit_price": [5.99],
        }
    )

    result = validate_inventory_data(df)

    assert result.is_valid is False
    assert any("closing stock does not match" in error for error in result.errors)


def test_column_name_standardization():
    df = pd.DataFrame(
        {
            "Date": ["2025-10-01"],
            "Store ID": ["S001"],
            "Product ID": ["P001"],
            "Product Name": ["Veggie Burger"],
            "Category": ["Food"],
            "Opening Stock": [100],
            "Purchased Quantity": [20],
            "Sold Quantity": [30],
            "Wasted Quantity": [5],
            "Closing Stock": [85],
            "Unit Price": [5.99],
        }
    )

    result = validate_inventory_data(df)

    assert result.is_valid is True
    assert "product_name" in result.cleaned_data.columns
    assert "sold_quantity" in result.cleaned_data.columns