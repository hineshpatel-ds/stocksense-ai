from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "product_name",
    "category",
    "opening_stock",
    "purchased_quantity",
    "sold_quantity",
    "wasted_quantity",
    "closing_stock",
    "unit_price",
]

NUMERIC_COLUMNS = [
    "opening_stock",
    "purchased_quantity",
    "sold_quantity",
    "wasted_quantity",
    "closing_stock",
    "unit_price",
]


@dataclass
class ValidationResult:
    """
    Stores the result of inventory data validation.

    is_valid:
        True if there are no critical errors.

    data_quality_score:
        Score from 0 to 100 based on validation checks.

    errors:
        Critical problems that must be fixed.

    warnings:
        Non-critical problems that should be reviewed.

    cleaned_data:
        Standardized DataFrame after basic cleaning.
    """

    is_valid: bool
    data_quality_score: int
    errors: List[str]
    warnings: List[str]
    cleaned_data: pd.DataFrame


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names into a consistent format.

    Example:
    'Product Name' becomes 'product_name'
    'Sold Quantity' becomes 'sold_quantity'
    """

    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_inventory_file(file_path: str) -> pd.DataFrame:
    """
    Load inventory data from CSV or Excel file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format. Please upload CSV or Excel file.")


def validate_inventory_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate uploaded inventory data.

    This function checks whether the uploaded inventory data is safe and useful
    for analytics, forecasting, recommendations, and chatbot answers.
    """

    errors = []
    warnings = []
    score = 100

    cleaned_df = standardize_column_names(df)

    # 1. Check required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in cleaned_df.columns]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        score -= min(30, len(missing_columns) * 5)

        return ValidationResult(
            is_valid=False,
            data_quality_score=max(score, 0),
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_df,
        )

    # 2. Check duplicate rows
    duplicate_count = cleaned_df.duplicated().sum()

    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate rows.")
        score -= min(10, duplicate_count)

    # 3. Validate date column
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    invalid_date_count = cleaned_df["date"].isna().sum()

    if invalid_date_count > 0:
        errors.append(f"Found {invalid_date_count} rows with invalid dates.")
        score -= min(15, invalid_date_count * 2)

    # 4. Validate numeric columns
    for col in NUMERIC_COLUMNS:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        invalid_numeric_count = cleaned_df[col].isna().sum()

        if invalid_numeric_count > 0:
            errors.append(f"Column '{col}' has {invalid_numeric_count} invalid numeric values.")
            score -= min(10, invalid_numeric_count * 2)

    # 5. Check missing values in required columns
    missing_value_count = cleaned_df[REQUIRED_COLUMNS].isna().sum().sum()

    if missing_value_count > 0:
        errors.append(f"Found {missing_value_count} missing values in required columns.")
        score -= min(20, missing_value_count)

    # 6. Check negative values
    stock_columns = [
        "opening_stock",
        "purchased_quantity",
        "sold_quantity",
        "wasted_quantity",
        "closing_stock",
    ]

    for col in stock_columns:
        negative_count = (cleaned_df[col] < 0).sum()

        if negative_count > 0:
            errors.append(f"Column '{col}' has {negative_count} negative values.")
            score -= min(15, negative_count * 2)

    # 7. Check empty product names
    empty_product_names = cleaned_df["product_name"].astype(str).str.strip().eq("").sum()

    if empty_product_names > 0:
        errors.append(f"Found {empty_product_names} rows with empty product names.")
        score -= min(10, empty_product_names * 2)

    # 8. Check inventory equation
    expected_closing_stock = (
        cleaned_df["opening_stock"]
        + cleaned_df["purchased_quantity"]
        - cleaned_df["sold_quantity"]
        - cleaned_df["wasted_quantity"]
    )

    equation_mismatch_count = (
        expected_closing_stock.round(2) != cleaned_df["closing_stock"].round(2)
    ).sum()

    if equation_mismatch_count > 0:
        errors.append(
            f"Found {equation_mismatch_count} rows where closing stock does not match inventory equation."
        )
        score -= min(25, equation_mismatch_count)

    # 9. Check impossible sales
    available_stock = cleaned_df["opening_stock"] + cleaned_df["purchased_quantity"]
    impossible_sales_count = (cleaned_df["sold_quantity"] > available_stock).sum()

    if impossible_sales_count > 0:
        errors.append(f"Found {impossible_sales_count} rows where sold quantity is greater than available stock.")
        score -= min(20, impossible_sales_count * 2)

    # 10. Check high wastage rate
    purchased_non_zero = cleaned_df["purchased_quantity"].replace(0, pd.NA)
    waste_rate = cleaned_df["wasted_quantity"] / purchased_non_zero
    high_waste_count = (waste_rate > 0.25).sum()

    if high_waste_count > 0:
        warnings.append(f"Found {high_waste_count} rows with wastage rate above 25%.")
        score -= min(10, high_waste_count)

    # 11. Simple outlier check for sold quantity
    sold_mean = cleaned_df["sold_quantity"].mean()
    sold_std = cleaned_df["sold_quantity"].std()

    if sold_std > 0:
        outlier_count = (cleaned_df["sold_quantity"] > sold_mean + 3 * sold_std).sum()

        if outlier_count > 0:
            warnings.append(f"Found {outlier_count} possible sales outliers.")
            score -= min(10, outlier_count)

    final_score = max(int(score), 0)
    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        data_quality_score=final_score,
        errors=errors,
        warnings=warnings,
        cleaned_data=cleaned_df,
    )


def validate_inventory_file(file_path: str) -> ValidationResult:
    """
    Load and validate inventory data from a file path.
    """

    df = load_inventory_file(file_path)
    return validate_inventory_data(df)


if __name__ == "__main__":
    result = validate_inventory_file("data/sample/sample_inventory.csv")

    print("Validation completed")
    print(f"Is valid: {result.is_valid}")
    print(f"Data quality score: {result.data_quality_score}/100")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"- {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")