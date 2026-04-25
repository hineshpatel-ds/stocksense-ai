from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_inventory_data(output_path: str = "data/sample/sample_inventory.csv") -> None:
    """
    Generate synthetic inventory data for demo and testing.

    This creates fake inventory records for a fictional business.
    We are not using real company data because real inventory data can be sensitive.
    """

    np.random.seed(42)

    products = [
        {"product_id": "P001", "product_name": "Veggie Burger", "category": "Food", "unit_price": 5.99},
        {"product_id": "P002", "product_name": "Chicken Burger", "category": "Food", "unit_price": 6.99},
        {"product_id": "P003", "product_name": "French Fries", "category": "Food", "unit_price": 3.49},
        {"product_id": "P004", "product_name": "Cola 500ml", "category": "Beverage", "unit_price": 2.49},
        {"product_id": "P005", "product_name": "Orange Juice", "category": "Beverage", "unit_price": 3.99},
        {"product_id": "P006", "product_name": "Running Shoes", "category": "Sports", "unit_price": 89.99},
        {"product_id": "P007", "product_name": "Training T-Shirt", "category": "Sports", "unit_price": 24.99},
        {"product_id": "P008", "product_name": "Protein Bar", "category": "Health", "unit_price": 2.99},
    ]

    stores = ["S001", "S002"]
    dates = pd.date_range(start="2025-10-01", periods=90, freq="D")

    rows = []

    for store_id in stores:
        for product in products:
            current_stock = np.random.randint(80, 180)

            for date in dates:
                purchased_quantity = np.random.choice([0, 0, 0, 20, 40, 60])

                base_demand = np.random.randint(5, 35)

                # Add simple weekend demand increase
                if date.weekday() in [5, 6]:
                    base_demand = int(base_demand * 1.25)

                sold_quantity = min(current_stock + purchased_quantity, base_demand)

                # Food and beverage have slightly higher waste possibility
                if product["category"] in ["Food", "Beverage"]:
                    wasted_quantity = np.random.randint(0, 5)
                else:
                    wasted_quantity = np.random.randint(0, 2)

                available_stock = current_stock + purchased_quantity
                wasted_quantity = min(wasted_quantity, max(available_stock - sold_quantity, 0))

                closing_stock = available_stock - sold_quantity - wasted_quantity

                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store_id": store_id,
                        "product_id": product["product_id"],
                        "product_name": product["product_name"],
                        "category": product["category"],
                        "opening_stock": current_stock,
                        "purchased_quantity": purchased_quantity,
                        "sold_quantity": sold_quantity,
                        "wasted_quantity": wasted_quantity,
                        "closing_stock": closing_stock,
                        "unit_price": product["unit_price"],
                    }
                )

                current_stock = closing_stock

    df = pd.DataFrame(rows)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Sample inventory data generated successfully: {output_file}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    generate_sample_inventory_data()