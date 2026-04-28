from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file
from src.recommendations.recommendation_engine import (
    generate_recommendations,
    recommendations_to_dataframe,
    summarize_recommendations,
)


def main() -> None:
    file_path = "data/sample/sample_inventory.csv"

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        print("Data validation failed.")
        for error in validation_result.errors:
            print(f"- {error}")
        return

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    recommendations = generate_recommendations(kpi_result["product_performance"])
    recommendation_summary = summarize_recommendations(recommendations)
    recommendation_df = recommendations_to_dataframe(recommendations)

    print("===== Recommendation Summary =====")
    for key, value in recommendation_summary.items():
        print(f"{key}: {value}")

    print("\n===== Top Recommendations =====")
    if recommendation_df.empty:
        print("No recommendations generated.")
    else:
        columns = [
            "priority",
            "product_name",
            "recommendation_type",
            "action",
            "confidence",
        ]
        print(recommendation_df[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()