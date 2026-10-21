"""SmartMarkdown markdown decision engine.

The core recommend_markdown() logic below follows the tested implementation in
Notebook 07. This module does not retrain or modify the saved ML model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd

try:
    from .feature_builder import build_feature_row
except ImportError:
    from feature_builder import build_feature_row


DEFAULT_CANDIDATE_DISCOUNTS = [0, 5, 8, 10, 12, 15, 18, 20, 23, 25, 27, 30]


def adjust_demand_for_discount(
    baseline_demand,
    current_discount,
    candidate_discount,
    relative_response_pct,
):
    """Notebook 07's historical-association-based demand adjustment."""
    discount_change = candidate_discount - current_discount
    proportional_change = discount_change * relative_response_pct / 100
    adjusted_demand = baseline_demand * (1 + proportional_change)
    return max(float(adjusted_demand), 0.0)


def get_adaptive_prediction_interval(predicted_demand, decision_config=None):
    """Use saved adaptive residual bounds, with Notebook 07 defaults as fallback."""
    config = decision_config or {}
    bands = config.get("prediction_intervals", {})
    if bands:
        if predicted_demand <= bands.get("low", {}).get("max_demand", 25):
            band = bands["low"]
        elif predicted_demand <= bands.get("medium", {}).get("max_demand", 50):
            band = bands["medium"]
        else:
            band = bands["high"]
        lower_residual = band["lower_residual"]
        upper_residual = band["upper_residual"]
    elif predicted_demand <= 25:
        lower_residual, upper_residual = -5.53, 9.25
    elif predicted_demand <= 50:
        lower_residual, upper_residual = -12.56, 17.99
    else:
        lower_residual, upper_residual = -19.91, 27.43

    lower_bound = max(float(predicted_demand) + float(lower_residual), 0.0)
    upper_bound = max(float(predicted_demand) + float(upper_residual), 0.0)
    return lower_bound, upper_bound


def recommend_markdown(
    sku_id,
    latest_sku_rows,
    raw_df,
    price_response_df,
    demand_model,
    feature_cols,
    candidate_discounts=None,
    target_sell_through_pct=45,
    max_allowed_discount=30,
    min_margin_pct=35,
    min_profit_retention_pct=80,
    min_conservative_profit_retention_pct=75,
    decision_config=None,
    current_discount_override: Optional[float] = None,
    closing_inventory_override: Optional[float] = None,
):
    """Return (recommendation_dict, scenario_dataframe).

    The function signature is backward-compatible with Notebook 07's tested
    calls. Optional config/override arguments are for application use.
    latest_sku_rows must contain one model-ready row per SKU/week, including
    Week_Start and every feature in feature_cols.
    """
    if candidate_discounts is None:
        candidate_discounts = DEFAULT_CANDIDATE_DISCOUNTS

    sku_match = latest_sku_rows.loc[latest_sku_rows["SKU_ID"].astype(str) == str(sku_id)]
    if sku_match.empty:
        raise ValueError(f"{sku_id} not found in latest SKU data.")
    selected_row = sku_match.iloc[0]

    raw_dates = pd.to_datetime(raw_df["Week_Start"], errors="coerce")
    selected_date = pd.to_datetime(selected_row["Week_Start"], errors="coerce")
    raw_match = raw_df.loc[
        (raw_df["SKU_ID"].astype(str) == str(sku_id))
        & (raw_dates == selected_date)
    ]
    if raw_match.empty:
        raise ValueError(f"Raw record not found for {sku_id} at {selected_date}.")
    selected_raw_row = raw_match.iloc[0]

    response_match = price_response_df.loc[
        price_response_df["SKU_ID"].astype(str) == str(sku_id)
    ]
    if response_match.empty:
        raise ValueError(f"Price-response information not found for {sku_id}.")
    selected_response = response_match.iloc[0]

    missing_features = [c for c in feature_cols if c not in selected_row.index]
    if missing_features:
        raise ValueError(f"Model-ready row is missing features: {missing_features}")
    X_sku = pd.DataFrame([selected_row[list(feature_cols)]], columns=list(feature_cols))
    baseline_demand = max(float(demand_model.predict(X_sku)[0]), 0.0)

    current_discount = float(
        selected_raw_row["Discount_Pct"]
        if current_discount_override is None
        else current_discount_override
    )
    mrp = float(selected_raw_row["MRP"])
    unit_cost = float(selected_raw_row["Unit_Cost"])
    available_inventory = float(
        selected_raw_row["Closing_Inventory"]
        if closing_inventory_override is None
        else closing_inventory_override
    )

    if not 0 <= current_discount <= 100:
        raise ValueError("Current discount must be between 0 and 100.")
    if available_inventory < 0:
        raise ValueError("Closing inventory cannot be negative.")
    if mrp <= 0:
        raise ValueError("MRP must be greater than zero.")
    if unit_cost < 0:
        raise ValueError("Unit cost cannot be negative.")

    if available_inventory <= 0:
        recommendation = {
            "SKU_ID": sku_id,
            "Baseline_Demand": baseline_demand,
            "Current_Discount": current_discount,
            "Recommended_Discount": np.nan,
            "Projected_Sell_Through_Pct": np.nan,
            "Expected_Gross_Profit": 0.0,
            "Expected_Units_Sold": 0.0,
            "Profit_Retention_Pct": np.nan,
            "Margin_Pct": np.nan,
            "Expected_Leftover_Inventory": 0.0,
            "Reason": "No markdown action - no inventory  available.",
        }
        return recommendation, pd.DataFrame()

    relative_response = float(selected_response["Relative_Response_Per_1Pct_Pct"])
    scenario_results = []

    for candidate_discount in candidate_discounts:
        candidate_discount = float(candidate_discount)
        if not 0 <= candidate_discount <= 100:
            raise ValueError(f"Invalid candidate discount: {candidate_discount}")

        selling_price = mrp * (1 - candidate_discount / 100)
        adjusted_demand = adjust_demand_for_discount(
            baseline_demand, current_discount, candidate_discount, relative_response
        )
        lower_demand, upper_demand = get_adaptive_prediction_interval(
            adjusted_demand, decision_config
        )
        expected_units_sold = min(adjusted_demand, available_inventory)
        revenue = expected_units_sold * selling_price
        gross_profit = expected_units_sold * (selling_price - unit_cost)
        margin_pct = (
            (selling_price - unit_cost) / selling_price * 100
            if selling_price > 0 else 0.0
        )
        leftover_inventory = max(available_inventory - expected_units_sold, 0.0)
        conservative_units_sold = min(lower_demand, available_inventory)
        conservative_profit = conservative_units_sold * (selling_price - unit_cost)

        scenario_results.append({
            "Candidate_Discount": candidate_discount,
            "Selling_Price": selling_price,
            "Adjusted_Demand": adjusted_demand,
            "Demand_Lower": lower_demand,
            "Demand_Upper": upper_demand,
            "Expected_Units_Sold": expected_units_sold,
            "Projected_Sell_Through_Pct": (
                expected_units_sold / available_inventory * 100
                if available_inventory > 0 else 0.0
            ),
            "Revenue": revenue,
            "Gross_Profit": gross_profit,
            "Margin_Pct": margin_pct,
            "Leftover_Inventory": leftover_inventory,
            "Conservative_Gross_Profit": conservative_profit,
        })

    scenario_df = pd.DataFrame(scenario_results)

    current_selling_price = mrp * (1 - current_discount / 100)
    current_expected_units = min(baseline_demand, available_inventory)
    current_gross_profit = current_expected_units * (current_selling_price - unit_cost)
    current_lower_demand, _ = get_adaptive_prediction_interval(
        baseline_demand, decision_config
    )
    current_conservative_units = min(current_lower_demand, available_inventory)
    current_conservative_profit = (
        current_conservative_units * (current_selling_price - unit_cost)
    )

    scenario_df["Profit_Retention_Pct"] = (
        scenario_df["Gross_Profit"] / current_gross_profit * 100
        if current_gross_profit > 0 else 0.0
    )
    scenario_df["Conservative_Profit_Retention_Pct"] = (
        scenario_df["Conservative_Gross_Profit"] / current_conservative_profit * 100
        if current_conservative_profit > 0 else 0.0
    )

    scenario_df["Meets_Max_Discount"] = (
        scenario_df["Candidate_Discount"] <= max_allowed_discount
    )
    scenario_df["Meets_Min_Margin"] = scenario_df["Margin_Pct"] >= min_margin_pct
    scenario_df["Meets_Profit_Retention"] = (
        scenario_df["Profit_Retention_Pct"] >= min_profit_retention_pct
    )
    scenario_df["Meets_Conservative_Profit"] = (
        scenario_df["Conservative_Profit_Retention_Pct"]
        >= min_conservative_profit_retention_pct
    )
    scenario_df["Feasible"] = scenario_df[
        [
            "Meets_Max_Discount",
            "Meets_Min_Margin",
            "Meets_Profit_Retention",
            "Meets_Conservative_Profit",
        ]
    ].all(axis=1)

    feasible_scenarios = scenario_df.loc[scenario_df["Feasible"]].copy()
    if feasible_scenarios.empty:
        recommendation = {
            "SKU_ID": sku_id,
            "Baseline_Demand": baseline_demand,
            "Current_Discount": current_discount,
            "Recommended_Discount": np.nan,
            "Projected_Sell_Through_Pct": np.nan,
            "Expected_Gross_Profit": np.nan,
            "Expected_Units_Sold": np.nan,
            "Profit_Retention_Pct": np.nan,
            "Margin_Pct": np.nan,
            "Expected_Leftover_Inventory": np.nan,
            "Reason": "No feasible markdown scenario found.",
        }
        return recommendation, scenario_df

    target_scenarios = feasible_scenarios.loc[
        feasible_scenarios["Projected_Sell_Through_Pct"] >= target_sell_through_pct
    ]
    if not target_scenarios.empty:
        recommended_scenario = target_scenarios.sort_values("Candidate_Discount").iloc[0]
        reason = "Smallest feasible markdown meeting the target sell-through."
    else:
        recommended_scenario = feasible_scenarios.sort_values(
            ["Projected_Sell_Through_Pct", "Candidate_Discount"],
            ascending=[False, True],
        ).iloc[0]
        reason = (
            "No feasible markdown reached the target sell-through; selected the "
            "feasible scenario with the highest projected sell-through."
        )

    recommendation = {
        "SKU_ID": sku_id,
        "Baseline_Demand": baseline_demand,
        "Current_Discount": current_discount,
        "Recommended_Discount": recommended_scenario["Candidate_Discount"],
        "Projected_Sell_Through_Pct": recommended_scenario["Projected_Sell_Through_Pct"],
        "Expected_Gross_Profit": recommended_scenario["Gross_Profit"],
        "Expected_Units_Sold": recommended_scenario["Expected_Units_Sold"],
        "Profit_Retention_Pct": recommended_scenario["Profit_Retention_Pct"],
        "Margin_Pct": recommended_scenario["Margin_Pct"],
        "Expected_Leftover_Inventory": recommended_scenario["Leftover_Inventory"],
        "Reason": reason,
    }
    return recommendation, scenario_df


def load_runtime_artifacts(project_root):
    """Load the actual SmartMarkdown files from the project's root directory."""
    root = Path(project_root).resolve()
    raw_path = root / "data" / "raw" / "SmartMarkdown_Data_Dev_2024_2025.xlsx"
    feature_path = root / "data" / "processed" / "smartmarkdown_feature_engineered.csv"
    response_csv_path = root / "data" / "processed" / "smartmarkdown_price_response.csv"
    model_path = root / "models" / "smartmarkdown_demand_model.joblib"
    metadata_path = root / "models" / "smartmarkdown_demand_model_metadata.joblib"
    config_path = root / "models" / "smartmarkdown_decision_config.joblib"

    required_paths = [
        raw_path, feature_path, response_csv_path, model_path, metadata_path, config_path
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required SmartMarkdown artifact(s) not found:\n- " + "\n- ".join(missing)
        )

    raw_df = pd.read_excel(raw_path)
    raw_df["Week_Start"] = pd.to_datetime(raw_df["Week_Start"], errors="coerce")
    feature_df = pd.read_csv(feature_path, parse_dates=["Week_Start"])
    price_response_df = pd.read_csv(response_csv_path)
    demand_model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    decision_config = joblib.load(config_path)

    feature_cols = metadata.get("features")
    if not feature_cols:
        raise ValueError("Model metadata does not contain a non-empty 'features' list.")
    if len(feature_cols) != 33:
        raise ValueError(f"Expected 33 model features in metadata, got {len(feature_cols)}.")
    if int(metadata.get("feature_count", len(feature_cols))) != len(feature_cols):
        raise ValueError("Metadata feature_count does not match metadata['features'].")

    missing_features = [col for col in feature_cols if col not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Feature-engineered CSV is missing model columns: {missing_features}")

    return {
        "project_root": root,
        "raw_df": raw_df,
        "feature_df": feature_df,
        "price_response_df": price_response_df,
        "demand_model": demand_model,
        "metadata": metadata,
        "feature_cols": list(feature_cols),
        "decision_config": decision_config,
    }


def recommend_for_target_week(
    artifacts,
    sku_id,
    target_week,
    *,
    current_discount=None,
    closing_inventory=None,
    competitor_price=None,
    event_override=None,
    festival_override=None,
    season_override=None,
):
    """Application entry point: build features, then run Notebook 07's decision logic.

    target_week is the Monday being forecast. The feature builder requires a
    recorded decision-week history exactly seven days earlier.
    """
    raw_df = artifacts["raw_df"]
    feature_cols = artifacts["feature_cols"]
    target = pd.Timestamp(target_week).normalize()
    decision_week = target - pd.Timedelta(days=7)

    features = build_feature_row(
        raw_df,
        str(sku_id),
        target,
        current_discount=current_discount,
        closing_inventory=closing_inventory,
        competitor_price=competitor_price,
        event_override=event_override,
        festival_override=festival_override,
        season_override=season_override,
        feature_columns=feature_cols,
    )
    model_row = features.copy()
    model_row["Week_Start"] = decision_week

    config = artifacts["decision_config"]
    recommendation, scenarios = recommend_markdown(
        sku_id=str(sku_id),
        latest_sku_rows=model_row,
        raw_df=raw_df,
        price_response_df=artifacts["price_response_df"],
        demand_model=artifacts["demand_model"],
        feature_cols=feature_cols,
        candidate_discounts=config.get("candidate_discounts", DEFAULT_CANDIDATE_DISCOUNTS),
        target_sell_through_pct=config.get("target_sell_through_pct", 45),
        max_allowed_discount=config.get("max_allowed_discount", 30),
        min_margin_pct=config.get("min_margin_pct", 35),
        min_profit_retention_pct=config.get("min_profit_retention_pct", 80),
        min_conservative_profit_retention_pct=config.get(
            "min_conservative_profit_retention_pct", 75
        ),
        decision_config=config,
        current_discount_override=current_discount,
        closing_inventory_override=closing_inventory,
    )
    recommendation["Target_Week"] = target.date().isoformat()
    return recommendation, scenarios
