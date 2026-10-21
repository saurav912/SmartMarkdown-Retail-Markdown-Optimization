"""Build the exact 33 input features expected by the SmartMarkdown demand model.

The model predicts Future_Latent_Demand for a target week using the latest
completed week (the decision week). Feature formulas mirror Notebook 03.

Expected raw columns are from Historical_Weekly_Data in
SmartMarkdown_Data_Dev_2024_2025.xlsx.
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd


# Keep this order synchronized with smartmarkdown_demand_model_metadata.joblib.
FEATURE_COLUMNS = [
    "SKU_ID", "Category", "Subcategory", "Brand",
    "Future_Month", "Future_Quarter", "Future_Week_of_Year",
    "Future_Festival_Flag", "Future_Retail_Event", "Future_Season_Tag",
    "Lag_1_Demand", "Lag_2_Demand", "Lag_4_Demand",
    "Rolling_4W_Avg_Demand", "Rolling_8W_Avg_Demand", "Rolling_4W_Std_Demand",
    "Discount_Pct", "Price_Index", "Competitor_Gap_Pct", "Unit_Margin",
    "Closing_Inventory", "Days_In_Inventory", "Sell_Through", "Weeks_of_Supply",
    "Inventory_Pressure", "Lag_1_Views", "Lag_1_Cart_Adds",
    "Rolling_4W_Avg_Views", "Rolling_4W_Avg_Cart_Adds",
    "Units_Sold", "Views", "Cart_Adds", "Stockout_Flag",
]

REQUIRED_RAW_COLUMNS = {
    "Week_Start", "SKU_ID", "Brand", "Category", "Subcategory", "MRP",
    "Unit_Cost", "Selling_Price", "Discount_Pct", "Available_Inventory",
    "Latent_Demand", "Units_Sold", "Closing_Inventory", "Stockout_Flag",
    "Days_In_Inventory", "Retail_Event", "Festival_Flag", "Season_Tag",
    "Competitor_Price", "Views", "Cart_Adds",
}


def _normalise_week(value, field_name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value).normalize()
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid date (YYYY-MM-DD).") from exc
    if pd.isna(result):
        raise ValueError(f"{field_name} must be a valid date (YYYY-MM-DD).")
    return result


def _clean_event(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "No_Event"
    return str(value).strip()


def _target_calendar(
    raw: pd.DataFrame,
    sku_id: str,
    target_week: pd.Timestamp,
    decision_row: pd.Series,
    event_override: Optional[str],
    festival_override: Optional[bool],
    season_override: Optional[str],
) -> dict:
    """Use the target week's recorded calendar when available; otherwise safe defaults/overrides."""
    target_rows = raw.loc[(raw["SKU_ID"] == sku_id) & (raw["Week_Start"] == target_week)]
    if not target_rows.empty:
        target_record = target_rows.iloc[-1]
        event = _clean_event(target_record.get("Retail_Event"))
        festival = int(target_record.get("Festival_Flag", 0) or 0)
        season = str(target_record.get("Season_Tag", decision_row.get("Season_Tag", "All Season")))
    else:
        # Future calendar cannot be inferred reliably from this historical dataset.
        # No_Event / non-festival are explicit defaults; season carries forward only
        # as a fallback and can be overridden by the caller.
        event = "No_Event"
        festival = 0
        season = str(decision_row.get("Season_Tag", "All Season"))

    if event_override is not None and str(event_override).strip():
        event = str(event_override).strip()
    if festival_override is not None:
        festival = int(bool(festival_override))
    if season_override is not None and str(season_override).strip():
        season = str(season_override).strip()

    iso = target_week.isocalendar()
    return {
        "Future_Month": int(target_week.month),
        "Future_Quarter": int((target_week.month - 1) // 3 + 1),
        "Future_Week_of_Year": int(iso.week),
        "Future_Festival_Flag": festival,
        "Future_Retail_Event": event,
        "Future_Season_Tag": season,
    }


def build_feature_row(
    raw_df: pd.DataFrame,
    sku_id: str,
    target_week,
    *,
    current_discount: Optional[float] = None,
    closing_inventory: Optional[float] = None,
    competitor_price: Optional[float] = None,
    event_override: Optional[str] = None,
    festival_override: Optional[bool] = None,
    season_override: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return one-row DataFrame with exactly the model's expected features.

    ``target_week`` is the Monday whose demand is to be forecast. The decision
    row is the completed week exactly seven days earlier. If that decision week
    is not present in the source data, this function raises rather than silently
    using stale history.

    Overrides describe the decision state at the end of the decision week.
    Candidate markdown simulation remains the responsibility of the decision
    engine, not this feature builder.
    """
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise ValueError("raw_df must be a non-empty pandas DataFrame.")
    missing = sorted(REQUIRED_RAW_COLUMNS - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    if sku_id is None or not str(sku_id).strip():
        raise ValueError("sku_id is required.")
    sku_id = str(sku_id).strip()
    target = _normalise_week(target_week, "target_week")
    decision_week = target - pd.Timedelta(days=7)

    raw = raw_df.copy()
    raw["Week_Start"] = pd.to_datetime(raw["Week_Start"], errors="coerce").dt.normalize()
    raw = raw.dropna(subset=["Week_Start"])
    raw = raw.sort_values(["SKU_ID", "Week_Start"]).reset_index(drop=True)

    sku_history = raw.loc[raw["SKU_ID"] == sku_id].copy()
    if sku_history.empty:
        raise ValueError(f"Unknown SKU_ID: {sku_id}")

    decision_rows = sku_history.loc[sku_history["Week_Start"] == decision_week]
    if decision_rows.empty:
        latest = sku_history["Week_Start"].max()
        earliest = sku_history["Week_Start"].min()
        raise ValueError(
            f"No completed history for {sku_id} on {decision_week.date()}. "
            f"Available history: {earliest.date()} to {latest.date()}. "
            "Choose a target week exactly seven days after a recorded decision week."
        )
    decision_row = decision_rows.iloc[-1].copy()

    # Derive lag/rolling features from demand/engagement known before or during
    # the completed decision week, using the exact Notebook 03 shift semantics.
    hist = sku_history.loc[sku_history["Week_Start"] <= decision_week].copy()
    hist = hist.sort_values("Week_Start").reset_index(drop=True)
    if hist.empty or hist.iloc[-1]["Week_Start"] != decision_week:
        raise ValueError("Could not align the selected decision week with SKU history.")
    demand = pd.to_numeric(hist["Latent_Demand"], errors="coerce")
    views = pd.to_numeric(hist["Views"], errors="coerce")
    carts = pd.to_numeric(hist["Cart_Adds"], errors="coerce")
    if len(hist) < 9:
        raise ValueError("At least 9 weekly observations are needed for all lag/rolling features.")

    row = {
        "SKU_ID": sku_id,
        "Category": decision_row["Category"],
        "Subcategory": decision_row["Subcategory"],
        "Brand": decision_row["Brand"],
        "Lag_1_Demand": demand.iloc[-2],
        "Lag_2_Demand": demand.iloc[-3],
        "Lag_4_Demand": demand.iloc[-5],
        "Rolling_4W_Avg_Demand": demand.iloc[-5:-1].mean(),
        "Rolling_8W_Avg_Demand": demand.iloc[-9:-1].mean(),
        "Rolling_4W_Std_Demand": demand.iloc[-5:-1].std(ddof=1),
        "Lag_1_Views": views.iloc[-2],
        "Lag_1_Cart_Adds": carts.iloc[-2],
        "Rolling_4W_Avg_Views": views.iloc[-5:-1].mean(),
        "Rolling_4W_Avg_Cart_Adds": carts.iloc[-5:-1].mean(),
        "Discount_Pct": float(decision_row["Discount_Pct"] if current_discount is None else current_discount),
        "Closing_Inventory": float(decision_row["Closing_Inventory"] if closing_inventory is None else closing_inventory),
        "Days_In_Inventory": float(decision_row["Days_In_Inventory"]),
        "Sell_Through": (
            float(decision_row["Units_Sold"]) / float(decision_row["Available_Inventory"])
            if float(decision_row["Available_Inventory"]) > 0 else 0.0
        ),
        "Units_Sold": float(decision_row["Units_Sold"]),
        "Views": float(decision_row["Views"]),
        "Cart_Adds": float(decision_row["Cart_Adds"]),
        "Stockout_Flag": int(decision_row["Stockout_Flag"]),
    }

    # Pricing features follow Notebook 03. Recalculate selling price from MRP
    # and discount only when discount was explicitly overridden.
    discount = row["Discount_Pct"]
    if not 0 <= discount <= 100:
        raise ValueError("current_discount must be between 0 and 100.")
    selling_price = (
        float(decision_row["MRP"]) * (1 - discount / 100)
        if current_discount is not None else float(decision_row["Selling_Price"])
    )
    comp_price = float(decision_row["Competitor_Price"] if competitor_price is None else competitor_price)
    if comp_price <= 0:
        raise ValueError("competitor_price must be greater than zero.")
    row["Price_Index"] = selling_price / float(decision_row["MRP"])
    row["Competitor_Gap_Pct"] = (selling_price - comp_price) / comp_price * 100
    row["Unit_Margin"] = selling_price - float(decision_row["Unit_Cost"])

    avg4 = float(row["Rolling_4W_Avg_Demand"])
    inventory = row["Closing_Inventory"]
    row["Weeks_of_Supply"] = inventory / avg4 if avg4 > 0 else np.nan
    row["Inventory_Pressure"] = inventory / (avg4 + 1)

    row.update(_target_calendar(
        raw, sku_id, target, decision_row,
        event_override, festival_override, season_override,
    ))

    expected = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    missing_features = [name for name in expected if name not in row]
    extra_features = [name for name in row if name not in expected]
    if missing_features or extra_features:
        raise ValueError(
            f"Feature schema mismatch. Missing={missing_features}; unexpected={extra_features}"
        )
    result = pd.DataFrame([[row[name] for name in expected]], columns=expected)
    if result.columns.tolist() != expected:
        raise AssertionError("Feature order does not match the expected model schema.")
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Generated features contain missing values: {bad}")
    return result


def load_raw_history(path: str, sheet_name: str = "Historical_Weekly_Data") -> pd.DataFrame:
    """Load and validate the source Excel history."""
    data = pd.read_excel(path, sheet_name=sheet_name)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Excel source is missing required columns: {missing}")
    data["Week_Start"] = pd.to_datetime(data["Week_Start"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["Week_Start"])
    return data.sort_values(["SKU_ID", "Week_Start"]).reset_index(drop=True)
