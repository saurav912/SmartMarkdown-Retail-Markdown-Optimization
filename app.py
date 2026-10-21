from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import math

from flask import Flask, render_template, request
import pandas as pd

from src.markdown_decision_engine import (
    load_runtime_artifacts,
    recommend_for_target_week,
)
from clustering_engine import load_cluster_mapping, get_sku_cluster

PROJECT_ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

_ARTIFACTS = None
_ARTIFACTS_ERROR = None

def _get_artifacts():
    global _ARTIFACTS, _ARTIFACTS_ERROR
    if _ARTIFACTS is None and _ARTIFACTS_ERROR is None:
        try:
            _ARTIFACTS = load_runtime_artifacts(PROJECT_ROOT)
            cluster_path = PROJECT_ROOT / "data" / "processed" / "smartmarkdown_sku_cluster_mapping.csv"
            _ARTIFACTS["cluster_mapping"] = load_cluster_mapping(cluster_path)
        except Exception as exc:
            _ARTIFACTS_ERROR = f"{type(exc).__name__}: {exc}"
    return _ARTIFACTS, _ARTIFACTS_ERROR


def _as_optional_float(value: str, field_name: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number.")
    return result


def _fmt_num(value):
    if value is None or pd_is_missing(value):
        return "—"
    return f"{float(value):,.2f}"


def pd_is_missing(value):
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def _fmt_pct(value):
    if pd_is_missing(value):
        return "—"
    return f"{float(value):.2f}%"


def _fmt_money(value):
    if pd_is_missing(value):
        return "—"
    return f"₹{float(value):,.2f}"


@app.route("/")
def index():
    artifacts, startup_error = _get_artifacts()
    sku_count = 0
    record_count = 0
    latest_week = "—"
    cluster_stats = []
    cluster_count = 0
    if artifacts:
        raw_df = artifacts.get("raw_df")
        if raw_df is not None and not raw_df.empty:
            sku_count = int(raw_df["SKU_ID"].dropna().astype(str).nunique())
            record_count = int(len(raw_df))
            latest = raw_df["Week_Start"].max()
            if not pd_is_missing(latest):
                latest_week = latest.strftime("%d %b %Y")
        try:
            mapping_path = PROJECT_ROOT / "data" / "processed" / "smartmarkdown_sku_cluster_mapping.csv"
            mapping_df = pd.read_csv(mapping_path)
            # Use the actual saved mapping rather than hardcoding cluster totals.
            cluster_col = next((c for c in mapping_df.columns if c.lower() == "cluster"), None)
            label_col = next((c for c in mapping_df.columns if c.lower() in {"cluster_label", "cluster_name", "cluster_description"}), None)
            sku_col = next((c for c in mapping_df.columns if c.lower() == "sku_id"), None)
            if cluster_col:
                grouped = mapping_df.groupby(cluster_col, dropna=False)
                total = int(mapping_df[sku_col].nunique()) if sku_col else int(len(mapping_df))
                for cluster_id, group in grouped:
                    label = str(group[label_col].dropna().iloc[0]) if label_col and not group[label_col].dropna().empty else f"Cluster {cluster_id}"
                    count = int(group[sku_col].nunique()) if sku_col else int(len(group))
                    cluster_stats.append({"label": label, "count": count, "share": round(count * 100 / total, 1) if total else 0})
                cluster_stats.sort(key=lambda item: item["count"], reverse=True)
                cluster_count = len(cluster_stats)
        except Exception:
            cluster_stats = []
            cluster_count = 0
    return render_template("index.html", startup_error=startup_error, sku_count=sku_count,
                           record_count=record_count, latest_week=latest_week,
                           cluster_stats=cluster_stats, cluster_count=cluster_count)


@app.route("/clusters")
def clusters():
    artifacts, startup_error = _get_artifacts()
    cluster_stats = []
    sku_rows = []
    selected_cluster = (request.args.get("cluster") or "all").strip()
    search = (request.args.get("q") or "").strip().lower()

    try:
        mapping_path = PROJECT_ROOT / "data" / "processed" / "smartmarkdown_sku_cluster_mapping.csv"
        mapping_df = pd.read_csv(mapping_path)
        mapping_df.columns = [str(c).strip() for c in mapping_df.columns]
        # Expected mapping schema: SKU_ID, Cluster, Cluster_Label.
        sku_col = next((c for c in mapping_df.columns if c.lower() == "sku_id"), None)
        cluster_col = next((c for c in mapping_df.columns if c.lower() == "cluster"), None)
        label_col = next((c for c in mapping_df.columns if c.lower() in {"cluster_label", "cluster_name", "cluster_description"}), None)
        if not sku_col or not cluster_col:
            raise ValueError("The cluster mapping must contain SKU_ID and Cluster columns.")

        raw = artifacts.get("raw_df") if artifacts else None
        aggregate_by_sku = {}
        if raw is not None and not raw.empty and "SKU_ID" in raw.columns:
            raw = raw.copy()
            raw["SKU_ID"] = raw["SKU_ID"].astype(str)
            agg_spec = {}
            if "Units_Sold" in raw.columns: agg_spec["Units_Sold"] = "mean"
            if "Discount_Pct" in raw.columns: agg_spec["Discount_Pct"] = "mean"
            if "Closing_Inventory" in raw.columns: agg_spec["Closing_Inventory"] = "last"
            if "Stockout_Flag" in raw.columns: agg_spec["Stockout_Flag"] = "mean"
            if "Category" in raw.columns: agg_spec["Category"] = "last"
            if agg_spec:
                aggregated = raw.sort_values("Week_Start").groupby("SKU_ID", dropna=False).agg(agg_spec).reset_index()
                aggregate_by_sku = aggregated.set_index("SKU_ID").to_dict(orient="index")

        mapping_df[cluster_col] = mapping_df[cluster_col].astype(str)
        total_skus = int(mapping_df[sku_col].nunique())
        for cid, group in mapping_df.groupby(cluster_col, sort=True, dropna=False):
            label = str(group[label_col].dropna().iloc[0]) if label_col and not group[label_col].dropna().empty else f"Cluster {cid}"
            ids = group[sku_col].dropna().astype(str).unique().tolist()
            rows = [aggregate_by_sku.get(sku, {}) for sku in ids]
            mean_units = [float(r["Units_Sold"]) for r in rows if r.get("Units_Sold") is not None and pd.notna(r.get("Units_Sold"))]
            mean_discount = [float(r["Discount_Pct"]) for r in rows if r.get("Discount_Pct") is not None and pd.notna(r.get("Discount_Pct"))]
            cluster_stats.append({
                "id": cid, "label": label, "count": len(ids),
                "share": round(len(ids) * 100 / total_skus, 1) if total_skus else 0,
                "avg_units": round(sum(mean_units) / len(mean_units), 1) if mean_units else None,
                "avg_discount": round(sum(mean_discount) / len(mean_discount), 1) if mean_discount else None,
                "tone": {"0":"blue", "1":"purple", "2":"green"}.get(cid, "blue"),
                "sku_ids": ids,
            })

        for _, item in mapping_df.iterrows():
            sku = str(item[sku_col])
            cid = str(item[cluster_col])
            if selected_cluster != "all" and cid != selected_cluster:
                continue
            if search and search not in sku.lower():
                continue
            metrics = aggregate_by_sku.get(sku, {})
            cluster_label = str(item[label_col]) if label_col and pd.notna(item[label_col]) else f"Cluster {cid}"
            sku_rows.append({
                "sku": sku, "cluster": cid, "label": cluster_label,
                "category": metrics.get("Category", "—") or "—",
                "avg_units": round(float(metrics["Units_Sold"]), 1) if metrics.get("Units_Sold") is not None and pd.notna(metrics.get("Units_Sold")) else None,
                "avg_discount": round(float(metrics["Discount_Pct"]), 1) if metrics.get("Discount_Pct") is not None and pd.notna(metrics.get("Discount_Pct")) else None,
                "inventory": round(float(metrics["Closing_Inventory"]), 1) if metrics.get("Closing_Inventory") is not None and pd.notna(metrics.get("Closing_Inventory")) else None,
                "stockout_pct": round(float(metrics["Stockout_Flag"]) * 100, 1) if metrics.get("Stockout_Flag") is not None and pd.notna(metrics.get("Stockout_Flag")) else None,
                "tone": {"0":"blue", "1":"purple", "2":"green"}.get(cid, "blue"),
            })
        sku_rows.sort(key=lambda x: x["sku"])
    except Exception as exc:
        if not startup_error:
            startup_error = f"Cluster analysis could not be prepared: {type(exc).__name__}: {exc}"

    return render_template("clusters.html", startup_error=startup_error,
                           cluster_stats=cluster_stats, sku_rows=sku_rows,
                           selected_cluster=selected_cluster, search=search,
                           total_skus=len({r.get("sku") for r in sku_rows}) if selected_cluster != "all" or search else sum(c["count"] for c in cluster_stats))


@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    artifacts, startup_error = _get_artifacts()
    form = {
        "sku_id": "",
        "target_week": "",
        "current_discount": "",
        "closing_inventory": "",
        "competitor_price": "",
        "event_override": "",
        "festival_override": False,
        "season_override": "",
    }
    recommendation = None
    scenarios = []
    cluster_info = None
    error = None
    skus = []

    if artifacts:
        skus = sorted(artifacts["raw_df"]["SKU_ID"].dropna().astype(str).unique().tolist())
        if skus:
            form["sku_id"] = skus[0]

        # The latest valid decision date determines the default next target week.
        latest_date = artifacts["raw_df"]["Week_Start"].max()
        if not pd_is_missing(latest_date):
            form["target_week"] = (latest_date + timedelta(days=7)).date().isoformat()

    if request.method == "GET" and request.args.get("sku_id"):
        requested_sku = request.args.get("sku_id", "").strip()
        if requested_sku in skus:
            form["sku_id"] = requested_sku

    if request.method == "POST":
        form.update({
            "sku_id": (request.form.get("sku_id") or "").strip(),
            "target_week": (request.form.get("target_week") or "").strip(),
            "current_discount": (request.form.get("current_discount") or "").strip(),
            "closing_inventory": (request.form.get("closing_inventory") or "").strip(),
            "competitor_price": (request.form.get("competitor_price") or "").strip(),
            "event_override": (request.form.get("event_override") or "").strip(),
            "festival_override": request.form.get("festival_override") == "yes",
            "season_override": (request.form.get("season_override") or "").strip(),
        })

        if not artifacts:
            error = "Project artifacts could not be loaded. Check the file paths and terminal output."
        else:
            try:
                if form["sku_id"] not in skus:
                    raise ValueError("Please select a valid SKU.")
                if not form["target_week"]:
                    raise ValueError("Target week is required.")

                target = date.fromisoformat(form["target_week"])
                if target.weekday() != 0:
                    raise ValueError("Target week must be a Monday.")

                current_discount = _as_optional_float(form["current_discount"], "Current discount")
                closing_inventory = _as_optional_float(form["closing_inventory"], "Closing inventory")
                competitor_price = _as_optional_float(form["competitor_price"], "Competitor price")

                if current_discount is not None and not 0 <= current_discount <= 100:
                    raise ValueError("Current discount must be between 0 and 100.")
                if closing_inventory is not None and closing_inventory < 0:
                    raise ValueError("Closing inventory cannot be negative.")
                if competitor_price is not None and competitor_price <= 0:
                    raise ValueError("Competitor price must be greater than zero.")

                cluster_info = get_sku_cluster(artifacts["cluster_mapping"], form["sku_id"])
                recommendation, scenario_df = recommend_for_target_week(
                    artifacts,
                    sku_id=form["sku_id"],
                    target_week=target.isoformat(),
                    current_discount=current_discount,
                    closing_inventory=closing_inventory,
                    competitor_price=competitor_price,
                    event_override=form["event_override"] or None,
                    festival_override=True if form["festival_override"] else None,
                    season_override=form["season_override"] or None,
                )
                if scenario_df is not None and not scenario_df.empty:
                    scenarios = scenario_df.to_dict(orient="records")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

    return render_template(
        "recommendations.html",
        skus=skus,
        form=form,
        recommendation=recommendation,
        scenarios=scenarios,
        cluster_info=cluster_info,
        error=error,
        startup_error=startup_error,
        fmt_num=_fmt_num,
        fmt_pct=_fmt_pct,
        fmt_money=_fmt_money,
    )


if __name__ == "__main__":
    print("Starting SmartMarkdown Flask app...")
    print(f"Project root: {PROJECT_ROOT}")
    app.run(debug=True)
