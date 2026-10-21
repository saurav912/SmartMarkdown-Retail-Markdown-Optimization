"""Lookup the precomputed SmartMarkdown SKU cluster assignment.

The clustering notebook fits K-Means at SKU-profile level and exports a stable
SKU-to-cluster mapping. The web app should use that mapping for display rather
than refitting K-Means on each form submission.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"SKU_ID", "Cluster", "Cluster_Label"}


def load_cluster_mapping(path: str | Path) -> pd.DataFrame:
    """Load and validate the precomputed SKU cluster mapping CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SKU cluster mapping not found: {path}")

    mapping = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Cluster mapping is missing required columns: {sorted(missing)}"
        )

    mapping = mapping[["SKU_ID", "Cluster", "Cluster_Label"]].copy()
    mapping["SKU_ID"] = mapping["SKU_ID"].astype(str).str.strip()
    mapping["Cluster"] = pd.to_numeric(mapping["Cluster"], errors="raise").astype(int)
    mapping["Cluster_Label"] = mapping["Cluster_Label"].astype(str).str.strip()

    if mapping["SKU_ID"].duplicated().any():
        duplicates = mapping.loc[mapping["SKU_ID"].duplicated(), "SKU_ID"].tolist()
        raise ValueError(f"Duplicate SKU IDs in cluster mapping: {duplicates}")
    if mapping["Cluster_Label"].eq("").any():
        raise ValueError("Cluster mapping contains empty cluster labels.")
    return mapping.sort_values("SKU_ID").reset_index(drop=True)


def get_sku_cluster(cluster_mapping: pd.DataFrame, sku_id: str) -> dict:
    """Return the saved cluster ID and business label for one SKU."""
    if not isinstance(cluster_mapping, pd.DataFrame):
        raise TypeError("cluster_mapping must be a pandas DataFrame.")
    sku_id = str(sku_id).strip()
    rows = cluster_mapping.loc[cluster_mapping["SKU_ID"].astype(str) == sku_id]
    if rows.empty:
        raise ValueError(f"No saved cluster assignment found for SKU {sku_id}.")
    row = rows.iloc[0]
    return {
        "SKU_ID": sku_id,
        "Cluster": int(row["Cluster"]),
        "Cluster_Label": str(row["Cluster_Label"]),
    }
