
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# =============================================================================
# User settings
# =============================================================================

YEAR = 2022

# Input files
FIGARO_U4_PATH = Path(r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data\estat_naio_10_fcp_u4.tsv.gz")
DK_NAIO1_PATH = Path(r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data\2026325144020611236578NAIO1.xlsx")

# Output file on user's PC
OUTPUT_DIR = Path(r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data")
OUTPUT_FILE = OUTPUT_DIR / f"hh_consumption_weights_DK_{YEAR}.xlsx"

# Set to True if you want the script to fail when the output folder does not exist.
REQUIRE_OUTPUT_DIR = False

# =============================================================================
# Economic definitions
# =============================================================================

EA19_2022 = [
    "AT", "BE", "CY", "DE", "EE", "EL", "ES", "FI", "FR",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK",
]

# Keep ROW internally so we aggregate correctly first
ORIGIN_BUCKETS_ALL = ["domestic", "EA", "US", "ROW"]
ORIGIN_BUCKETS_CES = ["domestic", "EA", "US"]

# 10a3 -> 5-group mapping for Denmark
DK_GROUP_MAP = {
    "A": "il_el",
    "B": "il_eh",
    "C": "ih_el",
    "D_E": "il_eh",
    "F": "NT",
    "G_I": "ih_eh",
    "J": "il_el",
    "K": "ih_el",
    "L": "NT",
    "M_N": "ih_eh",
    "O_Q": "NT",
    "R_S": "NT",
}

SECTOR_ORDER = ["A", "B", "C", "D_E", "F", "G_I", "J", "K", "L", "M_N", "O_Q", "R_S"]
GROUP_ORDER = ["ih_eh", "ih_el", "il_eh", "il_el", "NT"]

SECTOR_LABELS = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D_E": "Utilities",
    "F": "Construction",
    "G_I": "Trade and transport etc.",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate (LA+LB)",
    "M_N": "Other business services",
    "O_Q": "Public administration, education and health",
    "R_S": "Arts, entertainment and other services",
}

# FIGARO product -> 10a3 sector mapping
PRODUCT_TO_SECTOR = {
    # A
    "CPA_A01": "A",
    "CPA_A02": "A",
    "CPA_A03": "A",
    # B
    "CPA_B": "B",
    # C
    "CPA_C10-12": "C",
    "CPA_C13-15": "C",
    "CPA_C16": "C",
    "CPA_C17": "C",
    "CPA_C18": "C",
    "CPA_C19": "C",
    "CPA_C20": "C",
    "CPA_C21": "C",
    "CPA_C22": "C",
    "CPA_C23": "C",
    "CPA_C24": "C",
    "CPA_C25": "C",
    "CPA_C26": "C",
    "CPA_C27": "C",
    "CPA_C28": "C",
    "CPA_C29": "C",
    "CPA_C30": "C",
    "CPA_C31_32": "C",
    "CPA_C33": "C",
    # D_E
    "CPA_D35": "D_E",
    "CPA_E36": "D_E",
    "CPA_E37-39": "D_E",
    # F
    "CPA_F": "F",
    # G_I
    "CPA_G45": "G_I",
    "CPA_G46": "G_I",
    "CPA_G47": "G_I",
    "CPA_H49": "G_I",
    "CPA_H50": "G_I",
    "CPA_H51": "G_I",
    "CPA_H52": "G_I",
    "CPA_H53": "G_I",
    "CPA_I": "G_I",
    # J
    "CPA_J58": "J",
    "CPA_J59_60": "J",
    "CPA_J61": "J",
    "CPA_J62_63": "J",
    # K
    "CPA_K64": "K",
    "CPA_K65": "K",
    "CPA_K66": "K",
    # L
    "CPA_L": "L",
    "OP_NRES": "L",
    "OP_RES": "L",
    # M_N
    "CPA_M69_70": "M_N",
    "CPA_M71": "M_N",
    "CPA_M72": "M_N",
    "CPA_M73": "M_N",
    "CPA_M74_75": "M_N",
    "CPA_N77": "M_N",
    "CPA_N78": "M_N",
    "CPA_N79": "M_N",
    "CPA_N80-82": "M_N",
    # O_Q
    "CPA_O84": "O_Q",
    "CPA_P85": "O_Q",
    "CPA_Q86": "O_Q",
    "CPA_Q87_88": "O_Q",
    # R_S
    "CPA_R90-92": "R_S",
    "CPA_R93": "R_S",
    "CPA_S94": "R_S",
    "CPA_S95": "R_S",
    "CPA_S96": "R_S",
    "CPA_T": "R_S",
    "CPA_U": "R_S",
}

# Exclude balancing and tax items from the product-level basket
EXCLUDED_PRODUCTS = {"B2A3G", "D1", "D21X31", "D29X39"}

# =============================================================================
# Helpers
# =============================================================================

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.fillna(0.0)


def _assign_origin_bucket_dk(destinations: Iterable[str], origins: Iterable[str]) -> pd.Series:
    """
    Denmark-only origin classification:
    - domestic: Danish origin to Danish destination
    - EA: other EA19 origins (excluding DK)
    - US: United States
    - ROW: everything else
    """
    dest = pd.Series(destinations, copy=False).astype(str).str.strip().str.upper()
    orig = pd.Series(origins, copy=False).astype(str).str.strip().str.upper()

    bucket = pd.Series(np.full(len(dest), "ROW", dtype=object), index=dest.index)

    # Domestic can appear as DOM or DK when destination is DK
    is_domestic = orig.eq("DOM") | (orig.eq("DK") & dest.eq("DK"))
    bucket[is_domestic] = "domestic"

    # Other euro area origins, excluding DK
    is_ea = orig.isin(EA19_2022) & ~orig.eq("DK")
    bucket[is_ea] = "EA"

    # United States
    bucket[orig.eq("US")] = "US"

    return bucket


def _add_selected_origin_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CES-relevant totals and weights using only domestic + EA + US.
    ROW is kept in the data but excluded from the CES normalization.
    """
    df = df.copy()

    for col in ORIGIN_BUCKETS_ALL:
        if col not in df.columns:
            df[col] = 0.0

    df["total_all_origins"] = df[ORIGIN_BUCKETS_ALL].sum(axis=1)
    df["total_selected_origins"] = df[ORIGIN_BUCKETS_CES].sum(axis=1)

    # Shares within selected origins: these sum to 1 by construction
    df["w_domestic_in_selected"] = _safe_divide(df["domestic"], df["total_selected_origins"])
    df["w_EA_in_selected"] = _safe_divide(df["EA"], df["total_selected_origins"])
    df["w_US_in_selected"] = _safe_divide(df["US"], df["total_selected_origins"])

    return df


def load_figaro_household_flows_dk(figaro_path: Path, year: int, chunksize: int = 400_000) -> pd.DataFrame:
    """
    Load Danish P3_S14 household final consumption from FIGARO u4
    and aggregate by product and origin bucket.
    Returns product-level values in MIO_EUR.
    """
    destination_set = {"DK"}
    candidate_cols = [str(year), f"{year}", f"{year} ", f" {year}"]

    records = []

    for chunk in pd.read_csv(figaro_path, sep="\t", compression="gzip", chunksize=chunksize, dtype=str):
        first_col = chunk.columns[0]
        meta = chunk[first_col].str.split(",", expand=True)
        meta.columns = ["freq", "ind_use", "prd_ava", "c_dest", "unit", "c_orig"]

        year_column = None
        for c in candidate_cols:
            if c in chunk.columns:
                year_column = c
                break
        if year_column is None:
            raise KeyError(f"Could not find year column for {year}. Available columns: {chunk.columns.tolist()}")

        tmp = pd.DataFrame({
            "ind_use": meta["ind_use"].astype(str).str.strip(),
            "product_code": meta["prd_ava"].astype(str).str.strip(),
            "c_dest": meta["c_dest"].astype(str).str.strip().str.upper(),
            "c_orig": meta["c_orig"].astype(str).str.strip().str.upper(),
            "unit": meta["unit"].astype(str).str.strip(),
        })

        mask = (
            tmp["ind_use"].eq("P3_S14")
            & tmp["unit"].eq("MIO_EUR")
            & tmp["c_dest"].isin(destination_set)
        )
        if not mask.any():
            continue

        tmp = tmp.loc[mask].copy()
        tmp["value"] = pd.to_numeric(chunk.loc[mask, year_column], errors="coerce").fillna(0.0)

        if tmp.empty:
            continue

        tmp = tmp.loc[~tmp["product_code"].isin(EXCLUDED_PRODUCTS)].copy()
        tmp = tmp.loc[tmp["product_code"].isin(PRODUCT_TO_SECTOR)].copy()

        if tmp.empty:
            continue

        tmp["origin_bucket"] = _assign_origin_bucket_dk(tmp["c_dest"], tmp["c_orig"])

        records.append(
            tmp.groupby(["product_code", "origin_bucket"], as_index=False)["value"].sum()
        )

    if not records:
        raise ValueError(f"No FIGARO records found for Denmark, year={year}")

    out = pd.concat(records, ignore_index=True)
    out = out.groupby(["product_code", "origin_bucket"], as_index=False)["value"].sum()

    wide = out.pivot(index="product_code", columns="origin_bucket", values="value").fillna(0.0)

    for col in ORIGIN_BUCKETS_ALL:
        if col not in wide.columns:
            wide[col] = 0.0

    wide = wide.reset_index()
    wide["economy"] = "DK"
    wide["sector_10a3"] = wide["product_code"].map(PRODUCT_TO_SECTOR)
    wide["group5"] = wide["sector_10a3"].map(DK_GROUP_MAP)
    wide["sector_label"] = wide["sector_10a3"].map(SECTOR_LABELS)

    wide = _add_selected_origin_weights(wide)

    return wide.sort_values(["sector_10a3", "product_code"]).reset_index(drop=True)


def aggregate_to_sector_and_group(product_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate product-level flows to 10a3 sectors and 5 groups for Denmark."""

    sector = (
        product_df.groupby(["economy", "sector_10a3", "sector_label"], as_index=False)[ORIGIN_BUCKETS_ALL]
        .sum()
        .sort_values(["economy", "sector_10a3"])
        .reset_index(drop=True)
    )

    group_lookup = product_df[["economy", "sector_10a3", "group5"]].drop_duplicates()
    sector = sector.merge(group_lookup, on=["economy", "sector_10a3"], how="left")

    sector = _add_selected_origin_weights(sector)

    group = (
        sector.groupby(["economy", "group5"], as_index=False)[ORIGIN_BUCKETS_ALL]
        .sum()
        .sort_values(["economy", "group5"])
        .reset_index(drop=True)
    )

    group = _add_selected_origin_weights(group)

    # Group shares in the CES consumption basket:
    # denominator is total Danish household consumption from domestic + EA + US only
    economy_totals = (
        group.groupby("economy", as_index=False)["total_selected_origins"]
        .sum()
        .rename(columns={"total_selected_origins": "economy_total_selected"})
    )
    group = group.merge(economy_totals, on="economy", how="left")
    group["w_group_in_total"] = _safe_divide(group["total_selected_origins"], group["economy_total_selected"])

    # Optional: split weights that add up exactly to w_group_in_total
    group["w_group_domestic_in_total"] = group["w_group_in_total"] * group["w_domestic_in_selected"]
    group["w_group_EA_in_total"] = group["w_group_in_total"] * group["w_EA_in_selected"]
    group["w_group_US_in_total"] = group["w_group_in_total"] * group["w_US_in_selected"]

    return sector, group


def parse_dk_naio1(naio_path: Path) -> pd.DataFrame:
    """Parse the uploaded DST NAIO1 extract for Danish household consumption."""
    raw = pd.read_excel(naio_path)

    records = []
    for _, row in raw.iloc[2:].iterrows():
        sector_text = row.iloc[3]
        if pd.isna(sector_text):
            continue

        sector_code = str(sector_text).split(" ", 1)[0].replace("LA", "L").replace("LB", "L")
        if sector_code not in SECTOR_ORDER:
            continue

        records.append({
            "sector_10a3": sector_code,
            "sector_label": SECTOR_LABELS.get(sector_code),
            "dk_hh_domestic_mDKK": pd.to_numeric(row.iloc[4], errors="coerce"),
            "dk_hh_imports_mDKK": pd.to_numeric(row.iloc[5], errors="coerce"),
        })

    out = pd.DataFrame(records)
    out = (
        out.groupby(["sector_10a3", "sector_label"], as_index=False)[["dk_hh_domestic_mDKK", "dk_hh_imports_mDKK"]]
        .sum()
        .sort_values("sector_10a3")
        .reset_index(drop=True)
    )
    out["group5"] = out["sector_10a3"].map(DK_GROUP_MAP)
    out["dk_hh_total_mDKK"] = out["dk_hh_domestic_mDKK"] + out["dk_hh_imports_mDKK"]
    out["w_domestic_in_total"] = _safe_divide(out["dk_hh_domestic_mDKK"], out["dk_hh_total_mDKK"])
    out["w_imports_in_total"] = _safe_divide(out["dk_hh_imports_mDKK"], out["dk_hh_total_mDKK"])

    return out


def make_metadata_sheet() -> pd.DataFrame:
    rows = [
        ["year", YEAR],
        ["figaro_file", str(FIGARO_U4_PATH)],
        ["dk_naio1_file", str(DK_NAIO1_PATH)],
        ["output_file", str(OUTPUT_FILE)],
        ["country", "Denmark only"],
        ["household_use_code", "P3_S14"],
        ["figaro_unit", "MIO_EUR"],
        ["excluded_figaro_items", ", ".join(sorted(EXCLUDED_PRODUCTS))],
        ["l_mapping", "LA and LB collapsed to L"],
        ["origin_rule_domestic", "Origin DOM or DK for destination DK"],
        ["origin_rule_ea", "Origin in EA19 excluding DK"],
        ["origin_rule_us", "Origin US"],
        ["origin_rule_row", "All other origins kept as ROW internally"],
        ["note_1", "All origins are kept during aggregation."],
        ["note_2", "Final CES weights are normalized over domestic + EA + US only."],
        ["note_3", "ROW is retained for transparency/checks but excluded from CES normalization."],
        ["note_4", "w_group_in_total sums to 1 across the five groups."],
        ["note_5", "Within each group, w_domestic_in_selected + w_EA_in_selected + w_US_in_selected = 1."],
        ["note_6", "w_group_domestic_in_total + w_group_EA_in_total + w_group_US_in_total = w_group_in_total."],
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def main() -> None:
    if REQUIRE_OUTPUT_DIR and not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Product-level flows from FIGARO for Denmark only
    dk_product = load_figaro_household_flows_dk(FIGARO_U4_PATH, YEAR)

    # Aggregates
    sector_dk, group_dk = aggregate_to_sector_and_group(dk_product)

    # Danish NAIO1 cross-check
    dk_naio1 = parse_dk_naio1(DK_NAIO1_PATH)

    # T vs NT flags
    dk_product["tradability"] = np.where(dk_product["group5"].eq("NT"), "NT", "T")
    sector_dk["tradability"] = np.where(sector_dk["group5"].eq("NT"), "NT", "T")
    group_dk["tradability"] = np.where(group_dk["group5"].eq("NT"), "NT", "T")

    # Compact output tables
    product_weights = dk_product[[
        "economy", "product_code", "sector_10a3", "sector_label", "group5", "tradability",
        "domestic", "EA", "US", "ROW",
        "total_all_origins", "total_selected_origins",
        "w_domestic_in_selected", "w_EA_in_selected", "w_US_in_selected",
    ]].copy()

    sector_weights = sector_dk[[
        "economy", "sector_10a3", "sector_label", "group5", "tradability",
        "domestic", "EA", "US", "ROW",
        "total_all_origins", "total_selected_origins",
        "w_domestic_in_selected", "w_EA_in_selected", "w_US_in_selected",
    ]].copy()

    group_weights = group_dk[[
        "economy", "group5", "tradability",
        "domestic", "EA", "US", "ROW",
        "total_all_origins", "total_selected_origins",
        "w_domestic_in_selected", "w_EA_in_selected", "w_US_in_selected",
        "w_group_in_total",
        "w_group_domestic_in_total", "w_group_EA_in_total", "w_group_US_in_total",
    ]].copy()

    # Optional ordering
    group_weights["group5"] = pd.Categorical(group_weights["group5"], categories=GROUP_ORDER, ordered=True)
    group_weights = group_weights.sort_values("group5").reset_index(drop=True)
    group_weights["group5"] = group_weights["group5"].astype(str)

    # Write Excel workbook
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        make_metadata_sheet().to_excel(writer, sheet_name="README", index=False)
        dk_product.to_excel(writer, sheet_name="product_detail", index=False)
        product_weights.to_excel(writer, sheet_name="product_weights", index=False)
        sector_dk.to_excel(writer, sheet_name="sector_detail", index=False)
        sector_weights.to_excel(writer, sheet_name="sector_weights", index=False)
        group_dk.to_excel(writer, sheet_name="group_detail", index=False)
        group_weights.to_excel(writer, sheet_name="group_weights", index=False)
        dk_naio1.to_excel(writer, sheet_name="DK_NAIO1_check", index=False)

    print(f"Done. Wrote Excel file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()