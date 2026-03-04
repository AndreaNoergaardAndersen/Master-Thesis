import re
import os
import numpy as np
import pandas as pd

# -----------------------------
# USER INPUT
# -----------------------------
IN_PATH = r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data\ADB-MRIO-2024-August 2025.xlsx"
SHEET_MRIO = None
SHEET_LEGEND = "Legend"

# Gem output i samme folder som input-filen
out_folder = os.path.dirname(IN_PATH)
OUT_LONG_PATH = os.path.join(out_folder, "adb_mrio_intermediate_imports_long.xlsx")

# -----------------------------
# Helpers
# -----------------------------
EA20 = {
    "Austria","Belgium","Croatia","Cyprus","Estonia","Finland","France","Germany",
    "Greece","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
    "Portugal","Slovakia","Slovenia","Spain"
}

EA_CODE = "EA"
EA_NAME = "Euro area (EA20)"

def detect_mrio_sheet(xls: pd.ExcelFile) -> str:
    for s in xls.sheet_names:
        if s.lower() != "legend":
            return s
    raise ValueError("Kunne ikke finde MRIO-arket (andet end 'Legend').")

def find_header_row(df0: pd.DataFrame) -> tuple[int, int]:
    for i in range(min(80, df0.shape[0])):
        row = df0.iloc[i, :].astype(str).tolist()
        hits = sum(bool(re.fullmatch(r"c\d+", x.strip())) for x in row if x not in ("nan", "None"))
        if hits >= 20:
            for j, x in enumerate(row):
                if x.strip() == "c1":
                    return i, j
    raise ValueError("Kunne ikke finde sector-code header row (c1,c2,...) i de første 80 rækker.")

def is_country_code(x) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip()
    return bool(re.fullmatch(r"[A-Z]{3}|RoW", x))

def read_intermediate_block(in_path: str, sheet_mrio: str, sheet_legend: str):
    xls = pd.ExcelFile(in_path, engine="openpyxl")
    if sheet_mrio is None:
        sheet_mrio = detect_mrio_sheet(xls)

    legend = pd.read_excel(in_path, sheet_name=sheet_legend, engine="openpyxl")
    legend = legend.rename(columns={legend.columns[0]: "Code", legend.columns[1]: "Country"})
    ctry_code_to_name = dict(zip(legend["Code"].astype(str), legend["Country"].astype(str)))
    ctry_name_to_code = dict(zip(legend["Country"].astype(str), legend["Code"].astype(str)))

    df0 = pd.read_excel(in_path, sheet_name=sheet_mrio, header=None, engine="openpyxl")

    sector_code_row, col_start = find_header_row(df0)
    country_code_row = sector_code_row - 1
    sector_name_row = sector_code_row - 2
    data_start_row = sector_code_row + 1

    col_countries = df0.iloc[country_code_row, col_start:].tolist()
    col_sectors = df0.iloc[sector_code_row, col_start:].tolist()

    int_cols = []
    for k, (cc, sc) in enumerate(zip(col_countries, col_sectors)):
        if not (isinstance(sc, str) and re.fullmatch(r"c\d+", sc.strip())):
            break
        if not is_country_code(str(cc)):
            break
        int_cols.append(col_start + k)

    if len(int_cols) < 200:
        raise ValueError(f"Fandt kun {len(int_cols)} intermediate kolonner – noget tyder på andet layout.")

    sector_codes_35 = [str(x).strip() for x in df0.iloc[sector_code_row, col_start:col_start+35].tolist()]
    sector_names_35 = [str(x).strip() for x in df0.iloc[sector_name_row, col_start:col_start+35].tolist()]
    sector_lookup = dict(zip(sector_codes_35, sector_names_35))

    row_sectorname_col = col_start - 3
    row_country_col = col_start - 2
    row_sectorcode_col = col_start - 1

    rows = []
    row_keys = []
    for i in range(data_start_row, df0.shape[0]):
        cc = df0.iat[i, row_country_col]
        sc = df0.iat[i, row_sectorcode_col]
        sn = df0.iat[i, row_sectorname_col]

        if not is_country_code(str(cc)):
            break
        if not (isinstance(sc, str) and re.fullmatch(r"c\d+", sc.strip())):
            break
        if pd.isna(sn):
            break

        rows.append(i)
        row_keys.append((str(cc).strip(), str(sc).strip()))

    if len(rows) < 200:
        raise ValueError(f"Fandt kun {len(rows)} intermediate rækker – noget tyder på andet layout.")

    Z = df0.iloc[rows, int_cols].copy()
    Z = Z.apply(pd.to_numeric, errors="coerce")

    Z.index = pd.MultiIndex.from_tuples(row_keys, names=["exp_country", "exp_sector"])
    col_keys = [(str(df0.iat[country_code_row, j]).strip(), str(df0.iat[sector_code_row, j]).strip()) for j in int_cols]
    Z.columns = pd.MultiIndex.from_tuples(col_keys, names=["imp_country", "imp_sector"])

    return Z, sector_lookup, ctry_code_to_name, ctry_name_to_code

def to_long(Z: pd.DataFrame, sector_lookup: dict, ctry_code_to_name: dict,
            exporters: list[str], importers: list[str], label: str) -> pd.DataFrame:
    sub = Z.loc[Z.index.get_level_values("exp_country").isin(exporters),
                Z.columns.get_level_values("imp_country").isin(importers)]
    long = sub.stack(["imp_country", "imp_sector"]).reset_index()
    long = long.rename(columns={0: "value"})
    long["scenario"] = label

    long["exp_country_name"] = long["exp_country"].map(ctry_code_to_name)
    long["imp_country_name"] = long["imp_country"].map(ctry_code_to_name)
    long["exp_sector_name"] = long["exp_sector"].map(sector_lookup)
    long["imp_sector_name"] = long["imp_sector"].map(sector_lookup)
    return long

def collapse_by_importer_industry(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (long_df
           .groupby(group_cols, as_index=False)["value"]
           .sum(min_count=1))
    out["view"] = "by_importer_industry"
    return out

# ---- NEW: add EA aggregate rows (EA as exporter/importer) ----
def add_area_aggregate_long(long_df: pd.DataFrame,
                            area_code: str,
                            area_name: str,
                            member_exporter_codes: list[str] | None = None,
                            member_importer_codes: list[str] | None = None) -> pd.DataFrame:
    """
    Takes a 'detailed_Z_long'-style dataframe and appends rows where:
      - exp_country is replaced by area_code and summed over member_exporter_codes (EA as exporter)
      - imp_country is replaced by area_code and summed over member_importer_codes (EA as importer)
      - both replaced (EA -> EA) if both lists provided
    Keeps sector detail (exp_sector, imp_sector).
    """

    df = long_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out_parts = [df]

    keys = ["exp_country","exp_sector","imp_country","imp_sector","scenario","view",
            "exp_country_name","imp_country_name","exp_sector_name","imp_sector_name"]

    if member_exporter_codes is not None:
        exp = df[df["exp_country"].isin(member_exporter_codes)].copy()
        if not exp.empty:
            exp["exp_country"] = area_code
            exp["exp_country_name"] = area_name
            exp = exp.groupby(keys, as_index=False)["value"].sum(min_count=1)
            out_parts.append(exp)

    if member_importer_codes is not None:
        imp = df[df["imp_country"].isin(member_importer_codes)].copy()
        if not imp.empty:
            imp["imp_country"] = area_code
            imp["imp_country_name"] = area_name
            imp = imp.groupby(keys, as_index=False)["value"].sum(min_count=1)
            out_parts.append(imp)

    if (member_exporter_codes is not None) and (member_importer_codes is not None):
        both = df[df["exp_country"].isin(member_exporter_codes) & df["imp_country"].isin(member_importer_codes)].copy()
        if not both.empty:
            both["exp_country"] = area_code
            both["imp_country"] = area_code
            both["exp_country_name"] = area_name
            both["imp_country_name"] = area_name
            both = both.groupby(keys, as_index=False)["value"].sum(min_count=1)
            out_parts.append(both)

    return pd.concat(out_parts, ignore_index=True)

# -----------------------------
# Main
# -----------------------------
def main():
    out_path = OUT_LONG_PATH

    Z, sector_lookup, ctry_code_to_name, ctry_name_to_code = read_intermediate_block(
        IN_PATH, SHEET_MRIO, SHEET_LEGEND
    )

    DEN = ctry_name_to_code.get("Denmark", "DEN")
    USA = ctry_name_to_code.get("United States", "USA")

    present_countries = set(ctry_name_to_code.keys())
    ea_present_names = sorted(EA20.intersection(present_countries))
    ea_present_codes = [ctry_name_to_code[n] for n in ea_present_names]

    # DK import from USA
    long_us_to_dk = to_long(
        Z, sector_lookup, ctry_code_to_name,
        exporters=[USA], importers=[DEN],
        label="DK_imports_from_USA"
    )
    long_us_to_dk["view"] = "by_exporter_and_importer_industry"

    # DK import from EA: exporters = EA countries, importer = DK
    long_ea_to_dk = to_long(
        Z, sector_lookup, ctry_code_to_name,
        exporters=ea_present_codes, importers=[DEN],
        label="DK_imports_from_EA"
    )
    long_ea_to_dk["view"] = "by_exporter_and_importer_industry"

    # EA import from USA: exporter = USA, importers = EA countries
    long_us_to_ea = to_long(
        Z, sector_lookup, ctry_code_to_name,
        exporters=[USA], importers=ea_present_codes,
        label="EA_imports_from_USA"
    )
    long_us_to_ea["view"] = "by_exporter_and_importer_industry"

    # Combine detailed
    long_all = pd.concat([long_us_to_dk, long_ea_to_dk, long_us_to_ea], ignore_index=True)

    # ---- NEW: add EA pseudo-country rows in the detailed table ----
    # For each scenario, add EA as exporter and/or importer:
    # - DK_imports_from_EA: create exp_country = EA (sum over EA exporters)
    # - EA_imports_from_USA: create imp_country = EA (sum over EA importers)
    # Additionally EA->EA rows (optional) from the full long_all slice.
    long_all_with_ea = pd.concat([
        # Add EA as exporter within DK_imports_from_EA
        add_area_aggregate_long(
            long_all[long_all["scenario"] == "DK_imports_from_EA"],
            area_code=EA_CODE, area_name=EA_NAME,
            member_exporter_codes=ea_present_codes,
            member_importer_codes=None
        ),
        # Add EA as importer within EA_imports_from_USA
        add_area_aggregate_long(
            long_all[long_all["scenario"] == "EA_imports_from_USA"],
            area_code=EA_CODE, area_name=EA_NAME,
            member_exporter_codes=None,
            member_importer_codes=ea_present_codes
        ),
        # Keep DK_imports_from_USA unchanged (no EA dimension inside)
        long_all[long_all["scenario"] == "DK_imports_from_USA"].copy()
    ], ignore_index=True)

    # Collapsed views (per importer industry)
    collapsed = []
    collapsed.append(
        collapse_by_importer_industry(
            long_us_to_dk,
            group_cols=["scenario", "imp_country", "imp_country_name", "imp_sector", "imp_sector_name"]
        )
    )
    collapsed.append(
        collapse_by_importer_industry(
            long_ea_to_dk,
            group_cols=["scenario", "imp_country", "imp_country_name", "imp_sector", "imp_sector_name"]
        )
    )
    collapsed.append(
        collapse_by_importer_industry(
            long_us_to_ea,
            group_cols=["scenario", "imp_country", "imp_country_name", "imp_sector", "imp_sector_name"]
        )
    )
    collapsed_all = pd.concat(collapsed, ignore_index=True)

    # ---- NEW: add EA pseudo-country rows to collapsed views ----
    # DK_imports_from_EA: exp side already collapsed away, so importer is DK; EA pseudo-country not relevant here.
    # EA_imports_from_USA: importer is EA countries; we want an EA aggregate importer row.
    # We'll build it by summing the EA importers into imp_country = EA.
    collapsed_ea = collapsed_all[collapsed_all["scenario"] == "EA_imports_from_USA"].copy()
    if not collapsed_ea.empty:
        collapsed_ea["imp_country"] = EA_CODE
        collapsed_ea["imp_country_name"] = EA_NAME
        collapsed_ea = (collapsed_ea
                        .groupby(["scenario","imp_country","imp_country_name","imp_sector","imp_sector_name","view"], as_index=False)["value"]
                        .sum(min_count=1))
        collapsed_all_with_ea = pd.concat([collapsed_all, collapsed_ea], ignore_index=True)
    else:
        collapsed_all_with_ea = collapsed_all

    # Write to Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        long_all_with_ea.to_excel(writer, sheet_name="detailed_Z_long", index=False)
        collapsed_all_with_ea.to_excel(writer, sheet_name="collapsed_importer_ind", index=False)

    print("Saved:", out_path)
    print("EA countries present in file:", ea_present_names)
    if "Slovakia" not in ea_present_names:
        print("NOTE: Slovakia not present in this MRIO file; EA aggregate is EA minus missing members.")

if __name__ == "__main__":
    main()

