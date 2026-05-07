#!/usr/bin/env python3
"""
Append Denmark -> Euro Area export rows to the `Export` table in the
`Eksport Tabel` sheet of import_export_sector_shares.xlsx.

What it does
------------
1. Reads the ADB MRIO workbook and calculates Danish exports to available EA20
   countries by ISIC sector:
   - intermediate exports = Danish sector rows to each EA destination's 35
     intermediate-use columns
   - final exports = Danish sector rows to each EA destination's 5 final-demand
     columns
2. Reads the NACE mapping already used in the target workbook (`Import tabel`,
   table `Mapping`) and aggregates ISIC rows to the existing NACE sectors.
3. Appends 16 rows at the bottom of `Eksport Tabel` / table `Export`:
   DEN, Denmark -> EA, Euro area (EA20).
4. Updates the Excel table range so the new rows are part of the `Export` table.

The sheet can be protected/locked. This script edits the xlsx package directly
and preserves the sheet protection; Excel may still ask for a password if you try
manual edits in the UI.
"""

from __future__ import annotations

import argparse
import math
import posixpath
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd


EA20_CODES = [
    "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "GER", "GRC", "IRE",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NET", "POR", "SVK", "SVN", "SPA",
]

# Fallback order used by the ADB MRIO workbook if the sector-code column cannot
# be detected from the first metadata columns. The workbook usually contains a
# code column, so this is only a safety net.
ADB_ISIC_ORDER = [
    "c1", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18",
    "c19", "c2", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27",
    "c28", "c29", "c3", "c30", "c31", "c32", "c33", "c34", "c35", "c4",
    "c5", "c6", "c7", "c8", "c9",
]

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

M = f"{{{MAIN_NS}}}"
R = f"{{{REL_NS}}}"
PKG = f"{{{PKG_REL_NS}}}"

ET.register_namespace("", MAIN_NS)
ET.register_namespace("r", REL_NS)
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("xr", "http://schemas.microsoft.com/office/spreadsheetml/2014/revision")
ET.register_namespace("xr3", "http://schemas.microsoft.com/office/spreadsheetml/2016/revision3")
ET.register_namespace("x15", "http://schemas.microsoft.com/office/spreadsheetml/2010/11/main")
ET.register_namespace("x14ac", "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac")
ET.register_namespace("xr2", "http://schemas.microsoft.com/office/spreadsheetml/2015/revision2")
ET.register_namespace("xr6", "http://schemas.microsoft.com/office/spreadsheetml/2016/revision6")
ET.register_namespace("xr10", "http://schemas.microsoft.com/office/spreadsheetml/2016/revision10")
ET.register_namespace("x15ac", "http://schemas.microsoft.com/office/spreadsheetml/2010/11/ac")


@dataclass
class SharedStrings:
    root: ET.Element
    values: list[str]
    index: dict[str, int]
    added_occurrences: int = 0

    @classmethod
    def from_zip(cls, zin: zipfile.ZipFile) -> "SharedStrings":
        root = ET.fromstring(zin.read("xl/sharedStrings.xml"))
        values: list[str] = []
        index: dict[str, int] = {}
        for i, si in enumerate(root.findall(f"{M}si")):
            text = "".join(t.text or "" for t in si.iter(f"{M}t"))
            values.append(text)
            index.setdefault(text, i)
        return cls(root=root, values=values, index=index)

    def get_index(self, text: str) -> int:
        text = "" if text is None else str(text)
        self.added_occurrences += 1
        if text in self.index:
            return self.index[text]
        si = ET.SubElement(self.root, f"{M}si")
        t = ET.SubElement(si, f"{M}t")
        if text[:1].isspace() or text[-1:].isspace():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        idx = len(self.values)
        self.values.append(text)
        self.index[text] = idx
        return idx

    def cell_text(self, cell: ET.Element | None) -> str:
        if cell is None:
            return ""
        v = cell.find(f"{M}v")
        if v is None or v.text is None:
            inline = cell.find(f"{M}is/{M}t")
            return inline.text if inline is not None and inline.text is not None else ""
        if cell.get("t") == "s":
            return self.values[int(v.text)]
        return v.text

    def to_bytes(self) -> bytes:
        if self.added_occurrences:
            old_count = int(self.root.get("count", str(len(self.values))))
            self.root.set("count", str(old_count + self.added_occurrences))
        self.root.set("uniqueCount", str(len(self.values)))
        return ET.tostring(self.root, encoding="utf-8", xml_declaration=True)


def excel_col_letter(col_num: int) -> str:
    result = ""
    while col_num:
        col_num, rem = divmod(col_num - 1, 26)
        result = chr(65 + rem) + result
    return result


def col_to_index(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + ord(ch.upper()) - 64
    return n


def split_cell_ref(ref: str) -> tuple[str, int]:
    m = re.fullmatch(r"([A-Z]+)(\d+)", ref)
    if not m:
        raise ValueError(f"Invalid cell reference: {ref}")
    return m.group(1), int(m.group(2))


def parse_range(ref: str) -> tuple[str, int, str, int]:
    start, end = ref.split(":")
    c1, r1 = split_cell_ref(start)
    c2, r2 = split_cell_ref(end)
    return c1, r1, c2, r2


def country_intermediate_columns(country_index_zero_based: int) -> list[int]:
    """1-based Excel columns for the destination's 35 intermediate-use columns."""
    start = 5 + 35 * country_index_zero_based
    return list(range(start, start + 35))


def country_final_demand_columns(country_index_zero_based: int) -> list[int]:
    """1-based Excel columns for the destination's 5 final-demand columns."""
    start = 2630 + 5 * country_index_zero_based
    return list(range(start, start + 5))


def normalize_text(x: object) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def normalize_isic_code(x: object) -> str:
    s = str(x).strip().lower()
    m = re.fullmatch(r"c0*(\d+)", s)
    return f"c{int(m.group(1))}" if m else s


def sheet_path_by_name(zin: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(zin.read("xl/workbook.xml"))
    rels = ET.fromstring(zin.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target") for rel in rels.findall(f"{PKG}Relationship")}

    available = []
    for sheet in workbook.find(f"{M}sheets"):
        name = sheet.get("name")
        available.append(name)
        if name == sheet_name:
            rid = sheet.get(f"{R}id")
            target = targets[rid]
            return posixpath.normpath("xl/" + target.lstrip("/"))

    raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {available}")


def first_existing_sheet_path(zin: zipfile.ZipFile, names: Iterable[str]) -> tuple[str, str]:
    workbook = ET.fromstring(zin.read("xl/workbook.xml"))
    available = [s.get("name") for s in workbook.find(f"{M}sheets")]
    for name in names:
        if name in available:
            return name, sheet_path_by_name(zin, name)
    raise ValueError(f"None of these sheets were found: {list(names)}. Available sheets: {available}")


def table_path_from_sheet(
    zin: zipfile.ZipFile,
    sheet_path: str,
    table_name: str,
) -> str:
    rels_path = posixpath.join(
        posixpath.dirname(sheet_path),
        "_rels",
        posixpath.basename(sheet_path) + ".rels",
    )
    rels = ET.fromstring(zin.read(rels_path))
    sheet_dir = posixpath.dirname(sheet_path)

    candidates = []
    for rel in rels.findall(f"{PKG}Relationship"):
        if not (rel.get("Type") or "").endswith("/table"):
            continue
        target = posixpath.normpath(posixpath.join(sheet_dir, rel.get("Target")))
        table_root = ET.fromstring(zin.read(target))
        candidates.append((target, table_root.get("name"), table_root.get("displayName")))
        if table_root.get("name") == table_name or table_root.get("displayName") == table_name:
            return target

    raise ValueError(f"Table {table_name!r} not found on sheet {sheet_path}. Found: {candidates}")


def row_cells_by_col(row: ET.Element) -> dict[int, ET.Element]:
    out: dict[int, ET.Element] = {}
    for cell in row.findall(f"{M}c"):
        col_letters, _ = split_cell_ref(cell.get("r"))
        out[col_to_index(col_letters)] = cell
    return out


def read_table_rows(
    sheet_root: ET.Element,
    table_ref: str,
    shared: SharedStrings,
) -> list[dict[str, str]]:
    c1, r1, c2, r2 = parse_range(table_ref)
    start_col = col_to_index(c1)
    end_col = col_to_index(c2)
    sheet_data = sheet_root.find(f"{M}sheetData")

    rows_by_num = {int(row.get("r")): row for row in sheet_data.findall(f"{M}row")}
    header_row = rows_by_num[r1]
    header_cells = row_cells_by_col(header_row)
    headers = [
        shared.cell_text(header_cells.get(col))
        for col in range(start_col, end_col + 1)
    ]

    records: list[dict[str, str]] = []
    for r in range(r1 + 1, r2 + 1):
        row = rows_by_num.get(r)
        if row is None:
            continue
        cells = row_cells_by_col(row)
        values = [shared.cell_text(cells.get(col)) for col in range(start_col, end_col + 1)]
        records.append(dict(zip(headers, values)))
    return records


def read_mapping_from_target(target_workbook: Path) -> pd.DataFrame:
    with zipfile.ZipFile(target_workbook) as zin:
        shared = SharedStrings.from_zip(zin)
        _, mapping_sheet_path = first_existing_sheet_path(zin, ["Import tabel", "Import table"])
        mapping_table_path = table_path_from_sheet(zin, mapping_sheet_path, "Mapping")
        mapping_root = ET.fromstring(zin.read(mapping_table_path))
        mapping_sheet_root = ET.fromstring(zin.read(mapping_sheet_path))
        records = read_table_rows(mapping_sheet_root, mapping_root.get("ref"), shared)

    df = pd.DataFrame(records)
    required = ["ISIC_sector_code", "ISIC_sector", "NACE_sector_code", "NACE_sector_name", "Import Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping table is missing columns: {missing}")

    # Use the Denmark block. The NACE mapping is the same across blocks, but this
    # avoids duplicates from the Euro Area and US blocks.
    den = df[df["Import Country"].str.casefold() == "denmark"].copy()
    if den.empty:
        # Fallback: first occurrence per ISIC sector if the Import Country label changes.
        den = df.drop_duplicates("ISIC_sector_code").copy()

    den["ISIC_sector_code"] = den["ISIC_sector_code"].map(normalize_isic_code)
    den = den.drop_duplicates("ISIC_sector_code")
    return den[required[:-1]].reset_index(drop=True)


def read_export_table_order(target_workbook: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(target_workbook) as zin:
        shared = SharedStrings.from_zip(zin)
        _, export_sheet_path = first_existing_sheet_path(zin, ["Eksport Tabel", "Export Tabel", "Export table"])
        export_table_path = table_path_from_sheet(zin, export_sheet_path, "Export")
        export_root = ET.fromstring(zin.read(export_table_path))
        export_sheet_root = ET.fromstring(zin.read(export_sheet_path))
        records = read_table_rows(export_sheet_root, export_root.get("ref"), shared)

    rows = [r for r in records if r.get("exp_country_code") == "DEN" and r.get("dest_country_code") == "USA"]
    if not rows:
        # Fallback: use first country block in the table.
        rows = records[:16]
    order = []
    for r in rows:
        item = (r["NACE_sector_code"], r["NACE_sector_name"])
        if item not in order:
            order.append(item)
    return order


def detect_sector_codes(meta: pd.DataFrame, mapping: pd.DataFrame) -> pd.Series:
    # 1) Direct code detection: c1, c2, ..., c35.
    for col in meta.columns:
        vals = meta[col].astype(str).map(normalize_isic_code)
        if vals.str.fullmatch(r"c\d+").sum() >= 30:
            return vals

    # 2) Sector-name detection using the target workbook's Mapping table.
    name_to_code = {
        normalize_text(name): code
        for name, code in zip(mapping["ISIC_sector"], mapping["ISIC_sector_code"])
    }
    for col in meta.columns:
        vals_norm = meta[col].map(normalize_text)
        matched = vals_norm.map(name_to_code)
        if matched.notna().sum() >= 30:
            return matched

    # 3) Fallback to the known ADB MRIO sector order.
    if len(meta) == len(ADB_ISIC_ORDER):
        return pd.Series(ADB_ISIC_ORDER, index=meta.index)

    raise ValueError(
        "Could not detect the ISIC sector codes in the ADB MRIO sheet. "
        "Expected one of the first metadata columns to contain c1...c35, or "
        "sector names matching the target workbook's Mapping table."
    )


def build_denmark_to_ea_by_isic(mrio_path: Path, mapping: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    legend = pd.read_excel(mrio_path, sheet_name="Legend", header=None, usecols="A:B")
    legend = legend.dropna(how="all")
    legend.columns = ["Code", "Country"]
    if str(legend.iloc[0, 0]).strip() == "Code":
        legend = legend.iloc[1:].reset_index(drop=True)

    country_codes = legend["Code"].astype(str).str.strip().tolist()
    country_index = {code: i for i, code in enumerate(country_codes)}

    if "DEN" not in country_index:
        raise ValueError("DEN was not found in the Legend sheet.")

    available_ea = [code for code in EA20_CODES if code in country_index]
    missing_ea = [code for code in EA20_CODES if code not in country_index]
    if not available_ea:
        raise ValueError("None of the EA20 countries were found in the MRIO Legend sheet.")

    den_index = country_index["DEN"]
    den_row_start = 8 + 35 * den_index  # first Danish sector row, 1-based Excel row number

    ea_inter_cols = [col for code in available_ea for col in country_intermediate_columns(country_index[code])]
    ea_fd_cols = [col for code in available_ea for col in country_final_demand_columns(country_index[code])]

    # Read metadata columns B:D plus the destination columns needed.
    usecols = [2, 3, 4] + ea_inter_cols + ea_fd_cols
    usecols_excel = ",".join(excel_col_letter(c) for c in usecols)

    df = pd.read_excel(
        mrio_path,
        sheet_name="ADB MRIO 2024",
        header=None,
        skiprows=den_row_start - 1,
        nrows=35,
        usecols=usecols_excel,
    )

    meta = df.iloc[:, :3]
    sector_codes = detect_sector_codes(meta, mapping)

    inter_block = df.iloc[:, 3 : 3 + len(ea_inter_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    final_block = df.iloc[:, 3 + len(ea_inter_cols) :].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    out = pd.DataFrame(
        {
            "ISIC_sector_code": sector_codes.map(normalize_isic_code),
            "exports_intermediate": inter_block.sum(axis=1),
            "exports_final": final_block.sum(axis=1),
        }
    )
    out = out.groupby("ISIC_sector_code", as_index=False).sum()
    return out, available_ea, missing_ea


def build_rows_to_append(target_workbook: Path, mrio_path: Path) -> tuple[list[list[object]], list[str], list[str]]:
    mapping = read_mapping_from_target(target_workbook)
    nace_order = read_export_table_order(target_workbook)
    exports_by_isic, available_ea, missing_ea = build_denmark_to_ea_by_isic(mrio_path, mapping)

    merged = mapping.merge(exports_by_isic, on="ISIC_sector_code", how="left")
    merged[["exports_intermediate", "exports_final"]] = merged[["exports_intermediate", "exports_final"]].fillna(0.0)

    not_mapped = sorted(set(exports_by_isic["ISIC_sector_code"]) - set(mapping["ISIC_sector_code"]))
    if not_mapped:
        print("Warning: these MRIO ISIC sectors are not in the target workbook Mapping table and are ignored:")
        print(", ".join(not_mapped))

    grouped = (
        merged.groupby(["NACE_sector_code", "NACE_sector_name"], as_index=False)[["exports_intermediate", "exports_final"]]
        .sum()
    )
    grouped["exports_total"] = grouped["exports_intermediate"] + grouped["exports_final"]
    lookup = {
        (row["NACE_sector_code"], row["NACE_sector_name"]): row
        for _, row in grouped.iterrows()
    }

    rows: list[list[object]] = []
    for code, name in nace_order:
        vals = lookup.get((code, name))
        inter = float(vals["exports_intermediate"]) if vals is not None else 0.0
        final = float(vals["exports_final"]) if vals is not None else 0.0
        rows.append([
            "DEN",
            "Denmark",
            "EA",
            "Euro area (EA20)",
            code,
            name,
            inter,
            final,
            inter + final,
        ])
    return rows, available_ea, missing_ea


def make_string_cell(ref: str, value: str, shared: SharedStrings) -> ET.Element:
    cell = ET.Element(f"{M}c", {"r": ref, "t": "s"})
    v = ET.SubElement(cell, f"{M}v")
    v.text = str(shared.get_index(value))
    return cell


def make_number_cell(ref: str, value: object) -> ET.Element:
    x = float(value)
    if not math.isfinite(x):
        x = 0.0
    cell = ET.Element(f"{M}c", {"r": ref})
    v = ET.SubElement(cell, f"{M}v")
    v.text = format(x, ".15g")
    return cell


def existing_den_to_ea_rows(sheet_root: ET.Element, table_ref: str, shared: SharedStrings) -> int:
    records = read_table_rows(sheet_root, table_ref, shared)
    return sum(1 for r in records if r.get("exp_country_code") == "DEN" and r.get("dest_country_code") == "EA")


def set_workbook_full_calc(workbook_xml: bytes) -> bytes:
    root = ET.fromstring(workbook_xml)
    calc_pr = root.find(f"{M}calcPr")
    if calc_pr is None:
        calc_pr = ET.SubElement(root, f"{M}calcPr")
    calc_pr.set("calcMode", "auto")
    calc_pr.set("fullCalcOnLoad", "1")
    calc_pr.set("forceFullCalc", "1")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_row_xml(row_number: int, row_values: list[object], shared: SharedStrings) -> str:
    """
    Build one worksheet row as raw XML.

    Important: we insert raw row XML instead of re-serialising the whole sheet.
    That preserves Excel's original namespace declarations on the protected sheet.
    Re-serialising that sheet with ElementTree can make Excel report that the
    workbook is damaged because mc:Ignorable prefixes such as x14ac/xr2 may be
    rewritten or dropped.
    """
    cells: list[str] = []
    for j, value in enumerate(row_values, start=1):
        ref = f"{excel_col_letter(j)}{row_number}"
        if j <= 6:
            # Use inline strings for the appended rows. This avoids rewriting
            # xl/sharedStrings.xml, which makes the patch much less invasive.
            text = xml_escape(str(value))
            preserve = ' xml:space="preserve"' if str(value)[:1].isspace() or str(value)[-1:].isspace() else ""
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>')
        else:
            x = float(value)
            if not math.isfinite(x):
                x = 0.0
            cells.append(f'<c r="{ref}"><v>{format(x, ".15g")}</v></c>')

    # Match the existing rows in the protected Export sheet: they are hidden
    # and carry x14ac:dyDescent. Because we do not reserialise the sheet XML,
    # the original x14ac namespace declaration is preserved.
    return (
        f'<row r="{row_number}" spans="1:9" hidden="1" x14ac:dyDescent="0.75">'
        + "".join(cells)
        + "</row>"
    )


def replace_first_or_fail(text: str, old: str, new: str, context: str) -> str:
    if old not in text:
        raise ValueError(f"Could not find {old!r} while updating {context}.")
    return text.replace(old, new, 1)


def append_rows_to_export_table(
    target_workbook: Path,
    output_workbook: Path,
    rows_to_append: list[list[object]],
) -> None:
    if output_workbook.resolve() == target_workbook.resolve():
        raise ValueError("Refusing to overwrite the input workbook directly. Choose a different output path.")

    with zipfile.ZipFile(target_workbook, "r") as zin:
        shared = SharedStrings.from_zip(zin)
        export_sheet_name, export_sheet_path = first_existing_sheet_path(
            zin, ["Eksport Tabel", "Export Tabel", "Export table"]
        )
        export_table_path = table_path_from_sheet(zin, export_sheet_path, "Export")

        # Read/parse only for locating the current table and duplicate checks.
        # We do NOT write the parsed worksheet back.
        sheet_root = ET.fromstring(zin.read(export_sheet_path))
        table_root = ET.fromstring(zin.read(export_table_path))
        table_ref = table_root.get("ref")
        c1, header_row, c2, last_row = parse_range(table_ref)
        if c1 != "A" or c2 != "I":
            raise ValueError(f"Expected Export table to be A:I, but found {table_ref}")

        n_existing = existing_den_to_ea_rows(sheet_root, table_ref, shared)
        if n_existing:
            raise ValueError(
                f"Found {n_existing} existing DEN -> EA rows in {export_sheet_name!r}. "
                "To avoid duplicates, delete those rows before rerunning this script."
            )

        start_row = last_row + 1
        new_last_row = last_row + len(rows_to_append)
        old_ref = f"A{header_row}:I{last_row}"
        new_ref = f"A{header_row}:I{new_last_row}"

        # Preserve original sheet XML exactly, except for dimension + inserted rows.
        sheet_xml = zin.read(export_sheet_path).decode("utf-8")
        sheet_xml = replace_first_or_fail(sheet_xml, f'<dimension ref="{old_ref}"', f'<dimension ref="{new_ref}"', "worksheet dimension")
        insert_at = sheet_xml.rfind("</sheetData>")
        if insert_at == -1:
            raise ValueError(f"Could not find </sheetData> in {export_sheet_path}.")
        new_rows_xml = "".join(
            make_row_xml(i, row_values, shared)
            for i, row_values in enumerate(rows_to_append, start=start_row)
        )
        sheet_xml = sheet_xml[:insert_at] + new_rows_xml + sheet_xml[insert_at:]

        # Preserve original table XML exactly, except for the table/autofilter refs.
        table_xml = zin.read(export_table_path).decode("utf-8")
        table_xml = table_xml.replace(f'ref="{old_ref}"', f'ref="{new_ref}"')

        replacements = {
            export_sheet_path: sheet_xml.encode("utf-8"),
            export_table_path: table_xml.encode("utf-8"),
            # Deliberately do not rewrite xl/sharedStrings.xml or xl/workbook.xml.
            # Rewriting either can disturb Excel-specific XML details and make
            # the workbook appear corrupted even though the data are correct.
        }

        # Write a fresh xlsx package with only the needed XML parts replaced.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp_path = Path(tmp.name)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zout:
                for item in zin.infolist():
                    data = replacements.get(item.filename)
                    if data is None:
                        data = zin.read(item.filename)
                    zout.writestr(item, data)
            shutil.move(str(tmp_path), output_workbook)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_with_DEN_to_EA_exports" + input_path.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append Denmark -> Euro Area export rows to the Export table."
    )
    parser.add_argument(
        "target_workbook",
        nargs="?",
        default=r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data\import_export_sector_shares.xlsx",
        help="Path to the workbook that contains the 'Eksport Tabel' sheet.",
    )
    parser.add_argument(
        "mrio_workbook",
        nargs="?",
        default=r"\\srv9dnbfil002\userhome\kons-ana\Desktop\Data\ADB-MRIO-2024-August 2025.xlsx",
        help="Path to the ADB MRIO workbook.",
    )
    parser.add_argument(
        "output_workbook",
        nargs="?",
        default=None,
        help="Output path. Defaults to <target>_with_DEN_to_EA_exports.xlsx.",
    )
    args = parser.parse_args()

    target_path = Path(args.target_workbook)
    mrio_path = Path(args.mrio_workbook)
    output_path = Path(args.output_workbook) if args.output_workbook else default_output_path(target_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Target workbook not found: {target_path}")
    if not mrio_path.exists():
        raise FileNotFoundError(f"MRIO workbook not found: {mrio_path}")

    print("Calculating Denmark -> EA exports from MRIO...")
    rows_to_append, available_ea, missing_ea = build_rows_to_append(target_path, mrio_path)

    print(f"Appending {len(rows_to_append)} rows to the Export table...")
    append_rows_to_export_table(target_path, output_path, rows_to_append)

    print(f"Created: {output_path}")
    print(f"Included EA countries ({len(available_ea)}): {', '.join(available_ea)}")
    print("Missing EA countries: " + (", ".join(missing_ea) if missing_ea else "None"))
    print(
        "Note: If downstream formulas use Export without filtering dest_country_code, "
        "check whether they should now distinguish USA vs EA."
    )


if __name__ == "__main__":
    main()
