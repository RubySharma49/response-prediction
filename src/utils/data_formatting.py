
import pandas as pd
import numpy as np
import re

def standarize_sex_colum(df, column_nm = "sex"):
    df = df.copy()
    df[column_nm] = df[column_nm].replace({"M": "Male", "F":"Female","MALE":"Male", "FEMALE":"Female"})
    return df

def extract_patient_info(df):
    print("No of unique subjects:" ,len(df["SUBJID"].unique()))
    print("No of unique cohorts:" ,len(df["Cohort"].unique()),"-",df["Cohort"].unique())
    print("No of unique arms:" ,len(df["Arm"].unique()), "-", df["Arm"].unique())
    print("No of patients with IMDC score:" ,df["IMDC"].notna().sum())
    print("Number of responders and non-responder:",df["ORR"].value_counts())
    print("Sample Type:",df["SampleType"].value_counts())
    

def match_patients(clinical_data,  column_to_match, id_to_match):
    pt_found = []
    ind = 0 
    for id in id_to_match:
        mt = clinical_data.loc[clinical_data[column_to_match] == id , column_to_match]
        if(len(mt)!=0):
             pt_found.append(ind)
        ind = ind+1  

    return pt_found



def select_representative_probes(
    df: pd.DataFrame,
    probe_id_col: str,
    gene_symbol_col: str,
    gene_desc_col: str = None,
) -> pd.DataFrame:
    """
    Select a representative probe per gene based on probe ID suffix priority
    and highest variance across samples.

    Priority order:
        1. *_at   (exact — excludes _s_at and _x_at)
        2. *_s_at
        3. *_x_at
        4. anything else (fallback)
    Within each priority tier, the probe with the highest variance
    across sample expression columns is chosen.

    Parameters
    ----------
    df            : pd.DataFrame
                    Input dataframe containing probe expression data.
    probe_id_col  : str
                    Column name for probe IDs (e.g. '1007_s_at').
    gene_symbol_col : str
                    Column name for gene symbols (e.g. 'BRCA1').
    gene_desc_col : str, optional
                    Column name for gene descriptions. If None or not
                    found in df, it is skipped silently.

    Returns
    -------
    pd.DataFrame
        One row per unique gene with columns:
        [gene_symbol, representative_probe_id, (gene_description), ...samples]

 
    """

    # ── Validate required columns ──────────────────────────────────────────────
    for col in [probe_id_col, gene_symbol_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # ── Resolve gene description column ───────────────────────────────────────
    use_desc = (
        gene_desc_col is not None
        and gene_desc_col in df.columns
    )

    # ── Auto-detect sample expression columns ─────────────────────────────────
    # Everything that is not a metadata column is treated as expression data
    meta_cols = {probe_id_col, gene_symbol_col}
    if use_desc:
        meta_cols.add(gene_desc_col)

    sample_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not sample_cols:
        raise ValueError(
            "No numeric sample expression columns found. "
            "Check that expression columns are numeric dtype."
        )

    # ── Probe priority classifier ──────────────────────────────────────────────
    def probe_priority(probe_id: str) -> int:
        """
        Returns priority rank (lower = higher priority).
            0 → ends with _at   (pure _at, not _s_at / _x_at)
            1 → ends with _s_at
            2 → ends with _x_at
            3 → anything else
        """
        pid = str(probe_id).lower().strip()
        if re.search(r'_s_at$', pid):
            return 1
        if re.search(r'_x_at$', pid):
            return 2
        if re.search(r'_at$', pid):
            return 0
        return 3

    # ── Add working columns ────────────────────────────────────────────────────
    working = df.copy()
    working['_priority'] = working[probe_id_col].apply(probe_priority)
    working['_variance'] = working[sample_cols].var(axis=1)

    # ── Select best probe per gene ─────────────────────────────────────────────
    def pick_best_probe(group: pd.DataFrame) -> pd.Series:
        """
        Within a gene group, pick the probe with the best priority tier.
        Ties within a tier are broken by highest variance.
        """
        best_priority = group['_priority'].min()
        candidates    = group[group['_priority'] == best_priority]
        best_row      = candidates.loc[candidates['_variance'].idxmax()]
        return best_row

    best_probes = (
        working
        .groupby(gene_symbol_col, sort=False)
        .apply(pick_best_probe)
        .reset_index(drop=True)
    )

    # ── Build clean output ─────────────────────────────────────────────────────
    output_cols = [gene_symbol_col, probe_id_col]
    if use_desc:
        output_cols.append(gene_desc_col)
    output_cols += sample_cols

    result = (
        best_probes[output_cols]
        .rename(columns={probe_id_col: 'representative_probe_id'})
        .reset_index(drop=True)
    )

    return result


def align_column_names(source_cols: list[str], target_cols: list[str]) -> list[str]:
    """
    Rename columns in source_cols to match names found in target_cols.
    
    Matching logic: checks if any target name is a substring of a source name
    (case-insensitive). If a match is found, the source name is replaced with
    the target name. Unmatched names are left unchanged.

    Parameters
    ----------
    source_cols : list[str]
        Column names to be renamed, e.g. ["CharacteristicPBRM1", "CharacteristicBAP1"]
    target_cols : list[str]
        Reference names to match against, e.g. ["PBRM1", "BAP1"]

    Returns
    -------
    list[str]
        Modified list with matched names replaced, unmatched names left as-is.

    Example
    -------
    source = ["CharacteristicPBRM1", "CharacteristicBAP1", "SampleID"]
    target = ["PBRM1", "BAP1", "VHL"]

    result = align_column_names(source, target)
    # → ["PBRM1", "BAP1", "SampleID"]
    #     ✓ matched   ✓ matched   ✓ left unchanged (not in target)
    """

    # Build a lookup: lowercase target name → original target name
    target_lookup = {t.lower(): t for t in target_cols}

    result = []
    for col in source_cols:
        col_lower = col.lower()
        matched   = None

        # Check if any target name appears inside this source column name
        for target_lower, target_original in target_lookup.items():
            if target_lower in col_lower:
                matched = target_original
                break

        result.append(matched if matched else col)

    return result
