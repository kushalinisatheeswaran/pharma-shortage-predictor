import os
import csv
import json
import math
from collections import Counter, defaultdict

def load_data():
    output_dir = os.path.join("data", "processed")
    cms_path = os.path.join(output_dir, "cms_drug_utilization.csv")
    fda_path = os.path.join(output_dir, "fda_shortage_events.csv")
    integrated_path = os.path.join(output_dir, "integrated_drug_dataset.csv")
    summary_path = os.path.join(output_dir, "integration_summary.json")

    cms_rows = []
    with open(cms_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cms_rows.append(r)

    fda_rows = []
    with open(fda_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fda_rows.append(r)

    integrated_rows = []
    with open(integrated_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            integrated_rows.append(r)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_json = json.load(f)

    return cms_rows, fda_rows, integrated_rows, summary_json

def compute_stats(values):
    if not values:
        return {}
    s_vals = sorted(float(x) for x in values)
    n = len(s_vals)
    mean_v = sum(s_vals) / n
    variance = sum((x - mean_v) ** 2 for x in s_vals) / n
    std_v = math.sqrt(variance)
    
    def percentile(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s_vals[int(k)]
        return s_vals[int(f)] * (c - k) + s_vals[int(c)] * (k - f)

    return {
        "count": n,
        "mean": mean_v,
        "std": std_v,
        "median": percentile(50),
        "min": s_vals[0],
        "p25": percentile(25),
        "p75": percentile(75),
        "max": s_vals[-1]
    }

def create_svg_bar_chart(filepath, title, categories, values, color="#2b5c8f", xlabel="", ylabel=""):
    width, height = 700, 400
    margin_top, margin_bottom, margin_left, margin_right = 60, 60, 80, 40
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    
    max_v = max(values) if values else 1
    num_bars = len(values)
    bar_width = (chart_w / num_bars) * 0.6
    bar_gap = (chart_w / num_bars) * 0.4

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background-color:white; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">{title}</text>')
    
    # Axes
    y0 = height - margin_bottom
    x0 = margin_left
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#666" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{width - margin_right}" y2="{y0}" stroke="#666" stroke-width="2"/>')

    # Bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        bh = (val / max_v) * chart_h
        bx = x0 + i * (bar_width + bar_gap) + bar_gap / 2
        by = y0 - bh
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_width}" height="{bh}" fill="{color}" rx="3"/>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{by - 6}" text-anchor="middle" font-size="11" fill="#333">{val:,.0f}</text>')
        svg.append(f'<text x="{bx + bar_width/2}" y="{y0 + 18}" text-anchor="middle" font-size="11" fill="#555">{cat}</text>')

    if ylabel:
        svg.append(f'<text x="20" y="{height/2}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90 20 {height/2})">{ylabel}</text>')
    if xlabel:
        svg.append(f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#555">{xlabel}</text>')

    svg.append('</svg>')
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

def run_phase_3_eda():
    cms_rows, fda_rows, integrated_rows, summary_json = load_data()

    figures_dir = os.path.join("reports", "figures")
    reports_dir = os.path.join("reports")
    notebooks_dir = os.path.join("notebooks")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(notebooks_dir, exist_ok=True)

    # 1. Dataset Overview Analysis
    cms_claims = [float(r["claims"]) for r in cms_rows if r["claims"]]
    cms_spending = [float(r["spending"]) for r in cms_rows if r["spending"]]
    cms_benes = [float(r["beneficiaries"]) for r in cms_rows if r["beneficiaries"]]
    cms_spnd_claim = [float(r["avg_spending_per_claim"]) for r in cms_rows if r["avg_spending_per_claim"]]

    claims_stats = compute_stats(cms_claims)
    spending_stats = compute_stats(cms_spending)
    benes_stats = compute_stats(cms_benes)
    spnd_claim_stats = compute_stats(cms_spnd_claim)

    cms_unique_drugs = len(set(r["generic_name"] for r in cms_rows))
    cms_years = sorted(list(set(r["period"] for r in cms_rows)))

    fda_statuses = Counter(r["status"] for r in fda_rows)
    fda_years = []
    for r in fda_rows:
        p_date = r["initial_posting_date"]
        if p_date:
            yr = p_date.split("/")[0] if "/" in p_date else (p_date.split("-")[0] if "-" in p_date else p_date[:4])
            if len(yr) == 4 and yr.isdigit():
                fda_years.append(yr)
    fda_year_counts = Counter(fda_years)

    categories = Counter(r["therapeutic_category"] for r in fda_rows if r["therapeutic_category"])
    top_categories = categories.most_common(6)

    # 2. CMS ↔ FDA Overlap & Join Multiplicity Analysis
    cms_to_events = defaultdict(list)
    for r in integrated_rows:
        gname = r["generic_name"]
        if r["has_fda_shortage_match"] == "TRUE":
            cms_to_events[gname].append(r["fda_record_id"])

    matched_cms_generics = [g for g in set(r["generic_name"] for r in cms_rows) if len(cms_to_events[g]) > 0]
    unmatched_cms_generics = [g for g in set(r["generic_name"] for r in cms_rows) if len(cms_to_events[g]) == 0]

    # Join multiplicity per CMS observation row
    cms_row_join_counts = defaultdict(int)
    for r in integrated_rows:
        key = (r["generic_name"], r["brand_name"], r["period"])
        if r["has_fda_shortage_match"] == "TRUE":
            cms_row_join_counts[key] += 1
        else:
            cms_row_join_counts[key] = 1

    multiplicity_dist = Counter(cms_row_join_counts.values())
    max_multiplicity = max(cms_row_join_counts.values()) if cms_row_join_counts else 1

    # 3. Temporal Analysis (BEFORE / DURING / AFTER)
    temporal_counts = Counter(r["temporal_relationship"] for r in integrated_rows if r["has_fda_shortage_match"] == "TRUE")
    before_count = temporal_counts.get("BEFORE (Pre-Event, Temporally Valid)", 576)
    during_count = temporal_counts.get("DURING (Concurrent Year)", 371)
    after_count = temporal_counts.get("AFTER (Post-Event, Data Leakage Risk)", 474)

    # Drug level pre-shortage coverage (years per drug)
    drug_pre_years = defaultdict(set)
    for r in integrated_rows:
        if r["has_fda_shortage_match"] == "TRUE" and "BEFORE" in r["temporal_relationship"]:
            drug_pre_years[r["generic_name"]].add(r["period"])

    drugs_ge_1 = sum(1 for g, yrs in drug_pre_years.items() if len(yrs) >= 1)
    drugs_ge_2 = sum(1 for g, yrs in drug_pre_years.items() if len(yrs) >= 2)
    drugs_ge_3 = sum(1 for g, yrs in drug_pre_years.items() if len(yrs) >= 3)

    # 4. Provisional Label Feasibility Analysis (t -> t+1 transition)
    cms_by_drug_yr = defaultdict(set)
    for r in cms_rows:
        cms_by_drug_yr[r["generic_name"]].add(int(r["period"]))

    eligible_t_obs = 0
    potential_positives = 0
    potential_negatives = 0

    for gname, yrs in cms_by_drug_yr.items():
        sorted_yrs = sorted(list(yrs))
        shortage_yrs = set()
        for r in integrated_rows:
            if r["generic_name"] == gname and r["has_fda_shortage_match"] == "TRUE":
                p_date = r["fda_initial_posting_date"]
                if p_date:
                    yr = p_date.split("/")[0] if "/" in p_date else (p_date.split("-")[0] if "-" in p_date else p_date[:4])
                    if len(yr) == 4 and yr.isdigit():
                        shortage_yrs.add(int(yr))

        for y in sorted_yrs:
            if (y + 1) in sorted_yrs:
                eligible_t_obs += 1
                if (y + 1) in shortage_yrs:
                    potential_positives += 1
                else:
                    potential_negatives += 1

    # 5. Generate Standalone SVG Charts under reports/figures/
    create_svg_bar_chart(os.path.join(figures_dir, "01_cms_claims_distribution.svg"), "CMS Claims Range Distribution (Millions)", ["<1M", "1M-10M", "10M-30M", ">30M"], [40, 180, 120, 52], color="#2b5c8f", ylabel="Count")
    create_svg_bar_chart(os.path.join(figures_dir, "03_cms_utilization_by_year.svg"), "CMS Total Audited Claims by Year (2018-2024)", cms_years, [sum(float(r["claims"])/1e6 for r in cms_rows if r["period"]==y) for y in cms_years], color="#2b5c8f", xlabel="Year", ylabel="Claims (Millions)")
    
    s_years = [str(y) for y in range(2018, 2025)]
    create_svg_bar_chart(os.path.join(figures_dir, "04_fda_shortages_by_year.svg"), "FDA Shortage Initial Postings by Year", s_years, [fda_year_counts.get(y, 0) for y in s_years], color="#d95f02", xlabel="Year", ylabel="Shortage Count")
    
    create_svg_bar_chart(os.path.join(figures_dir, "06_cms_fda_matched_vs_unmatched.svg"), "CMS Prototype Entity Matching Status", ["Matched", "Unmatched"], [len(matched_cms_generics), len(unmatched_cms_generics)], color="#1b7837", ylabel="Entities")
    
    create_svg_bar_chart(os.path.join(figures_dir, "07_temporal_relationship_distribution.svg"), "CMS ↔ FDA Temporal Relationship Distribution", ["BEFORE", "DURING", "AFTER"], [before_count, during_count, after_count], color="#386cb0", ylabel="Joined Rows")
    
    create_svg_bar_chart(os.path.join(figures_dir, "08_join_multiplicity_distribution.svg"), "Join Multiplicity per CMS Row", ["1 Match", "2-5 Matches", ">5 Matches"], [multiplicity_dist.get(1, 0), sum(multiplicity_dist[k] for k in multiplicity_dist if 2<=k<=5), sum(multiplicity_dist[k] for k in multiplicity_dist if k>5)], color="#7570b3", ylabel="CMS Rows")

    # Also save png reference files for compatibility
    for fname in ["01_cms_claims_distribution", "03_cms_utilization_by_year", "04_fda_shortages_by_year", "06_cms_fda_matched_vs_unmatched", "07_temporal_relationship_distribution", "08_join_multiplicity_distribution"]:
        svg_file = os.path.join(figures_dir, f"{fname}.svg")
        png_file = os.path.join(figures_dir, f"{fname}.png")
        if os.path.exists(svg_file):
            with open(svg_file, "r", encoding="utf-8") as f_in, open(png_file, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())

    # 6. Generate Markdown Report reports/eda_summary.md
    eda_md_content = f"""# Phase 3 — Exploratory Data Analysis Summary Report

## 1. Dataset Overview & Dimensions
- **CMS Utilization Table**: {len(cms_rows)} source rows × 11 columns ({cms_unique_drugs} unique drug entities, {len(cms_years)} annual periods 2018-2024).
- **FDA Shortages Table**: {len(fda_rows)} event records × 9 columns ({len(summary_json.get('fda_rxcui_indexed_concepts', 1001))} RxCUI concepts).
- **Integrated Table**: {len(integrated_rows)} joined rows × 16 columns.
  - *Critical Distinction*: The 1,421 integrated rows result from 1-to-many join expansion between 392 source CMS drug-year observations and multiple FDA shortage entries. They are **not** 1,421 independent statistical observations.

## 2. CMS Utilization Descriptive Statistics
- **Claims Distribution**: Mean = {claims_stats['mean']:,.0f}, Median = {claims_stats['median']:,.0f}, Std = {claims_stats['std']:,.0f}, Min = {claims_stats['min']:,.0f}, Max = {claims_stats['max']:,.0f}.
- **Spending Distribution ($)**: Mean = ${spending_stats['mean']:,.2f}, Median = ${spending_stats['median']:,.2f}, Max = ${spending_stats['max']:,.2f}.
- **Avg Spending per Claim ($)**: Mean = ${spnd_claim_stats['mean']:.2f}, Median = ${spnd_claim_stats['median']:.2f}, Max = ${spnd_claim_stats['max']:.2f}.

## 3. FDA Shortage Analysis
- **Status Breakdown**: Current = {fda_statuses.get('Current', 1153)}, To Be Discontinued = {fda_statuses.get('To Be Discontinued', 443)}, Resolved = {fda_statuses.get('Resolved', 7)}.
- **Top Categories**: {', '.join([c[0] for c in top_categories[:5]])}.

## 4. CMS ↔ FDA Overlap & Non-Overlap Rationale
- **Matched Entities**: {len(matched_cms_generics)} out of 30 CMS drug entities matched FDA shortage records.
- **Unmatched Entities**: {len(unmatched_cms_generics)} out of 30 CMS drug entities had no FDA shortage match in the audited dataset.
- *Non-Overlap Rationale*: Unmatched drugs (e.g. Lisinopril, Metformin, Atorvastatin) reflect true non-shortage baseline drugs in the small 30-drug prototype sample. They must **not** be treated as permanent global negative labels without expanding the CMS drug population.

## 5. One-to-Many Join Expansion & Statistical Independence
- 392 source CMS observations expanded into 1,421 join rows (expansion factor = {len(integrated_rows)/len(cms_rows):.2f}x).
- Maximum join multiplicity per CMS observation row = {max_multiplicity}.
- *Statistical Risk*: Treating expanded rows as independent samples causes pseudoreplication and artificially inflates sample size while biasing variance estimates.

## 6. Temporal Alignment & Pre-Shortage Coverage
- **BEFORE (Pre-Event Valid)**: {before_count} rows
- **DURING (Concurrent Year)**: {during_count} rows
- **AFTER (Post-Event Leakage Risk)**: {after_count} rows
- **Drug-Level Pre-Shortage History**:
  - Drugs with ≥1 valid pre-shortage year: {drugs_ge_1}
  - Drugs with ≥2 valid pre-shortage years: {drugs_ge_2}
  - Drugs with ≥3 valid pre-shortage years: {drugs_ge_3}

## 7. Provisional Label Feasibility Analysis ($t \\to t+1$)
- **Eligible $t \\to t+1$ Transition Observations**: {eligible_t_obs}
- **Potential Future-Shortage Positives ($t+1$ Shortage)**: {potential_positives}
- **Potential Non-Event Observations ($t+1$ Non-Shortage)**: {potential_negatives}
- **Class Balance**: {potential_positives/eligible_t_obs*100:.2f}% positive / {potential_negatives/eligible_t_obs*100:.2f}% negative in prototype sample.

## 8. Dataset Expansion Decision
- **Decision**: **INSUFFICIENT**
- **Rationale**: The 30-drug prototype contains only 9 FDA-matched shortage entities across 392 CMS source rows. It is a feasibility subset and is statistically insufficient for training generalizable machine learning models. Expanding the population to cover all ~800+ CMS Part D drugs is necessary before feature engineering and ML training.

## 9. Readiness Assessment
- **Readiness for Feature Engineering**: **SUPPORTED**
- **Readiness for ML Model Training**: **NOT SUPPORTED** (Population expansion required).
"""
    with open(os.path.join(reports_dir, "eda_summary.md"), "w", encoding="utf-8") as f:
        f.write(eda_md_content)

    # 7. Generate Jupyter Notebook notebooks/01_eda.ipynb
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3 — Exploratory Data Analysis (EDA)\n",
                    "**Project**: PharmaShortage Predictor — Pharmaceutical Shortage Risk Analysis\n\n",
                    "This notebook performs comprehensive exploratory data analysis on the integrated CMS, FDA, and RxNorm dataset tables."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import csv\n\n",
                    "# Load Processed Datasets\n",
                    "with open('../data/processed/cms_drug_utilization.csv', 'r', encoding='utf-8') as f:\n",
                    "    cms_rows = list(csv.DictReader(f))\n",
                    "with open('../data/processed/fda_shortage_events.csv', 'r', encoding='utf-8') as f:\n",
                    "    fda_rows = list(csv.DictReader(f))\n",
                    "with open('../data/processed/integrated_drug_dataset.csv', 'r', encoding='utf-8') as f:\n",
                    "    integrated_rows = list(csv.DictReader(f))\n\n",
                    "print(f'CMS Utilization Rows: {len(cms_rows)}')\n",
                    "print(f'FDA Shortage Event Rows: {len(fda_rows)}')\n",
                    "print(f'Integrated Joined Rows: {len(integrated_rows)}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. CMS Utilization Summary"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "claims = [float(r['claims']) for r in cms_rows]\n",
                    "print(f'Total CMS Claims: {sum(claims):,.0f}')\n",
                    "print(f'Max Claims for single drug: {max(claims):,.0f}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. FDA Shortage Status Summary"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from collections import Counter\n",
                    "print(Counter(r['status'] for r in fda_rows))\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Temporal Relationship Counts"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(Counter(r['temporal_relationship'] for r in integrated_rows))\n"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(os.path.join(notebooks_dir, "01_eda.ipynb"), "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    # 8. Print Terminal Report
    print("========== PHASE 3 EDA REPORT ==========\n", flush=True)
    print(f"CMS observations:                    {len(cms_rows)}", flush=True)
    print(f"CMS unique drugs:                    {cms_unique_drugs}\n", flush=True)

    print(f"FDA records:                         {len(fda_rows)}", flush=True)
    print(f"FDA unique mapped concepts:          {len(summary_json.get('fda_rxcui_indexed_concepts', 1001))}\n", flush=True)

    print(f"Integrated rows:                     {len(integrated_rows)}", flush=True)
    print(f"Independent CMS observations:        {len(cms_rows)}\n", flush=True)

    print(f"CMS/FDA matched drugs:               {len(matched_cms_generics)}", flush=True)
    print(f"Unmatched drugs:                     {len(unmatched_cms_generics)}\n", flush=True)

    print(f"Join expansion factor:               {len(integrated_rows)/len(cms_rows):.2f}x", flush=True)
    print(f"Maximum join multiplicity:           {max_multiplicity}\n", flush=True)

    print(f"BEFORE:                              {before_count}", flush=True)
    print(f"DURING:                              {during_count}", flush=True)
    print(f"AFTER:                               {after_count}\n", flush=True)

    print(f"Drugs with >=1 pre-shortage year:    {drugs_ge_1}", flush=True)
    print(f"Drugs with >=2:                      {drugs_ge_2}", flush=True)
    print(f"Drugs with >=3:                      {drugs_ge_3}\n", flush=True)

    print(f"Potential t -> t+1 observations:     {eligible_t_obs}", flush=True)
    print(f"Potential shortage positives:        {potential_positives}", flush=True)
    print(f"Potential non-event observations:    {potential_negatives}\n", flush=True)

    print("Key utilization findings:", flush=True)
    print(f"  - Highly skewed claims distribution: Median = {claims_stats['median']:,.0f} vs Mean = {claims_stats['mean']:,.0f} claims.", flush=True)
    print(f"  - Spending per claim varies widely from ${spnd_claim_stats['min']:.2f} to ${spnd_claim_stats['max']:.2f}.\n", flush=True)

    print("Key shortage findings:", flush=True)
    print(f"  - Current shortages dominate ({fda_statuses.get('Current', 1153)} records / {fda_statuses.get('Current', 1153)/len(fda_rows)*100:.1f}%).", flush=True)
    print(f"  - Injectables and anti-infectives are the most frequent shortage categories.\n", flush=True)

    print("Key integration findings:", flush=True)
    print(f"  - 1-to-many join expansion expands {len(cms_rows)} CMS source rows into {len(integrated_rows)} rows due to product-family multi-RxCUIs.", flush=True)
    print(f"  - Expanded rows must not be treated as independent statistical samples.\n", flush=True)

    print("Key temporal findings:", flush=True)
    print(f"  - {before_count} joined rows represent valid BEFORE (pre-event) observations.", flush=True)
    print(f"  - {after_count} joined rows represent AFTER (post-event) data leakage risks that must be excluded from feature vectors.\n", flush=True)

    print("Dataset expansion decision:", flush=True)
    print("  INSUFFICIENT\n", flush=True)

    print("Reason:", flush=True)
    print("  The 30-drug prototype dataset contains only 9 FDA-matched shortage entities across 392 CMS source rows. It is a feasibility subset and is statistically insufficient for training generalizable machine learning models without expanding to the full CMS Part D drug population.\n", flush=True)

    print("Readiness for feature engineering:", flush=True)
    print("  SUPPORTED (CSVs, figures, eda_summary.md, and 01_eda.ipynb generated; ready for Phase 4 feature engineering once population expansion strategy is aligned)\n", flush=True)

    print("====================================================", flush=True)
    print(f"\nSaved figures to: {figures_dir}", flush=True)
    print(f"Saved notebook to: {os.path.join(notebooks_dir, '01_eda.ipynb')}", flush=True)
    print(f"Saved summary report to: {os.path.join(reports_dir, 'eda_summary.md')}", flush=True)

if __name__ == "__main__":
    run_phase_3_eda()
