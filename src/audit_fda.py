import os
import json
import time
from collections import Counter
import requests

URL = "https://api.fda.gov/drug/shortages.json"

def fetch_with_retry(url, params, max_retries=5, backoff_factor=1.5, timeout=30):
    """
    Fetch data from openFDA API with limited exponential backoff retries.
    Handles temporary network interruptions (e.g. WinError 10054 / ConnectionResetError)
    without disabling SSL verification (verify=False is NEVER used).
    """
    errors_log = []
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json(), errors_log
        except requests.exceptions.RequestException as e:
            msg = f"Attempt {attempt}/{max_retries} error: {e}"
            errors_log.append(msg)
            if attempt == max_retries:
                raise
            time.sleep(backoff_factor ** attempt)

def audit_fda_dataset():
    all_errors = []
    
    # Step 1: Inspect API metadata to determine total available records
    # Pagination: openFDA supports limit & skip parameters. We query initial meta first.
    print("Inspecting openFDA API metadata...")
    init_params = {"limit": 1, "skip": 0}
    meta_json, errs = fetch_with_retry(URL, init_params)
    all_errors.extend(errs)
    
    total_available = meta_json.get("meta", {}).get("results", {}).get("total", 0)
    print(f"Total FDA drug shortage records reported by API metadata: {total_available}")

    # Step 2 & 3: Retrieve all records using safe, respectful pagination
    records = []
    limit = 100  # Safe openFDA batch limit per request
    skip = 0

    print(f"Fetching records in batches of {limit}...")
    while skip < total_available:
        params = {"limit": limit, "skip": skip}
        batch_json, errs = fetch_with_retry(URL, params)
        all_errors.extend(errs)
        
        batch_results = batch_json.get("results", [])
        if not batch_results:
            break
            
        records.extend(batch_results)
        skip += len(batch_results)
        
        # Small delay between paginated requests to respect openFDA API rate guidelines
        time.sleep(0.3)

    total_retrieved = len(records)
    print(f"Successfully retrieved {total_retrieved} total records.\n")

    # Step 5 & 6: Feature & Identifier Coverage Analysis
    # Identifier coverage tracking
    ndc_present = 0
    rxcui_present = 0

    # Additional field coverage tracking
    date_present = 0
    update_date_present = 0
    therapeutic_category_present = 0
    generic_name_present = 0
    dosage_form_present = 0

    # RxCUI cardinality tracking
    # Why multiple RxCUIs matter: One FDA shortage entry can map to multiple RxCUIs
    # because a product-level shortage notice may cover multiple strengths, dosage forms,
    # or clinical drug formulations. This affects entity resolution precision in downstream mapping.
    rxcui_zero = 0
    rxcui_one = 0
    rxcui_multiple = 0

    statuses = Counter()
    update_types = Counter()

    for r in records:
        # Check package NDC
        if r.get("package_ndc"):
            ndc_present += 1

        # Check RxCUI inside openfda object
        openfda = r.get("openfda", {})
        rxcuis = openfda.get("rxcui", [])
        
        if isinstance(rxcuis, str):
            rxcuis = [rxcuis]
            
        if rxcuis:
            rxcui_present += 1
            if len(rxcuis) == 1:
                rxcui_one += 1
            else:
                rxcui_multiple += 1
        else:
            rxcui_zero += 1

        # Additional fields
        if r.get("initial_posting_date"):
            date_present += 1

        if r.get("update_date") or r.get("last_updated"):
            update_date_present += 1

        if r.get("therapeutic_category"):
            therapeutic_category_present += 1

        if r.get("generic_name") or openfda.get("generic_name"):
            generic_name_present += 1

        if r.get("dosage_form") or openfda.get("dosage_form"):
            dosage_form_present += 1

        # Status & Update Type distributions (descriptive FDA status values, NOT ML labels)
        status = r.get("status", "Missing")
        statuses[status] += 1

        update_type = r.get("update_type", "Missing")
        update_types[update_type] += 1

    # Helper function for percentage formatting
    def pct(count, total):
        return (count / total * 100) if total > 0 else 0.0

    # Step 8: Terminal Display
    print("========== FDA DRUG SHORTAGE DATA AUDIT ==========\n")
    print(f"Total available records (meta):        {total_available}")
    print(f"Total records retrieved:               {total_retrieved}")

    print("\n--- Identifier Coverage ---")
    print(f"Package NDC present:                  {ndc_present} / {total_retrieved} ({pct(ndc_present, total_retrieved):.2f}%)")
    print(f"Package NDC missing:                  {total_retrieved - ndc_present}")

    print(f"RxCUI present:                        {rxcui_present} / {total_retrieved} ({pct(rxcui_present, total_retrieved):.2f}%)")
    print(f"RxCUI missing:                        {total_retrieved - rxcui_present}")

    print("\n--- RxCUI Cardinality Breakdown ---")
    print(f"No RxCUI (0):                         {rxcui_zero} ({pct(rxcui_zero, total_retrieved):.2f}%)")
    print(f"Exactly one RxCUI (1):                {rxcui_one} ({pct(rxcui_one, total_retrieved):.2f}%)")
    print(f"Multiple RxCUIs (>1):                 {rxcui_multiple} ({pct(rxcui_multiple, total_retrieved):.2f}%)")

    print("\n--- Additional Field Coverage ---")
    print(f"Initial posting date present:         {date_present} ({pct(date_present, total_retrieved):.2f}%)")
    print(f"Update date present:                  {update_date_present} ({pct(update_date_present, total_retrieved):.2f}%)")
    print(f"Therapeutic category present:         {therapeutic_category_present} ({pct(therapeutic_category_present, total_retrieved):.2f}%)")
    print(f"Generic name present:                 {generic_name_present} ({pct(generic_name_present, total_retrieved):.2f}%)")
    print(f"Dosage form present:                  {dosage_form_present} ({pct(dosage_form_present, total_retrieved):.2f}%)")

    print("\n--- Status Distribution ---")
    for status, count in statuses.most_common():
        print(f"{status:35} {count:5} ({pct(count, total_retrieved):.2f}%)")

    print("\n--- Update Type Distribution ---")
    for utype, count in update_types.most_common():
        print(f"{utype:35} {count:5} ({pct(count, total_retrieved):.2f}%)")

    if all_errors:
        print("\n--- Network / API Warnings Encountered ---")
        for err in all_errors:
            print(f"- {err}")
    else:
        print("\n--- Network / API Warnings Encountered ---")
        print("None. All requests completed cleanly.")

    print("\n===================================================")

    # Step 9: Save machine-readable audit summary to data/processed/fda_audit_summary.json
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    summary_filepath = os.path.join(output_dir, "fda_audit_summary.json")

    summary_data = {
        "metadata_total_records": total_available,
        "total_records_retrieved": total_retrieved,
        "identifier_coverage": {
            "package_ndc": {
                "present": ndc_present,
                "missing": total_retrieved - ndc_present,
                "coverage_pct": round(pct(ndc_present, total_retrieved), 2)
            },
            "rxcui": {
                "present": rxcui_present,
                "missing": total_retrieved - rxcui_present,
                "coverage_pct": round(pct(rxcui_present, total_retrieved), 2)
            }
        },
        "rxcui_cardinality": {
            "zero": rxcui_zero,
            "one": rxcui_one,
            "multiple": rxcui_multiple
        },
        "field_coverage": {
            "initial_posting_date": {
                "present": date_present,
                "coverage_pct": round(pct(date_present, total_retrieved), 2)
            },
            "update_date": {
                "present": update_date_present,
                "coverage_pct": round(pct(update_date_present, total_retrieved), 2)
            },
            "therapeutic_category": {
                "present": therapeutic_category_present,
                "coverage_pct": round(pct(therapeutic_category_present, total_retrieved), 2)
            },
            "generic_name": {
                "present": generic_name_present,
                "coverage_pct": round(pct(generic_name_present, total_retrieved), 2)
            },
            "dosage_form": {
                "present": dosage_form_present,
                "coverage_pct": round(pct(dosage_form_present, total_retrieved), 2)
            }
        },
        "status_distribution": dict(statuses),
        "update_type_distribution": dict(update_types),
        "network_errors_encountered": all_errors
    }

    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved machine-readable summary to: {summary_filepath}")

if __name__ == "__main__":
    audit_fda_dataset()