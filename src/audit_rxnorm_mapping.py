import os
import json
import time
from collections import Counter
import requests

FDA_URL = "https://api.fda.gov/drug/shortages.json"
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

# Global Caches for API Lookups
NDC_CACHE = {}
RXCUI_PROPS_CACHE = {}

def fetch_with_retry(url, params=None, max_retries=3, backoff_factor=1.5, timeout=10):
    """
    Helper function to query APIs with exponential backoff retries.
    Handles temporary WinError 10054 / ConnectionResetError without disabling SSL (verify=False is NEVER used).
    """
    errors_log = []
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json(), errors_log
        except requests.exceptions.RequestException as e:
            msg = f"Attempt {attempt}/{max_retries} error for URL {url}: {e}"
            errors_log.append(msg)
            if attempt == max_retries:
                return None, errors_log
            time.sleep(backoff_factor ** attempt)
    return None, errors_log

def query_rxnorm_ndc(ndc):
    """
    Queries official RxNorm findRxcuiById endpoint for NDC -> RxCUI mapping.
    Endpoint: GET /REST/rxcui.json?idtype=NDC&id=<NDC>
    Uses caching to avoid duplicate requests for identical NDCs.
    """
    clean_ndc = str(ndc).strip()
    if clean_ndc in NDC_CACHE:
        return NDC_CACHE[clean_ndc]

    url = f"{RXNORM_BASE_URL}/rxcui.json"
    params = {"idtype": "NDC", "id": clean_ndc}
    data, errs = fetch_with_retry(url, params=params)
    
    if data is None:
        res = (None, errs)
    else:
        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])
        if rxnorm_ids is None:
            rxnorm_ids = []
        res = (rxnorm_ids, errs)
        
    NDC_CACHE[clean_ndc] = res
    return res

def query_rxcui_properties(rxcui):
    """
    Retrieves RxNorm concept properties (name, term type TTY) for a given RxCUI.
    Endpoint: GET /REST/rxcui/<rxcui>/properties.json
    Uses caching to avoid duplicate property queries.
    """
    str_rxcui = str(rxcui).strip()
    if str_rxcui in RXCUI_PROPS_CACHE:
        return RXCUI_PROPS_CACHE[str_rxcui]

    url = f"{RXNORM_BASE_URL}/rxcui/{str_rxcui}/properties.json"
    data, errs = fetch_with_retry(url)
    
    if data is None:
        res = ({"rxcui": str_rxcui, "name": "N/A", "tty": "N/A"}, errs)
    else:
        props = data.get("properties", {})
        res = ({
            "rxcui": str_rxcui,
            "name": props.get("name", "N/A"),
            "tty": props.get("tty", "N/A")
        }, errs)

    RXCUI_PROPS_CACHE[str_rxcui] = res
    return res

def audit_rxnorm_mapping():
    all_errors = []
    
    # Task 2: Retrieve FDA Shortage Records safely
    print("Fetching FDA shortage records from openFDA...", flush=True)
    init_data, errs = fetch_with_retry(FDA_URL, params={"limit": 1, "skip": 0})
    all_errors.extend(errs)
    
    total_available = init_data.get("meta", {}).get("results", {}).get("total", 0) if init_data else 0
    
    fda_records = []
    limit = 100
    skip = 0
    
    while skip < total_available:
        batch_data, errs = fetch_with_retry(FDA_URL, params={"limit": limit, "skip": skip})
        all_errors.extend(errs)
        if not batch_data:
            break
        results = batch_data.get("results", [])
        if not results:
            break
        fda_records.extend(results)
        skip += len(results)
        time.sleep(0.2)

    print(f"Retrieved {len(fda_records)} total FDA shortage records.", flush=True)
    
    # Task 3: Identify missing RxCUI records
    missing_rxcui_records = []
    embedded_rxcui_records = []
    
    for record in fda_records:
        openfda = record.get("openfda", {})
        rxcuis = openfda.get("rxcui", [])
        if isinstance(rxcuis, str):
            rxcuis = [rxcuis]
            
        if not rxcuis:
            missing_rxcui_records.append(record)
        else:
            embedded_rxcui_records.append(record)
            
    missing_count = len(missing_rxcui_records)
    print(f"FDA records missing embedded openfda.rxcui: {missing_count}", flush=True)
    
    # Task 4 & 6: NDC -> RxCUI Recovery for records missing embedded RxCUI
    print("\nAuditing NDC -> RxCUI recovery for missing-RxCUI records...", flush=True)
    
    recovery_outcomes = {
        "mapped_one": 0,
        "mapped_multiple": 0,
        "unmapped": 0,
        "api_error": 0
    }
    
    cardinality_counts = {
        "zero": 0,
        "one": 0,
        "multiple": 0
    }
    
    recovery_details = []
    
    for idx, rec in enumerate(missing_rxcui_records):
        if (idx + 1) % 25 == 0 or idx == 0 or (idx + 1) == missing_count:
            print(f"Auditing missing-RxCUI record {idx+1}/{missing_count}...", flush=True)

        pkg_ndc = rec.get("package_ndc", [])
        if isinstance(pkg_ndc, str):
            ndc_list = [pkg_ndc]
        else:
            ndc_list = list(pkg_ndc) if pkg_ndc else []
            
        recovered_rxcuis = set()
        had_api_error = False
        
        for ndc in ndc_list:
            res, errs = query_rxnorm_ndc(ndc)
            all_errors.extend(errs)
            time.sleep(0.02)  # Small delay between distinct API calls
            
            if res is None:
                had_api_error = True
            else:
                recovered_rxcuis.update(res)
                
        rx_count = len(recovered_rxcuis)
        
        if had_api_error and rx_count == 0:
            outcome = "api_error"
        elif rx_count == 0:
            outcome = "unmapped"
            cardinality_counts["zero"] += 1
        elif rx_count == 1:
            outcome = "mapped_one"
            cardinality_counts["one"] += 1
        else:
            outcome = "mapped_multiple"
            cardinality_counts["multiple"] += 1
            
        recovery_outcomes[outcome] += 1
        
        recovery_details.append({
            "generic_name": rec.get("generic_name", "Unknown"),
            "package_ndc": ndc_list,
            "status": rec.get("status", "Unknown"),
            "returned_rxcuis": sorted(list(recovered_rxcuis)),
            "rxcui_count": rx_count,
            "outcome": outcome
        })

    mapped_through_ndc = recovery_outcomes["mapped_one"] + recovery_outcomes["mapped_multiple"]
    still_unmapped = recovery_outcomes["unmapped"]
    api_failures = recovery_outcomes["api_error"]
    recovery_rate_pct = (mapped_through_ndc / missing_count * 100) if missing_count > 0 else 0.0

    # Task 7: Validate existing embedded RxCUI mappings (deterministic sample of 20)
    print("\nValidating existing embedded RxCUI mappings (deterministic sample of 20)...", flush=True)
    sorted_embedded = sorted(
        embedded_rxcui_records,
        key=lambda r: (r.get("generic_name", ""), str(r.get("package_ndc", "")))
    )
    consistency_sample = sorted_embedded[:20]
    
    consistency_results = {
        "exact_set_match": 0,
        "overlap": 0,
        "no_overlap": 0,
        "RxNorm_unmapped": 0,
        "api_error": 0
    }
    
    consistency_sample_details = []
    
    for rec in consistency_sample:
        openfda = rec.get("openfda", {})
        embedded_rxcuis = set(openfda.get("rxcui", []))
        
        pkg_ndc = rec.get("package_ndc", [])
        if isinstance(pkg_ndc, str):
            ndc_list = [pkg_ndc]
        else:
            ndc_list = list(pkg_ndc) if pkg_ndc else []
            
        returned_rxcuis = set()
        had_api_error = False
        
        for ndc in ndc_list:
            res, errs = query_rxnorm_ndc(ndc)
            all_errors.extend(errs)
            time.sleep(0.05)
            
            if res is None:
                had_api_error = True
            else:
                returned_rxcuis.update(res)
                
        if had_api_error and len(returned_rxcuis) == 0:
            classification = "api_error"
        elif len(returned_rxcuis) == 0:
            classification = "RxNorm_unmapped"
        elif returned_rxcuis == embedded_rxcuis:
            classification = "exact_set_match"
        elif returned_rxcuis.intersection(embedded_rxcuis):
            classification = "overlap"
        else:
            classification = "no_overlap"
            
        consistency_results[classification] += 1
        
        consistency_sample_details.append({
            "generic_name": rec.get("generic_name", "Unknown"),
            "package_ndc": ndc_list,
            "embedded_rxcuis": sorted(list(embedded_rxcuis)),
            "returned_rxcuis": sorted(list(returned_rxcuis)),
            "classification": classification
        })

    # Task 8: Investigate Multi-RxCUI Structure (deterministic sample of 10)
    print("\nInvestigating multi-RxCUI records (deterministic sample of 10)...", flush=True)
    multi_rxcui_records = [
        r for r in embedded_rxcui_records
        if len(r.get("openfda", {}).get("rxcui", [])) > 1
    ]
    
    sorted_multi = sorted(
        multi_rxcui_records,
        key=lambda r: (r.get("generic_name", ""), str(r.get("package_ndc", "")))
    )
    multi_sample = sorted_multi[:10]
    
    multi_sample_details = []
    
    for rec in multi_sample:
        openfda = rec.get("openfda", {})
        embedded_rxcuis = openfda.get("rxcui", [])
        
        pkg_ndc = rec.get("package_ndc", [])
        if isinstance(pkg_ndc, str):
            ndc_list = [pkg_ndc]
        else:
            ndc_list = list(pkg_ndc) if pkg_ndc else []
            
        concepts = []
        for rxcui in embedded_rxcuis:
            prop, errs = query_rxcui_properties(rxcui)
            all_errors.extend(errs)
            time.sleep(0.05)
            concepts.append(prop)
            
        rec_detail = {
            "generic_name": rec.get("generic_name", "Unknown"),
            "package_ndc": ndc_list,
            "embedded_rxcui_count": len(embedded_rxcuis),
            "concepts": concepts
        }
        multi_sample_details.append(rec_detail)

    # Task 10: Terminal Output Display
    print("\n========== RXNORM ENTITY RESOLUTION AUDIT ==========\n", flush=True)
    print(f"FDA records examined:                 {len(fda_records)}", flush=True)
    print(f"FDA records missing embedded RxCUI:   {missing_count}", flush=True)

    print("\n--- NDC Recovery ---", flush=True)
    print(f"Mapped through NDC:                   {mapped_through_ndc}", flush=True)
    print(f"Still unmapped:                       {still_unmapped}", flush=True)
    print(f"API failures:                         {api_failures}", flush=True)
    print(f"Recovery rate:                        {recovery_rate_pct:.2f}%", flush=True)

    print("\n--- Mapping Cardinality ---", flush=True)
    print(f"Zero RxCUI:                           {cardinality_counts['zero']}", flush=True)
    print(f"Exactly one RxCUI:                    {cardinality_counts['one']}", flush=True)
    print(f"Multiple RxCUIs:                      {cardinality_counts['multiple']}", flush=True)

    print("\n--- Embedded Mapping Consistency Sample ---", flush=True)
    print(f"Exact set matches:                    {consistency_results['exact_set_match']}", flush=True)
    print(f"Overlap:                              {consistency_results['overlap']}", flush=True)
    print(f"No overlap:                           {consistency_results['no_overlap']}", flush=True)
    print(f"RxNorm unmapped:                      {consistency_results['RxNorm_unmapped']}", flush=True)
    print(f"API errors:                           {consistency_results['api_error']}", flush=True)

    print("\n--- Multi-RxCUI Diagnostic Sample ---", flush=True)
    for i, item in enumerate(multi_sample_details, 1):
        print(f"\nRecord #{i}: {item['generic_name']}", flush=True)
        print(f"  NDCs: {item['package_ndc']}", flush=True)
        print(f"  Embedded RxCUIs Count: {item['embedded_rxcui_count']}", flush=True)
        print("  Concepts:", flush=True)
        for c in item["concepts"]:
            print(f"    - RxCUI: {c['rxcui']:<10} TTY: {c['tty']:<6} Name: {c['name']}", flush=True)

    if all_errors:
        print("\n--- Network / API Warnings Encountered ---", flush=True)
        for err in all_errors:
            print(f"- {err}", flush=True)
    else:
        print("\n--- Network / API Warnings Encountered ---", flush=True)
        print("None. All requests completed cleanly.", flush=True)

    print("\n====================================================", flush=True)

    # Task 9: Save machine-readable audit summary JSON
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    summary_filepath = os.path.join(output_dir, "rxnorm_mapping_audit_summary.json")

    summary_data = {
        "fda_records_examined": len(fda_records),
        "fda_records_missing_embedded_rxcui": missing_count,
        "ndc_recovery": {
            "mapped_through_ndc": mapped_through_ndc,
            "still_unmapped": still_unmapped,
            "api_failures": api_failures,
            "recovery_rate_pct": round(recovery_rate_pct, 2)
        },
        "mapping_cardinality": cardinality_counts,
        "embedded_mapping_consistency_sample": consistency_results,
        "consistency_sample_details": consistency_sample_details,
        "multi_rxcui_diagnostic_sample": multi_sample_details,
        "total_network_errors_count": len(all_errors),
        "network_errors_encountered": all_errors
    }

    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved machine-readable audit summary to: {summary_filepath}", flush=True)

if __name__ == "__main__":
    audit_rxnorm_mapping()

