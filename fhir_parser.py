"""
FHIR IPS Bundle → flat dict translator for HealthClarify.

Converts a standards-compliant HL7 FHIR R4 IPS Bundle into the flat
dictionary format that app.py (and generate_sample_pdfs.py) expects.
This enables replacing the old custom flat JSON with real FHIR Bundles
while keeping all view/render code unchanged.

Usage:
    from fhir_parser import parse_fhir_bundle
    patient_data = parse_fhir_bundle(bundle)
"""

from __future__ import annotations

import json
import re
from datetime import date


def _calculate_age(birth_date_str: str, reference_date_str: str = "2026-03-18") -> int:
    """Return age in years at reference date."""
    try:
        birth = date.fromisoformat(birth_date_str)
        ref = date.fromisoformat(reference_date_str)
        age = ref.year - birth.year
        if (ref.month, ref.day) < (birth.month, birth.day):
            age -= 1
        return age
    except (ValueError, TypeError):
        return 0


def _format_lab_value(v) -> str:
    """Format a numeric lab value without spurious .0 decimals."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v) if v is not None else ""


def parse_fhir_bundle(bundle: dict) -> dict:
    """Convert a FHIR IPS Bundle (or the old flat format) into the flat
    dictionary that HealthClarify's view layer expects.

    If the incoming data does NOT look like a real FHIR Bundle (i.e. it is
    already flat), return it unchanged — this preserves backward compatibility
    with any other JSON files the user might upload.
    """
    # ── Guard: if it's already flat, pass through ─────────────────────────
    if bundle.get("resourceType") != "Bundle" or "entry" not in bundle:
        return bundle

    entries = bundle.get("entry", [])
    if not entries:
        return bundle

    # ── Index resources by type ───────────────────────────────────────────
    by_type: dict[str, list[dict]] = {}
    by_fullurl: dict[str, dict] = {}
    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")
        by_type.setdefault(rtype, []).append(resource)
        by_fullurl[entry.get("fullUrl", "")] = resource

    result: dict = {
        "id": bundle.get("id", ""),
        "timestamp": bundle.get("timestamp", "2026-03-18T14:30:00Z"),
        "resourceType": "Bundle",
        "type": bundle.get("type", "document"),
    }

    # ── Patient ───────────────────────────────────────────────────────────
    patients = by_type.get("Patient", [])
    patient_resource = patients[0] if patients else {}
    name_list = patient_resource.get("name", [{}])
    name = name_list[0] if name_list else {}
    given = " ".join(name.get("given", ["Unknown"]))
    family = name.get("family", "")
    full_name = f"{given} {family}".strip() or "Unknown Patient"
    age = _calculate_age(patient_resource.get("birthDate", ""))

    result["patient"] = {
        "name": full_name,
        "age": age,
        "gender": patient_resource.get("gender", "unknown"),
    }

    # ── Conditions ────────────────────────────────────────────────────────
    conditions = by_type.get("Condition", [])
    primary_dx = ""
    secondary: list[str] = []
    for c in conditions:
        dx_text = c.get("code", {}).get("text", "")
        if not dx_text:
            coding = c.get("code", {}).get("coding", [])
            dx_text = coding[0].get("display", "") if coding else ""
        cat = c.get("category", [{}])[0].get("coding", [{}])[0].get("code", "")
        if cat == "problem-list-item":
            if not primary_dx:
                primary_dx = dx_text
            else:
                secondary.append(dx_text)

    result["admission_diagnosis"] = primary_dx
    result["secondary_diagnoses"] = secondary

    # ── Composition sections: narrative, history, follow-up, warnings ─────
    compositions = by_type.get("Composition", [])
    comp = compositions[0] if compositions else {}
    for section in comp.get("section", []):
        title = section.get("title", "").lower()
        text_div = section.get("text", {}).get("div", "")

        if "past illness" in title or "history" in title:
            # Extract <p> tags as list items
            items = re.findall(r"<p>(.*?)</p>", text_div, re.DOTALL)
            if items:
                result["clinical_history"] = items
        elif "clinical narrative" in title or "document narrative" in title:
            clean = text_div.replace("<p>", "").replace("</p>", " ").strip()
            if clean:
                result["narrative"] = clean
        elif "plan of care" in title:
            # Extract follow-up text
            fu_match = re.search(r"Follow-up:</strong>\s*(.*?)</p>", text_div, re.DOTALL)
            if fu_match:
                result["follow_up"] = fu_match.group(1).strip()
            # Extract warning symptoms
            warn_items = re.findall(r"<li>(.*?)</li>", text_div, re.DOTALL)
            if warn_items:
                result["warning_symptoms"] = warn_items

    # ── Allergies ─────────────────────────────────────────────────────────
    allergies = by_type.get("AllergyIntolerance", [])
    allergy_list: list[str] = []
    for a in allergies:
        notes = a.get("note", [])
        for n in notes:
            text = n.get("text", "")
            if text:
                allergy_list.append(text)
    result["allergies"] = allergy_list

    # ── Procedures ────────────────────────────────────────────────────────
    procedures = by_type.get("Procedure", [])
    proc_list: list[str] = []
    for p in procedures:
        notes = p.get("note", [])
        for n in notes:
            text = n.get("text", "")
            if text:
                proc_list.append(text)
    result["procedures"] = proc_list

    # ── Observations (lab values) ─────────────────────────────────────────
    observations = by_type.get("Observation", [])
    lab_list: list[dict] = []
    for obs in observations:
        code = obs.get("code", {}).get("coding", [{}])[0].get("display", "")
        vq = obs.get("valueQuantity", {})
        ref_range = obs.get("referenceRange", [{}])[0].get("text", "")
        lab_list.append({
            "parameter": code,
            "value": _format_lab_value(vq.get("value", "")),
            "unit": vq.get("unit", ""),
            "reference": ref_range,
        })
    result["lab_values"] = lab_list

    # ── MedicationRequests ────────────────────────────────────────────────
    med_requests = by_type.get("MedicationRequest", [])
    med_list: list[dict] = []
    for mr in med_requests:
        med_code = mr.get("medicationCodeableConcept", {})
        med_text = med_code.get("text", "")
        # The text is "Aspirin 100 mg daily" — extract just the drug name (first word)
        name = med_text.split(" ")[0] if " " in med_text else med_text
        # Fallback: use ATC coding display if text-based name is too short or missing
        if not name or len(name) < 2:
            coding_list = med_code.get("coding", [])
            name = coding_list[0].get("display", "") if coding_list else ""

        dosage = mr.get("dosageInstruction", [{}])[0]
        dose = dosage.get("text", "")
        timing_text = dosage.get("timing", {}).get("code", {}).get("text", "")
        purpose = dosage.get("patientInstruction", "")

        med_list.append({
            "name": name.strip(),
            "dose": dose or med_text,
            "purpose": purpose,
            "time": timing_text or dose,
        })
    result["discharge_medications"] = med_list

    return result


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    lang_files = {
        "en": "fhir_discharge_summary_en.json",
        "it": "fhir_discharge_summary_it.json",
        "pt": "fhir_discharge_summary_pt.json",
        "fi": "fhir_discharge_summary_fi.json",
        "el": "fhir_discharge_summary_el.json",
    }

    ok = 0
    for lang, fname in lang_files.items():
        path = os.path.join(sample_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP {lang}: {fname} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        flat = parse_fhir_bundle(bundle)
        checks = [
            ("patient.name", flat.get("patient", {}).get("name", "")),
            ("admission_diagnosis", flat.get("admission_diagnosis", "")),
            ("lab_values", len(flat.get("lab_values", []))),
            ("medications", len(flat.get("discharge_medications", []))),
            ("narrative", bool(flat.get("narrative"))),
            ("clinical_history", len(flat.get("clinical_history", []))),
            ("warning_symptoms", len(flat.get("warning_symptoms", []))),
        ]
        failures = []
        for label, val in checks:
            if not val or (isinstance(val, int) and val == 0):
                failures.append(f"{label}={val!r}")
        if failures:
            print(f"  FAIL {lang}: {', '.join(failures)}")
        else:
            ok += 1
            print(f"  OK   {lang}: patient={flat['patient']['name']}, "
                  f"dx='{flat['admission_diagnosis'][:50]}...', "
                  f"labs={len(flat['lab_values'])}, meds={len(flat['discharge_medications'])}")

    sys.exit(0 if ok == 5 else 1)
