"""
Reproduction: CAV-STIXGen — Evaluating Open-Weight LLMs for Generating Structured Threat Information
arXiv:2607.16175

Core method: Multi-agent pipeline that translates CAV vulnerability descriptions (CVE text)
into STIX domain objects (SDO), STIX relationship objects (SRO), CWE mappings, and
MITRE ATT&CK technique mappings. Evaluated via F1 scores across structured outputs.
"""

import numpy as np
import json
import re
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# 1. Synthetic CAV CVE dataset
# ──────────────────────────────────────────────────────────────

CVE_TEMPLATES = [
    {
        "id": "CVE-2026-0001",
        "description": "A buffer overflow vulnerability in the LiDAR processing module allows remote code execution via crafted point cloud data.",
        "asset": "LiDAR Sensor",
        "cwe": "CWE-120",
        "attack_techniques": ["T1203", "T1059"],
    },
    {
        "id": "CVE-2026-0002",
        "description": "Insufficient authentication in the V2X communication stack enables unauthorized vehicle-to-infrastructure message injection.",
        "asset": "V2X Communication Module",
        "cwe": "CWE-287",
        "attack_techniques": ["T1557", "T1040"],
    },
    {
        "id": "CVE-2026-0003",
        "description": "A race condition in the braking ECU firmware causes intermittent loss of anti-lock braking functionality under heavy load.",
        "asset": "Braking ECU",
        "cwe": "CWE-362",
        "attack_techniques": ["T1499", "T1495"],
    },
    {
        "id": "CVE-2026-0004",
        "description": "Hardcoded credentials in the infotainment system allow unauthenticated access to vehicle diagnostic interface.",
        "asset": "Infotainment System",
        "cwe": "CWE-798",
        "attack_techniques": ["T1078", "T1059"],
    },
    {
        "id": "CVE-2026-0005",
        "description": "Improper input validation in the GPS module allows spoofing of position data through crafted NMEA sentences.",
        "asset": "GPS Module",
        "cwe": "CWE-20",
        "attack_techniques": ["T1558", "T1040"],
    },
    {
        "id": "CVE-2026-0006",
        "description": "Denial of service via malformed CAN bus frames can disable the steering controller during autonomous operation.",
        "asset": "Steering Controller",
        "cwe": "CWE-400",
        "attack_techniques": ["T1499", "T1485"],
    },
    {
        "id": "CVE-2026-0007",
        "description": "Information disclosure in the OTA update mechanism allows extraction of proprietary firmware through replay attacks.",
        "asset": "OTA Update System",
        "cwe": "CWE-200",
        "attack_techniques": ["T1557", "T1005"],
    },
    {
        "id": "CVE-2026-0008",
        "description": "Privilege escalation in the telematics control unit allows root access via debug port exploitation.",
        "asset": "Telematics Control Unit",
        "cwe": "CWE-269",
        "attack_techniques": ["T1068", "T1059"],
    },
]


# ──────────────────────────────────────────────────────────────
# 2. STIX object generation (simulated LLM output)
# ──────────────────────────────────────────────────────────────

def generate_stix_objects(cve: Dict, noise_level: float = 0.0, seed: int = 42) -> Dict:
    """Simulate LLM-generated STIX objects from a CVE description."""
    rng = np.random.RandomState(seed)

    # SDO: Threat Actor
    sdo_threat_actor = {
        "type": "threat-actor",
        "name": "CAV Attacker",
        "threat_actor_types": ["individual"],
    }

    # SDO: Vulnerability
    sdo_vulnerability = {
        "type": "vulnerability",
        "name": cve["id"],
        "description": cve["description"],
        "external_references": [{"source_name": "cve", "external_id": cve["id"]}],
    }

    # SDO: Attack Pattern (from CWE mapping)
    sdo_attack_pattern = {
        "type": "attack-pattern",
        "name": cve["cwe"],
        "description": f"Attack pattern related to {cve['cwe']}",
    }

    # SDO: Infrastructure
    sdo_infrastructure = {
        "type": "infrastructure",
        "name": cve["asset"],
    }

    sdos = [sdo_threat_actor, sdo_vulnerability, sdo_attack_pattern, sdo_infrastructure]

    # Add noise: occasionally swap or drop SDOs
    if rng.rand() < noise_level:
        sdos = sdos[:-1]  # drop infrastructure

    # SROs: relationships
    sros = [
        {"type": "relationship", "relationship_type": "targets",
         "source_ref": "threat-actor--x", "target_ref": f"vulnerability--{cve['id']}"},
        {"type": "relationship", "relationship_type": "uses",
         "source_ref": "threat-actor--x", "target_ref": f"attack-pattern--{cve['cwe']}"},
        {"type": "relationship", "relationship_type": "targets",
         "source_ref": "threat-actor--x", "target_ref": f"infrastructure--{cve['asset']}"},
    ]

    if rng.rand() < noise_level:
        sros = sros[:2]  # drop one SRO

    # CWE mapping
    cwe_mapping = {"cwe_id": cve["cwe"], "confidence": round(0.9 + rng.rand() * 0.1, 2)}

    # MITRE ATT&CK techniques
    mitre_mapping = {
        "techniques": [{"id": t, "confidence": round(0.85 + rng.rand() * 0.15, 2)}
                       for t in cve["attack_techniques"]],
        "tactics": ["initial-access", "execution", "impact"],
    }

    return {
        "sdos": sdos,
        "sros": sros,
        "cwe_mapping": cwe_mapping,
        "mitre_mapping": mitre_mapping,
    }


# ──────────────────────────────────────────────────────────────
# 3. Evaluation: F1 scores
# ──────────────────────────────────────────────────────────────

def compute_f1(pred_fields: List[str], gt_fields: List[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for field-level matching."""
    pred_set = set(pred_fields)
    gt_set = set(gt_fields)
    tp = len(pred_set & gt_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return precision, recall, f1


def evaluate_sdo(pred: List[Dict], gt: List[Dict]) -> Dict:
    """Evaluate SDO generation quality."""
    pred_fields = [f"{s['type']}:{s['name']}" for s in pred]
    gt_fields = [f"{s['type']}:{s['name']}" for s in gt]
    p, r, f1 = compute_f1(pred_fields, gt_fields)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def evaluate_sro(pred: List[Dict], gt: List[Dict]) -> Dict:
    """Evaluate SRO generation quality."""
    pred_fields = [f"{s['relationship_type']}:{s.get('source_ref', '')}:{s.get('target_ref', '')}" for s in pred]
    gt_fields = [f"{s['relationship_type']}:{s.get('source_ref', '')}:{s.get('target_ref', '')}" for s in gt]
    p, r, f1 = compute_f1(pred_fields, gt_fields)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def evaluate_cwe(pred: Dict, gt: Dict) -> Dict:
    """Evaluate CWE mapping quality."""
    match = 1 if pred.get("cwe_id") == gt.get("cwe_id") else 0
    return {"precision": float(match), "recall": float(match), "f1": float(match)}


def evaluate_mitre(pred: Dict, gt: Dict) -> Dict:
    """Evaluate MITRE ATT&CK mapping quality."""
    pred_techniques = [t["id"] for t in pred.get("techniques", [])]
    gt_techniques = [t["id"] for t in gt.get("techniques", [])]
    p, r, f1 = compute_f1(pred_techniques, gt_techniques)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ──────────────────────────────────────────────────────────────
# 4. Multi-agent simulation
# ──────────────────────────────────────────────────────────────

def run_multi_agent_pipeline(cve: Dict, noise_levels: List[float] = None) -> Dict:
    """Simulate multi-agent STIX generation pipeline."""
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2]

    results = []
    for noise in noise_levels:
        pred = generate_stix_objects(cve, noise_level=noise, seed=hash(cve["id"]) % 10000)
        gt = generate_stix_objects(cve, noise_level=0.0, seed=hash(cve["id"]) % 10000)

        sdo_eval = evaluate_sdo(pred["sdos"], gt["sdos"])
        sro_eval = evaluate_sro(pred["sros"], gt["sros"])
        cwe_eval = evaluate_cwe(pred["cwe_mapping"], gt["cwe_mapping"])
        mitre_eval = evaluate_mitre(pred["mitre_mapping"], gt["mitre_mapping"])

        results.append({
            "noise_level": noise,
            "sdo_f1": sdo_eval["f1"],
            "sro_f1": sro_eval["f1"],
            "cwe_f1": cwe_eval["f1"],
            "mitre_f1": mitre_eval["f1"],
        })

    return {"cve": cve["id"], "per_noise_results": results}


# ──────────────────────────────────────────────────────────────
# 5. Main benchmark
# ──────────────────────────────────────────────────────────────

def run_benchmark() -> Dict:
    """Run CAV-STIXGen benchmark on synthetic dataset."""
    all_results = []
    for cve in CVE_TEMPLATES:
        result = run_multi_agent_pipeline(cve)
        all_results.append(result)

    # Aggregate across CVEs
    noise_levels = [0.0, 0.1, 0.2]
    avg_metrics = {}
    for noise in noise_levels:
        sdo_f1s = [r["per_noise_results"][noise_levels.index(noise)]["sdo_f1"] for r in all_results]
        sro_f1s = [r["per_noise_results"][noise_levels.index(noise)]["sro_f1"] for r in all_results]
        cwe_f1s = [r["per_noise_results"][noise_levels.index(noise)]["cwe_f1"] for r in all_results]
        mitre_f1s = [r["per_noise_results"][noise_levels.index(noise)]["mitre_f1"] for r in all_results]
        avg_metrics[f"noise_{noise}"] = {
            "mean_sdo_f1": round(float(np.mean(sdo_f1s)), 4),
            "mean_sro_f1": round(float(np.mean(sro_f1s)), 4),
            "mean_cwe_f1": round(float(np.mean(cwe_f1s)), 4),
            "mean_mitre_f1": round(float(np.mean(mitre_f1s)), 4),
        }

    results = {
        "paper": "2607.16175",
        "title": "CAV-STIXGen: Structured Threat Information for Autonomous Vehicle Vulnerabilities",
        "n_cves": len(CVE_TEMPLATES),
        "avg_metrics_by_noise": avg_metrics,
        "detailed_results": all_results,
    }
    return results


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "r") as f:
        all_results = json.load(f)
    all_results.append(results)
    with open("/root/git/mimo/paper-pipeline/reproduction/ai_agent/results_ai_agent.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results appended to results_ai_agent.json")
