"""Explore earlier retrospective experiments; no operational forecasting."""
import hashlib
import json
from pathlib import Path
from demo_ui import main

ROOT = Path(__file__).parent
SOURCE = "TyphoFormerPlus/official_leadspecific_12to1_safeneg_summary.json"

def load():
    source = ROOT / SOURCE
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    rows = []
    for model, leads in data["baselines"].items():
        for result in leads.values():
            rows.append({"lead_hours":result["lead_hours"], "label":model, "value":result["err"], "seed_std_km":"not applicable"})
    for model, results in data["models"].items():
        for result in results["aggregate"].values():
            rows.append({"lead_hours":result["lead_hours"], "label":model, "value":result["err"]["mean"], "seed_std_km":result["err"]["std"]})
    if not rows:
        raise ValueError("The source summary contains no comparable rows.")
    return {"rows":rows, "filters":["lead_hours"], "metric":"DeltaR (km), lower is better. Neural values: stored seed means and standard deviations, not confidence intervals.", "source":SOURCE, "sha256":hashlib.sha256(source.read_bytes()).hexdigest()}

if __name__ == "__main__":
    main("TyphoFormer++ / Results explorer", "Browse the earlier strict-6-hour, 2022–2024 retrospective evaluation. These are historical implementation results, not the revised manuscript's final primary results or live weather forecasts.", load)
