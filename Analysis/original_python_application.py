"""
Phase 6 — First-Order Engineering Budgets and Range-Based Feasibility Analysis
American Football Launcher: Concept A (Elastic Rail Carriage + Three-Contactor Spin Module)

Purpose: screening calculations with parameter ranges, Monte Carlo sweep,
sensitivity analysis, and pass/caution/fail recommendations.
All results are first-order estimates subject to physical measurement refinement.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
OUTPUT_DIR = Path(__file__).parent / "phase6_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. INPUT PARAMETER RANGES
# ──────────────────────────────────────────────────────────────────────

params = {
    # Football properties
    "m_b":       {"low": 0.38,  "nom": 0.41,  "high": 0.45,  "unit": "kg",   "desc": "Ball mass"},
    "r_b":       {"low": 0.080, "nom": 0.087, "high": 0.095, "unit": "m",    "desc": "Ball radius (minor axis)"},
    "v_b":       {"low": 10.0,  "nom": 15.0,  "high": 20.0,  "unit": "m/s",  "desc": "Ball exit speed"},
    "N_b":       {"low": 600,   "nom": 800,   "high": 1000,  "unit": "rpm",  "desc": "Ball spin rate"},

    # Launch system
    "eta":       {"low": 0.30,  "nom": 0.50,  "high": 0.70,  "unit": "-",    "desc": "Launch efficiency"},
    "x_travel":  {"low": 0.40,  "nom": 0.70,  "high": 1.00,  "unit": "m",    "desc": "Carriage travel"},
    "m_c":       {"low": 0.50,  "nom": 1.00,  "high": 2.00,  "unit": "kg",   "desc": "Carriage mass"},
    "v_c_res":   {"low": 2.0,   "nom": 5.0,   "high": 10.0,  "unit": "m/s",  "desc": "Carriage residual speed"},
    "s_arrest":  {"low": 0.02,  "nom": 0.05,  "high": 0.10,  "unit": "m",    "desc": "Arrest stopping distance"},

    # Spin / contact
    "t_c":       {"low": 0.03,  "nom": 0.07,  "high": 0.15,  "unit": "s",    "desc": "Contact time in spin module"},
    "mu_c":      {"low": 0.30,  "nom": 0.55,  "high": 0.80,  "unit": "-",    "desc": "Contactor friction coefficient"},
    "N_preload": {"low": 10.0,  "nom": 17.5,  "high": 25.0,  "unit": "N",    "desc": "Preload per contactor"},
    "r_c":       {"low": 0.020, "nom": 0.035, "high": 0.050, "unit": "m",    "desc": "Contactor roller radius"},
    "n_contact": {"low": 3,     "nom": 3,     "high": 3,     "unit": "-",    "desc": "Number of contactors"},

    # Frame / stability
    "W_frame":   {"low": 80,    "nom": 150,   "high": 250,   "unit": "N",    "desc": "Frame weight"},
    "b_base":    {"low": 0.20,  "nom": 0.35,  "high": 0.50,  "unit": "m",    "desc": "Base half-width"},
    "h_launch":  {"low": 0.20,  "nom": 0.40,  "high": 0.60,  "unit": "m",    "desc": "Launch axis height"},
    "mu_ground": {"low": 0.30,  "nom": 0.50,  "high": 0.70,  "unit": "-",    "desc": "Ground friction coefficient"},
}

N_MC = 20000

# ──────────────────────────────────────────────────────────────────────
# 2. CALCULATION ENGINE
# ──────────────────────────────────────────────────────────────────────

def calculate(p):
    """Run all first-order calculations from a parameter dict. Returns results dict."""
    r = {}

    # --- Launch energy ---
    r["E_ball"] = 0.5 * p["m_b"] * p["v_b"]**2
    r["E_stored"] = r["E_ball"] / p["eta"]

    # --- Elastic drive ---
    r["k_spring"] = 2 * r["E_stored"] / p["x_travel"]**2
    r["F_avg_spring"] = r["E_stored"] / p["x_travel"]
    r["F_peak_spring"] = r["k_spring"] * p["x_travel"]

    # --- Ball acceleration ---
    r["a_ball"] = p["v_b"]**2 / (2 * p["x_travel"])
    r["F_ball"] = p["m_b"] * r["a_ball"]

    # --- Carriage arrest ---
    r["E_carriage"] = 0.5 * p["m_c"] * p["v_c_res"]**2
    r["F_arrest_avg"] = r["E_carriage"] / p["s_arrest"]

    # --- Spin module ---
    r["omega_b"] = 2 * np.pi * p["N_b"] / 60
    r["alpha_b"] = r["omega_b"] / p["t_c"]
    r["I_b"] = 0.4 * p["m_b"] * p["r_b"]**2
    r["T_spin"] = r["I_b"] * r["alpha_b"]

    # --- Friction / preload ---
    r["F_t_available"] = p["n_contact"] * p["mu_c"] * p["N_preload"]
    r["F_t_required"] = r["T_spin"] / p["r_b"]
    r["spin_margin"] = r["F_t_available"] / r["F_t_required"] if r["F_t_required"] > 0 else np.inf

    # --- Contactor RPM ---
    r["v_surface"] = r["omega_b"] * p["r_b"]
    r["contactor_rpm"] = (r["v_surface"] / (2 * np.pi * p["r_c"])) * 60

    # --- Frame stability ---
    r["F_launch"] = r["F_avg_spring"]
    r["M_tip"] = r["F_launch"] * p["h_launch"]
    r["M_stable"] = p["W_frame"] * p["b_base"]
    r["tip_ratio"] = r["M_stable"] / r["M_tip"] if r["M_tip"] > 0 else np.inf
    r["F_friction_ground"] = p["mu_ground"] * p["W_frame"]
    r["slide_ratio"] = r["F_friction_ground"] / r["F_launch"] if r["F_launch"] > 0 else np.inf

    return r


def assess(results):
    """Apply pass / caution / fail flags to a results dict."""
    flags = {}

    # Spin friction margin: >2 pass, 1-2 caution, <1 fail
    m = results["spin_margin"]
    flags["spin_margin"] = "PASS" if m >= 2.0 else ("CAUTION" if m >= 1.0 else "FAIL")

    # Tip-over ratio: >2 pass, 1-2 caution, <1 fail
    t = results["tip_ratio"]
    flags["tip_ratio"] = "PASS" if t >= 2.0 else ("CAUTION" if t >= 1.0 else "FAIL")

    # Slide ratio: >1.5 pass, 1-1.5 caution, <1 fail
    s = results["slide_ratio"]
    flags["slide_ratio"] = "PASS" if s >= 1.5 else ("CAUTION" if s >= 1.0 else "FAIL")

    # Arrest force: <2000 N pass, 2000-5000 caution, >5000 fail
    f = results["F_arrest_avg"]
    flags["arrest_force"] = "PASS" if f <= 2000 else ("CAUTION" if f <= 5000 else "FAIL")

    # Peak spring force: <500 N pass, 500-1000 caution, >1000 fail
    fp = results["F_peak_spring"]
    flags["peak_spring_force"] = "PASS" if fp <= 500 else ("CAUTION" if fp <= 1000 else "FAIL")

    # Stored energy: <200 J pass, 200-500 caution, >500 fail
    e = results["E_stored"]
    flags["stored_energy"] = "PASS" if e <= 200 else ("CAUTION" if e <= 500 else "FAIL")

    # Contactor RPM: <5000 pass, 5000-10000 caution, >10000 fail
    cr = results["contactor_rpm"]
    flags["contactor_rpm"] = "PASS" if cr <= 5000 else ("CAUTION" if cr <= 10000 else "FAIL")

    return flags


# ──────────────────────────────────────────────────────────────────────
# 3. NAMED CASES: LOW / NOMINAL / STRETCH
# ──────────────────────────────────────────────────────────────────────

def build_case(level):
    """Build a parameter dict for 'low', 'nom', or 'high' case."""
    return {k: v[level] for k, v in params.items()}

cases = {}
for level, label in [("low", "Safe Prototype"), ("nom", "Nominal"), ("high", "Stretch Target")]:
    p = build_case(level)
    r = calculate(p)
    f = assess(r)
    cases[label] = {"params": p, "results": r, "flags": f}


# ──────────────────────────────────────────────────────────────────────
# 4. MONTE CARLO SWEEP
# ──────────────────────────────────────────────────────────────────────

def sample_uniform(pdef, n):
    """Sample n parameter sets uniformly between low and high."""
    samples = {}
    for k, v in pdef.items():
        if v["low"] == v["high"]:
            samples[k] = np.full(n, v["low"])
        else:
            samples[k] = np.random.uniform(v["low"], v["high"], n)
    return samples

mc_samples = sample_uniform(params, N_MC)

mc_keys = [
    "E_ball", "E_stored", "k_spring", "F_avg_spring", "F_peak_spring",
    "a_ball", "F_ball", "E_carriage", "F_arrest_avg",
    "omega_b", "alpha_b", "I_b", "T_spin",
    "F_t_available", "F_t_required", "spin_margin",
    "v_surface", "contactor_rpm",
    "F_launch", "M_tip", "M_stable", "tip_ratio",
    "F_friction_ground", "slide_ratio",
]

mc_results = {k: np.zeros(N_MC) for k in mc_keys}

for i in range(N_MC):
    p_i = {k: mc_samples[k][i] for k in params}
    r_i = calculate(p_i)
    for k in mc_keys:
        mc_results[k][i] = r_i[k]


# ──────────────────────────────────────────────────────────────────────
# 5. SUMMARY STATISTICS
# ──────────────────────────────────────────────────────────────────────

stat_rows = []
for k in mc_keys:
    arr = mc_results[k]
    stat_rows.append({
        "Parameter": k,
        "Min": np.min(arr),
        "P5": np.percentile(arr, 5),
        "Median": np.median(arr),
        "Mean": np.mean(arr),
        "P95": np.percentile(arr, 95),
        "Max": np.max(arr),
        "Std": np.std(arr),
    })
df_stats = pd.DataFrame(stat_rows)

# Risk summary from MC
risk_counts = {"spin_margin": 0, "tip_ratio": 0, "slide_ratio": 0,
               "arrest_force": 0, "peak_spring_force": 0,
               "stored_energy": 0, "contactor_rpm": 0}

risk_counts["spin_margin"] = np.sum(mc_results["spin_margin"] < 1.0)
risk_counts["tip_ratio"] = np.sum(mc_results["tip_ratio"] < 1.0)
risk_counts["slide_ratio"] = np.sum(mc_results["slide_ratio"] < 1.0)
risk_counts["arrest_force"] = np.sum(mc_results["F_arrest_avg"] > 5000)
risk_counts["peak_spring_force"] = np.sum(mc_results["F_peak_spring"] > 1000)
risk_counts["stored_energy"] = np.sum(mc_results["E_stored"] > 500)
risk_counts["contactor_rpm"] = np.sum(mc_results["contactor_rpm"] > 10000)

risk_pct = {k: 100 * v / N_MC for k, v in risk_counts.items()}


# ──────────────────────────────────────────────────────────────────────
# 6. NAMED-CASE COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────

output_keys_report = [
    ("E_ball",          "Ball kinetic energy",     "J"),
    ("E_stored",        "Stored elastic energy",   "J"),
    ("k_spring",        "Spring stiffness",        "N/m"),
    ("F_avg_spring",    "Average spring force",    "N"),
    ("F_peak_spring",   "Peak spring force",       "N"),
    ("a_ball",          "Ball acceleration",       "m/s²"),
    ("F_ball",          "Force on ball",           "N"),
    ("E_carriage",      "Carriage residual energy","J"),
    ("F_arrest_avg",    "Average arrest force",    "N"),
    ("T_spin",          "Spin-up torque",          "Nm"),
    ("spin_margin",     "Spin friction margin",    "-"),
    ("contactor_rpm",   "Contactor RPM",           "rpm"),
    ("tip_ratio",       "Tip-over stability ratio","-"),
    ("slide_ratio",     "Sliding stability ratio", "-"),
]

rows_report = []
for key, desc, unit in output_keys_report:
    row = {"Output": desc, "Unit": unit}
    for label in ["Safe Prototype", "Nominal", "Stretch Target"]:
        row[label] = f"{cases[label]['results'][key]:.3g}"
    rows_report.append(row)
df_report = pd.DataFrame(rows_report)

# Flags table
flag_rows = []
for label in ["Safe Prototype", "Nominal", "Stretch Target"]:
    for fk, fv in cases[label]["flags"].items():
        flag_rows.append({"Case": label, "Check": fk, "Status": fv})
df_flags = pd.DataFrame(flag_rows)


# ──────────────────────────────────────────────────────────────────────
# 7. SENSITIVITY ANALYSIS — TORNADO CHART
# ──────────────────────────────────────────────────────────────────────

sensitivity_targets = [
    ("E_stored",     "Stored Energy (J)"),
    ("F_peak_spring","Peak Spring Force (N)"),
    ("spin_margin",  "Spin Friction Margin"),
    ("F_arrest_avg", "Average Arrest Force (N)"),
    ("tip_ratio",    "Tip-Over Stability Ratio"),
    ("contactor_rpm","Contactor RPM"),
]

for target_key, target_label in sensitivity_targets:
    nom_p = build_case("nom")
    nom_val = calculate(nom_p)[target_key]
    deltas = []

    for pk in params:
        p_lo = nom_p.copy()
        p_hi = nom_p.copy()
        p_lo[pk] = params[pk]["low"]
        p_hi[pk] = params[pk]["high"]
        val_lo = calculate(p_lo)[target_key]
        val_hi = calculate(p_hi)[target_key]
        deltas.append({
            "param": pk,
            "desc": params[pk]["desc"],
            "lo_delta": val_lo - nom_val,
            "hi_delta": val_hi - nom_val,
            "spread": abs(val_hi - val_lo),
        })

    deltas.sort(key=lambda d: d["spread"], reverse=True)
    top = deltas[:8]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top))
    lo_vals = [d["lo_delta"] for d in top]
    hi_vals = [d["hi_delta"] for d in top]
    labels = [d["desc"] for d in top]

    bars_lo = ax.barh(y_pos, lo_vals, color="#4a90d9", edgecolor="none", height=0.6, label="Low bound")
    bars_hi = ax.barh(y_pos, hi_vals, color="#d94a4a", edgecolor="none", height=0.6, label="High bound")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Change from nominal ({target_label} = {nom_val:.3g})", fontsize=10)
    ax.set_title(f"Sensitivity: {target_label}", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"tornado_{target_key}.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 8. DISTRIBUTION PLOTS — KEY OUTPUTS
# ──────────────────────────────────────────────────────────────────────

hist_targets = [
    ("E_stored",      "Stored Elastic Energy (J)",     None,  200),
    ("F_peak_spring", "Peak Spring Force (N)",          None,  1000),
    ("spin_margin",   "Spin Friction Margin",           1.0,   2.0),
    ("F_arrest_avg",  "Average Arrest Force (N)",       5000,  2000),
    ("tip_ratio",     "Tip-Over Stability Ratio",       1.0,   2.0),
    ("slide_ratio",   "Sliding Stability Ratio",        1.0,   1.5),
    ("contactor_rpm", "Contactor RPM",                  10000, 5000),
]

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for idx, (key, label, fail_line, caution_line) in enumerate(hist_targets):
    ax = axes[idx]
    ax.hist(mc_results[key], bins=80, color="#5b9bd5", edgecolor="none", alpha=0.85)
    if fail_line is not None:
        ax.axvline(fail_line, color="red", linewidth=1.5, linestyle="--", label=f"Fail threshold = {fail_line}")
    if caution_line is not None:
        ax.axvline(caution_line, color="orange", linewidth=1.5, linestyle="--", label=f"Caution = {caution_line}")
    ax.set_xlabel(label, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

for idx in range(len(hist_targets), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"Monte Carlo Distributions (n = {N_MC})", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / "mc_distributions.png", dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 9. SCATTER: SPIN MARGIN vs EXIT SPEED (coloured by contact time)
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(mc_samples["v_b"], mc_results["spin_margin"],
                c=mc_samples["t_c"], cmap="viridis", s=3, alpha=0.4)
ax.axhline(1.0, color="red", linewidth=1, linestyle="--", label="Margin = 1 (minimum)")
ax.axhline(2.0, color="orange", linewidth=1, linestyle="--", label="Margin = 2 (target)")
ax.set_xlabel("Ball exit speed (m/s)", fontsize=10)
ax.set_ylabel("Spin friction margin", fontsize=10)
ax.set_title("Spin Margin vs Exit Speed (coloured by contact time)", fontsize=11, fontweight="bold")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Contact time (s)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "spin_margin_vs_speed.png", dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 10. EXPORT TO EXCEL
# ──────────────────────────────────────────────────────────────────────

excel_path = OUTPUT_DIR / "phase6_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    # Input parameters
    df_inputs = pd.DataFrame([
        {"Parameter": k, "Description": v["desc"], "Low": v["low"],
         "Nominal": v["nom"], "High": v["high"], "Unit": v["unit"]}
        for k, v in params.items()
    ])
    df_inputs.to_excel(writer, sheet_name="Input Parameters", index=False)

    # Named case comparison
    df_report.to_excel(writer, sheet_name="Case Comparison", index=False)

    # Flags
    df_flags.to_excel(writer, sheet_name="Risk Flags", index=False)

    # MC statistics
    df_stats.to_excel(writer, sheet_name="MC Statistics", index=False)

    # MC risk percentages
    df_risk = pd.DataFrame([
        {"Check": k, "Fail Count": risk_counts[k],
         "Fail %": f"{risk_pct[k]:.1f}%"}
        for k in risk_counts
    ])
    df_risk.to_excel(writer, sheet_name="MC Risk Summary", index=False)


# ──────────────────────────────────────────────────────────────────────
# 11. CONSOLE OUTPUT
# ──────────────────────────────────────────────────────────────────────

print("=" * 80)
print("PHASE 6 -- FIRST-ORDER FEASIBILITY ANALYSIS")
print("Concept A: Elastic Rail Carriage + Three-Contactor Spin Module")
print("=" * 80)

print("\n-- NAMED CASE COMPARISON --\n")
print(df_report.to_string(index=False))

print("\n-- RISK FLAGS --\n")
for label in ["Safe Prototype", "Nominal", "Stretch Target"]:
    flags = cases[label]["flags"]
    line = f"  {label:20s} | " + " | ".join(f"{k}: {v}" for k, v in flags.items())
    print(line)

print("\n-- MONTE CARLO RISK SUMMARY --\n")
for k in risk_counts:
    print(f"  {k:25s}  FAIL in {risk_counts[k]:5d} / {N_MC} runs  ({risk_pct[k]:.1f}%)")

print("\n-- MONTE CARLO STATISTICS (selected) --\n")
sel = ["E_stored", "F_peak_spring", "spin_margin", "F_arrest_avg",
       "tip_ratio", "slide_ratio", "contactor_rpm"]
print(df_stats[df_stats["Parameter"].isin(sel)].to_string(index=False))

print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
print(f"Excel workbook:   {excel_path.resolve()}")
print("=" * 80)
