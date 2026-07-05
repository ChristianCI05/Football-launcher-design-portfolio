import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "phase6_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MC_SAMPLES = 20000


PARAMETER_RANGES = {
    "ball_mass": {"low": 0.38, "nom": 0.43, "high": 0.45, "unit": "kg"},
    "ball_radius": {"low": 0.080, "nom": 0.0875, "high": 0.095, "unit": "m"},
    "exit_velocity": {"low": 10.0, "nom": 15.0, "high": 20.0, "unit": "m/s"},
    "spinrate": {"low": 600.0, "nom": 800.0, "high": 1000.0, "unit": "rpm"},
    "efficiency": {"low": 0.30, "nom": 0.50, "high": 0.70, "unit": "-"},
    "travel": {"low": 0.40, "nom": 0.70, "high": 1.00, "unit": "m"},
    "carriage_mass": {"low": 0.50, "nom": 1.25, "high": 2.00, "unit": "kg"},
    "carriage_speed": {"low": 2.0, "nom": 6.0, "high": 10.0, "unit": "m/s"},
    "arrest_distance": {"low": 0.02, "nom": 0.06, "high": 0.10, "unit": "m"},
    "contact_time": {"low": 0.50, "nom": 0.50, "high": 0.50, "unit": "s"},
    "mu": {"low": 0.30, "nom": 0.55, "high": 0.80, "unit": "-"},
    "contactor_preload": {"low": 10.0, "nom": 17.5, "high": 25.0, "unit": "N"},
    "no_contactors": {"low": 3.0, "nom": 3.0, "high": 3.0, "unit": "-"},
    "contactor_radius": {"low": 0.020, "nom": 0.035, "high": 0.050, "unit": "m"},
    "frame_weight": {"low": 80.0, "nom": 165.0, "high": 250.0, "unit": "N"},
    "base_half_width": {"low": 0.20, "nom": 0.35, "high": 0.50, "unit": "m"},
    "launch_height": {"low": 1.22, "nom": 1.52, "high": 1.83, "unit": "m"},
    "ground_friction": {"low": 0.30, "nom": 0.50, "high": 0.70, "unit": "-"},
}


def make_case(case_name):
    params = {}
    for key in PARAMETER_RANGES:
        params[key] = PARAMETER_RANGES[key][case_name]
    return params


def calculate(p):
    e_ball = 0.5 * p["ball_mass"] * p["exit_velocity"] ** 2
    e_stored = e_ball / p["efficiency"]

    spring_stiffness = 2 * e_stored / p["travel"] ** 2
    f_avg = e_stored / p["travel"]
    f_peak = spring_stiffness * p["travel"]

    acceleration = p["exit_velocity"] ** 2 / (2 * p["travel"])
    f_ball = p["ball_mass"] * acceleration

    e_carriage = 0.5 * p["carriage_mass"] * p["carriage_speed"] ** 2
    f_arrest = e_carriage / p["arrest_distance"]

    omega = 2 * math.pi * p["spinrate"] / 60
    alpha = omega / p["contact_time"]
    inertia = 0.4 * p["ball_mass"] * p["ball_radius"] ** 2
    torque = inertia * alpha

    f_available = p["no_contactors"] * p["mu"] * p["contactor_preload"]
    f_required = torque / p["ball_radius"]
    spin_margin = f_available / f_required

    surface_speed = omega * p["ball_radius"]
    contactor_rpm = surface_speed / (2 * math.pi * p["contactor_radius"]) * 60

    tipping_moment = f_peak * p["launch_height"]
    stable_moment = p["frame_weight"] * p["base_half_width"]
    tip_ratio = stable_moment / tipping_moment

    ground_friction_force = p["ground_friction"] * p["frame_weight"]
    slide_margin = ground_friction_force / f_peak

    return {
        "E_ball": e_ball,
        "E_stored": e_stored,
        "spring_stiffness": spring_stiffness,
        "F_avg_launch": f_avg,
        "F_peak_launch": f_peak,
        "ball_acceleration": acceleration,
        "F_ball": f_ball,
        "E_carriage": e_carriage,
        "F_arrest": f_arrest,
        "omega": omega,
        "alpha": alpha,
        "ball_inertia": inertia,
        "spin_torque": torque,
        "F_available_spin": f_available,
        "F_required_spin": f_required,
        "spin_margin": spin_margin,
        "surface_speed": surface_speed,
        "contactor_rpm": contactor_rpm,
        "tipping_moment": tipping_moment,
        "stable_moment": stable_moment,
        "tip_ratio": tip_ratio,
        "ground_friction_force": ground_friction_force,
        "slide_margin": slide_margin,
    }


def status_from_margin(value):
    if value < 1.0:
        return "fail"
    if value < 1.5:
        return "caution"
    return "pass"


def assess(results):
    checks = {}
    checks["spin_margin_status"] = status_from_margin(results["spin_margin"])
    checks["tip_ratio_status"] = status_from_margin(results["tip_ratio"])
    checks["slide_margin_status"] = status_from_margin(results["slide_margin"])

    if results["F_arrest"] > 1500:
        checks["F_arrest_status"] = "fail"
    elif results["F_arrest"] > 1000:
        checks["F_arrest_status"] = "caution"
    else:
        checks["F_arrest_status"] = "pass"

    if results["contactor_rpm"] > 6000:
        checks["contactor_rpm_status"] = "fail"
    elif results["contactor_rpm"] > 4000:
        checks["contactor_rpm_status"] = "caution"
    else:
        checks["contactor_rpm_status"] = "pass"

    return checks


def run_named_cases():
    rows = []
    for case_name in ["low", "nom", "high"]:
        params = make_case(case_name)
        results = calculate(params)
        checks = assess(results)
        row = {"case": case_name}
        row.update(params)
        row.update(results)
        row.update(checks)
        rows.append(row)
    return pd.DataFrame(rows)


def sample_parameters(sample_count):
    samples = {}
    for key in PARAMETER_RANGES:
        low = PARAMETER_RANGES[key]["low"]
        high = PARAMETER_RANGES[key]["high"]
        if low == high:
            samples[key] = np.full(sample_count, low)
        else:
            samples[key] = np.random.uniform(low, high, sample_count)
    return samples


def run_monte_carlo(sample_count):
    samples = sample_parameters(sample_count)
    rows = []

    for i in range(sample_count):
        params = {}
        for key in samples:
            params[key] = samples[key][i]

        results = calculate(params)
        checks = assess(results)

        row = {}
        row.update(params)
        row.update(results)
        row.update(checks)
        rows.append(row)

    return pd.DataFrame(rows)


def make_summary(mc_df):
    output_columns = [
        "E_ball",
        "E_stored",
        "spring_stiffness",
        "F_peak_launch",
        "F_arrest",
        "spin_torque",
        "spin_margin",
        "contactor_rpm",
        "tip_ratio",
        "slide_margin",
    ]

    rows = []
    for column in output_columns:
        values = mc_df[column]
        rows.append({
            "output": column,
            "min": values.min(),
            "P5": values.quantile(0.05),
            "median": values.median(),
            "mean": values.mean(),
            "P95": values.quantile(0.95),
            "max": values.max(),
            "std": values.std(),
        })

    return pd.DataFrame(rows)


def make_risk_table(mc_df):
    check_columns = [
        "spin_margin_status",
        "tip_ratio_status",
        "slide_margin_status",
        "F_arrest_status",
        "contactor_rpm_status",
    ]
    rows = []

    for column in check_columns:
        fail_percent = (mc_df[column] == "fail").mean() * 100
        caution_percent = (mc_df[column] == "caution").mean() * 100
        pass_percent = (mc_df[column] == "pass").mean() * 100
        rows.append({
            "check": column.replace("_status", ""),
            "fail_percent": fail_percent,
            "caution_percent": caution_percent,
            "pass_percent": pass_percent,
        })

    return pd.DataFrame(rows)


def one_at_a_time_sensitivity(output_name):
    nominal_params = make_case("nom")
    nominal_result = calculate(nominal_params)[output_name]
    rows = []

    for parameter in PARAMETER_RANGES:
        low_params = nominal_params.copy()
        high_params = nominal_params.copy()
        low_params[parameter] = PARAMETER_RANGES[parameter]["low"]
        high_params[parameter] = PARAMETER_RANGES[parameter]["high"]

        low_result = calculate(low_params)[output_name]
        high_result = calculate(high_params)[output_name]

        rows.append({
            "parameter": parameter,
            "output": output_name,
            "nominal": nominal_result,
            "low_result": low_result,
            "high_result": high_result,
            "low_change": low_result - nominal_result,
            "high_change": high_result - nominal_result,
            "swing": abs(high_result - low_result),
        })

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df = sensitivity_df.sort_values("swing", ascending=False)
    return sensitivity_df


def save_histograms(mc_df):
    plots = [
        ("E_stored", "Stored Energy (J)", None),
        ("F_peak_launch", "Peak Launch Force (N)", None),
        ("F_arrest", "Carriage Arrest Force (N)", 1000),
        ("spin_margin", "Spin Friction Margin", 1),
        ("contactor_rpm", "Contactor Speed (rpm)", 4000),
        ("tip_ratio", "Tip Ratio", 1),
        ("slide_margin", "Slide Margin", 1),
        ("E_ball", "Ball Energy (J)", None),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()

    for i in range(len(plots)):
        column = plots[i][0]
        title = plots[i][1]
        threshold = plots[i][2]

        axes[i].hist(mc_df[column], bins=40, color="#4c78a8", edgecolor="white")
        axes[i].set_title(title)
        axes[i].set_ylabel("Count")
        if threshold is not None:
            axes[i].axvline(threshold, color="#e45756", linestyle="--", linewidth=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mc_distributions.png", dpi=200)
    plt.close(fig)


def save_scatter(mc_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        mc_df["exit_velocity"],
        mc_df["spin_margin"],
        c=mc_df["contact_time"],
        cmap="viridis",
        alpha=0.45,
        s=10,
    )
    ax.axhline(1.0, color="#e45756", linestyle="--", linewidth=2)
    ax.set_xlabel("Exit Velocity (m/s)")
    ax.set_ylabel("Spin Friction Margin")
    ax.set_title("Spin Margin vs Exit Velocity")
    fig.colorbar(scatter, ax=ax, label="Spin Dwell Time (s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spin_margin_vs_speed.png", dpi=200)
    plt.close(fig)


def save_tornado_chart(sensitivity_df, output_name):
    top = sensitivity_df.head(10).copy()
    top = top.sort_values("swing")

    fig, ax = plt.subplots(figsize=(9, 6))
    y_positions = np.arange(len(top))

    left_values = top["low_result"] - top["nominal"]
    right_values = top["high_result"] - top["nominal"]

    ax.barh(y_positions, left_values, color="#72b7b2", label="Low input")
    ax.barh(y_positions, right_values, color="#f58518", label="High input")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["parameter"])
    ax.set_xlabel("Change from Nominal")
    ax.set_title("Sensitivity of " + output_name)
    ax.legend()
    fig.tight_layout()

    filename = "tornado_" + output_name + ".png"
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_excel(named_cases, mc_df, summary_df, risk_df, sensitivity_tables):
    excel_path = OUTPUT_DIR / "phase6_results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        ranges_rows = []
        for key in PARAMETER_RANGES:
            row = {"parameter": key}
            row.update(PARAMETER_RANGES[key])
            ranges_rows.append(row)

        pd.DataFrame(ranges_rows).to_excel(writer, sheet_name="input_ranges", index=False)
        named_cases.to_excel(writer, sheet_name="named_cases", index=False)
        summary_df.to_excel(writer, sheet_name="mc_summary", index=False)
        risk_df.to_excel(writer, sheet_name="mc_risk", index=False)
        mc_df.head(2000).to_excel(writer, sheet_name="mc_first_2000", index=False)

        for output_name in sensitivity_tables:
            sheet_name = "sens_" + output_name
            sensitivity_tables[output_name].to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return excel_path


def print_console_summary(named_cases, summary_df, risk_df, excel_path):
    print("\nPHASE 6 FEASIBILITY ANALYSIS")
    print("Output folder:", OUTPUT_DIR)
    print("Excel workbook:", excel_path)

    print("\nNamed case margins:")
    columns = [
        "case",
        "spin_margin",
        "spin_margin_status",
        "tip_ratio",
        "tip_ratio_status",
        "slide_margin",
        "slide_margin_status",
        "F_arrest",
        "F_arrest_status",
        "contactor_rpm",
        "contactor_rpm_status",
    ]
    print(named_cases[columns].to_string(index=False))

    print("\nMonte Carlo summary:")
    print(summary_df.to_string(index=False))

    print("\nMonte Carlo risk percentages:")
    print(risk_df.to_string(index=False))

    print("\nMain engineering takeaway:")
    print("Spin margin is generally comfortable with the 0.5 s pre-launch dwell assumption.")
    print("Tip and slide stability are the main feasibility risks at realistic launch heights.")


def main():
    np.random.seed(42)

    named_cases = run_named_cases()
    mc_df = run_monte_carlo(MC_SAMPLES)
    summary_df = make_summary(mc_df)
    risk_df = make_risk_table(mc_df)

    sensitivity_outputs = [
        "E_stored",
        "F_peak_launch",
        "F_arrest",
        "spin_margin",
        "contactor_rpm",
        "tip_ratio",
        "slide_margin",
    ]

    sensitivity_tables = {}
    for output_name in sensitivity_outputs:
        sensitivity_df = one_at_a_time_sensitivity(output_name)
        sensitivity_tables[output_name] = sensitivity_df
        save_tornado_chart(sensitivity_df, output_name)

    save_histograms(mc_df)
    save_scatter(mc_df)
    excel_path = save_excel(named_cases, mc_df, summary_df, risk_df, sensitivity_tables)
    print_console_summary(named_cases, summary_df, risk_df, excel_path)


main()
