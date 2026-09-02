"""
Module 5: Biomarker & Genetic Risk Simulation (Biyobelirteç ve Genetik Risk Simülasyonu)

Down syndrome (Trisomy 21) causes a 3× overexpression of the APP (Amyloid Precursor
Protein) gene, which resides on chromosome 21.  This overexpression drives accelerated
amyloid-beta accumulation — the key molecular pathway toward Alzheimer's disease.

This module provides:
  1. APP overexpression scoring based on genomic data (or proxy cfDNA features)
  2. Age-weighted Alzheimer risk curve unique to Trisomy 21 individuals
  3. Composite risk integrating genetics + clinical + cfDNA features
  4. A forward-looking "Alzheimer development curve" for patient counselling

Academic reference angle:
  "Deterministic Genetic Risk Engine for APP-Driven Alzheimer Progression
   in Down Syndrome: Integrating cfDNA, Age, and Clinical Biomarkers"
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────────────
# Known genetic risk weights (literature-based)
# ─────────────────────────────────────────────
APP_BASELINE_COPIES = 2          # diploid
APP_TRISOMY21_COPIES = 3         # extra copy on Chr21
APP_OVEREXPRESSION_RATIO = 1.5   # ~50% more APP protein in T21

# Age-specific cumulative Alzheimer incidence in T21 (%) — based on published cohort data
# (Wiseman et al. 2015; Hithersay et al. 2019)
T21_ALZHEIMER_INCIDENCE = {
    30: 2,
    35: 5,
    40: 30,
    45: 55,
    50: 75,
    55: 88,
    60: 95,
    65: 97,
}

GENERAL_POPULATION_INCIDENCE = {
    30: 0.1,
    35: 0.2,
    40: 1.0,
    45: 2.5,
    50: 5.0,
    55: 10.0,
    60: 18.0,
    65: 25.0,
}


@dataclass
class GeneticProfile:
    patient_id: str
    age: int
    has_trisomy21: bool

    # Optional genomic inputs (from WGS / cfDNA proxy)
    app_copy_number: float = 3.0       # measured APP gene copies
    apoe_genotype: str = "E3/E3"       # APOE allele (E4 → higher risk)
    chr21_methylation_index: float = 0.5  # 0=hypomethylated (risk), 1=normal

    # Clinical & cfDNA features (bridged from Module 1)
    chr21_read_ratio: Optional[float] = None   # from cfDNA (>1.02 = elevated)
    fetal_fraction: Optional[float] = None

    def apoe_risk_multiplier(self) -> float:
        """APOE genotype → additional risk multiplier."""
        multipliers = {
            "E2/E2": 0.5, "E2/E3": 0.7, "E3/E3": 1.0,
            "E2/E4": 1.5, "E3/E4": 2.5, "E4/E4": 5.0,
        }
        return multipliers.get(self.apoe_genotype, 1.0)

    def app_overexpression_score(self) -> float:
        """0-1 score; 1 = maximal overexpression."""
        ratio = self.app_copy_number / APP_BASELINE_COPIES
        return float(np.clip((ratio - 1.0) / 2.0, 0, 1))  # 1.5x → 0.25, 3x → 1.0

    def methylation_risk(self) -> float:
        """Lower methylation index → higher epigenetic risk."""
        return round(1 - self.chr21_methylation_index, 3)


class AlzheimerRiskEngine:
    """
    Combines:
      - T21 age-incidence curve (epidemiological baseline)
      - APP overexpression score
      - APOE genotype multiplier
      - Epigenetic methylation risk
      - cfDNA Chr21 read ratio (optional, from Module 1)
    → Single composite risk score (0-100) and a 20-year forward projection.
    """

    def compute(self, profile: GeneticProfile) -> dict:
        # 1. Epidemiological baseline from age-incidence table
        base_incidence = self._interpolate_incidence(profile.age, profile.has_trisomy21)

        # 2. Genetic modifiers
        app_score = profile.app_overexpression_score()          # 0-1
        apoe_mult = profile.apoe_risk_multiplier()              # 0.5-5.0
        meth_risk = profile.methylation_risk()                   # 0-1

        # 3. cfDNA proxy (if available from module1)
        cfdna_modifier = 1.0
        if profile.chr21_read_ratio is not None:
            deviation = max(0, profile.chr21_read_ratio - 1.0) / 0.05
            cfdna_modifier = 1 + 0.2 * np.clip(deviation, 0, 3)

        # 4. Composite formula
        genetic_multiplier = (1 + app_score * 0.4) * (apoe_mult / 1.0) * (1 + meth_risk * 0.3)
        raw_risk = base_incidence * genetic_multiplier * cfdna_modifier
        composite_risk = float(np.clip(raw_risk, 0, 100))

        return {
            "patient_id": profile.patient_id,
            "age": profile.age,
            "base_incidence_pct": round(base_incidence, 2),
            "app_overexpression_score": round(app_score, 3),
            "apoe_risk_multiplier": apoe_mult,
            "methylation_risk": round(meth_risk, 3),
            "cfdna_modifier": round(cfdna_modifier, 3),
            "composite_risk_pct": round(composite_risk, 1),
            "risk_level": self._level(composite_risk),
            "genetic_multiplier": round(genetic_multiplier, 3),
        }

    def project_curve(self, profile: GeneticProfile, years: int = 20) -> pd.DataFrame:
        """
        Forward-projects composite risk for each year up to `years` from current age.
        Returns a DataFrame: age | base_incidence | composite_risk
        """
        rows = []
        for offset in range(years + 1):
            future_age = profile.age + offset
            future_profile = GeneticProfile(
                patient_id=profile.patient_id,
                age=future_age,
                has_trisomy21=profile.has_trisomy21,
                app_copy_number=profile.app_copy_number,
                apoe_genotype=profile.apoe_genotype,
                chr21_methylation_index=profile.chr21_methylation_index,
                chr21_read_ratio=profile.chr21_read_ratio,
                fetal_fraction=profile.fetal_fraction,
            )
            result = self.compute(future_profile)
            rows.append({
                "age": future_age,
                "base_incidence_pct": result["base_incidence_pct"],
                "composite_risk_pct": result["composite_risk_pct"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _interpolate_incidence(age: int, is_t21: bool) -> float:
        table = T21_ALZHEIMER_INCIDENCE if is_t21 else GENERAL_POPULATION_INCIDENCE
        ages = sorted(table.keys())
        if age <= ages[0]:
            return table[ages[0]]
        if age >= ages[-1]:
            return table[ages[-1]]
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                t = (age - ages[i]) / (ages[i + 1] - ages[i])
                return table[ages[i]] + t * (table[ages[i + 1]] - table[ages[i]])
        return 0.0

    @staticmethod
    def _level(score: float) -> str:
        if score < 20:
            return "Low"
        if score < 50:
            return "Moderate"
        if score < 75:
            return "High"
        return "Very High"


def simulate_cohort(n_patients: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic cohort of T21 patients with diverse genetic profiles."""
    rng = np.random.default_rng(seed)
    apoe_options = ["E3/E3", "E3/E3", "E3/E3", "E3/E4", "E4/E4", "E2/E3"]  # realistic freq
    engine = AlzheimerRiskEngine()

    rows = []
    for i in range(n_patients):
        age = int(rng.integers(25, 60))
        profile = GeneticProfile(
            patient_id=f"DS_{i+1:03d}",
            age=age,
            has_trisomy21=True,
            app_copy_number=rng.normal(3.0, 0.15),
            apoe_genotype=rng.choice(apoe_options),
            chr21_methylation_index=rng.uniform(0.2, 0.8),
            chr21_read_ratio=rng.normal(1.05, 0.02),
        )
        result = engine.compute(profile)
        rows.append(result)

    return pd.DataFrame(rows)


def plot_risk_curve(profile: GeneticProfile, projection_df: pd.DataFrame, save_path: str = None):
    """Plots the personalized Alzheimer development curve vs T21 population average."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ages = sorted(T21_ALZHEIMER_INCIDENCE.keys())
    pop_risks = [T21_ALZHEIMER_INCIDENCE[a] for a in ages]

    ax1 = axes[0]
    ax1.plot(ages, pop_risks, "b--", linewidth=1.5, label="T21 Population Average")
    ax1.plot(projection_df["age"], projection_df["composite_risk_pct"],
             "r-", linewidth=2.5, label=f"Patient {profile.patient_id}")
    ax1.fill_between(projection_df["age"], projection_df["base_incidence_pct"],
                     projection_df["composite_risk_pct"], alpha=0.15, color="red",
                     label="Genetic uplift")
    ax1.axhline(50, color="orange", linestyle=":", linewidth=1, label="50% risk line")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Cumulative Alzheimer Risk (%)")
    ax1.set_title("Personalized Alzheimer Development Curve")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 105)

    ax2 = axes[1]
    engine = AlzheimerRiskEngine()
    factors = {
        "APP Overexpression": profile.app_overexpression_score() * 100,
        "APOE Multiplier": (profile.apoe_risk_multiplier() - 1) * 20,
        "Methylation Risk": profile.methylation_risk() * 100,
        "cfDNA Chr21 Signal": (
            max(0, (profile.chr21_read_ratio or 1.0) - 1.0) / 0.05 * 20
            if profile.chr21_read_ratio else 0
        ),
    }
    ax2.barh(list(factors.keys()), list(factors.values()),
             color=["#E53935", "#FB8C00", "#43A047", "#1E88E5"])
    ax2.set_xlabel("Relative Contribution (%)")
    ax2.set_title("Risk Factor Breakdown")
    ax2.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Genetic Risk Engine — Patient: {profile.patient_id} (Age {profile.age})",
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[GeneticRisk] Plot saved: {save_path}")
    plt.close(fig)


def plot_cohort_distribution(cohort_df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(cohort_df["composite_risk_pct"], bins=20, color="#1976D2", edgecolor="white")
    axes[0].set_xlabel("Composite Risk (%)")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("Cohort Risk Distribution")
    axes[0].grid(alpha=0.3)

    level_counts = cohort_df["risk_level"].value_counts()
    colors = {"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#F44336", "Very High": "#9C27B0"}
    axes[1].bar(level_counts.index, level_counts.values,
                color=[colors.get(l, "gray") for l in level_counts.index])
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Risk Level Classification")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Cohort Genetic Risk Overview", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[GeneticRisk] Cohort plot saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    engine = AlzheimerRiskEngine()

    profile = GeneticProfile(
        patient_id="DS_001",
        age=38,
        has_trisomy21=True,
        app_copy_number=3.1,
        apoe_genotype="E3/E4",
        chr21_methylation_index=0.35,
        chr21_read_ratio=1.07,
    )

    result = engine.compute(profile)
    print("=== Single Patient Risk Assessment ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    projection = engine.project_curve(profile, years=25)
    projection.to_csv("data/alzheimer_risk_curve_DS_001.csv", index=False)
    print(f"\n20-year projection:\n{projection.to_string(index=False)}")

    plot_risk_curve(profile, projection, save_path="outputs/genetic_risk_curve.png")

    print("\n=== Simulating Cohort of 100 patients ===")
    cohort = simulate_cohort(n_patients=100)
    cohort.to_csv("data/cohort_genetic_risk.csv", index=False)
    print(cohort[["patient_id", "age", "composite_risk_pct", "risk_level"]].head(10).to_string())
    plot_cohort_distribution(cohort, save_path="outputs/cohort_risk_distribution.png")
