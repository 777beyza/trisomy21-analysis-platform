"""
Module 4: Behavioral & Executive Function Tracking — Caregiver Panel
(Davranışsal ve Yürütücü İşlev Takibi)

In Down syndrome, Alzheimer's often begins with personality change, apathy,
and increased agitation — NOT memory loss.  This module lets caregivers log
daily behavioral observations and uses CUSUM (Cumulative Sum) change-point
detection to signal clinically relevant shifts.

Behavioral dimensions tracked (0-10 scale, higher = more severe):
  - apathy          : loss of interest / motivation
  - agitation       : restlessness, repetitive behavior
  - mood            : depression/sadness (10 = most depressed)
  - compliance      : resistance to daily routines (10 = fully non-compliant)
  - social_withdrawal : reduced interaction with others

Academic reference angle:
  "Behavioral Symptom Hierarchy Integration in Down Syndrome–Alzheimer Software:
   A Time-Series Approach Prioritizing Non-Cognitive Early Markers"
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Optional


BEHAVIORAL_DIMS = ["apathy", "agitation", "mood", "compliance", "social_withdrawal"]

ALERT_THRESHOLDS = {
    "apathy":            7.0,
    "agitation":         7.5,
    "mood":              7.0,
    "compliance":        8.0,
    "social_withdrawal": 7.0,
}

CUSUM_K = 0.5   # reference value (half of expected shift)
CUSUM_H = 5.0   # alarm threshold (standard deviations)


@dataclass
class BehavioralEntry:
    date: str
    caregiver_id: str
    apathy: float
    agitation: float
    mood: float
    compliance: float
    social_withdrawal: float
    notes: str = ""

    def to_series(self) -> pd.Series:
        return pd.Series(asdict(self))


class CUSUMDetector:
    """
    Upper-sided CUSUM for detecting a sustained upward shift in a behavioral score.
    A sustained increase (worsening) triggers an alarm.
    """

    def __init__(self, k: float = CUSUM_K, h: float = CUSUM_H):
        self.k = k
        self.h = h

    def run(self, series: np.ndarray) -> dict:
        """
        series: 1-D array of scores.
        Returns dict with cusum_values, alarm_indices, first_alarm.
        """
        mu = series[:20].mean() if len(series) >= 20 else series.mean()
        sigma = series[:20].std() if len(series) >= 20 else (series.std() + 1e-9)

        standardized = (series - mu) / sigma
        C = np.zeros(len(series))
        alarms = []

        for t in range(1, len(series)):
            C[t] = max(0, C[t - 1] + standardized[t] - self.k)
            if C[t] >= self.h:
                alarms.append(t)

        return {
            "cusum_values": C,
            "alarm_indices": alarms,
            "first_alarm": alarms[0] if alarms else None,
            "baseline_mean": round(float(mu), 3),
            "baseline_std": round(float(sigma), 3),
        }


def simulate_behavioral_data(
    patient_id: str = "DS_001",
    n_normal: int = 60,
    n_prodromal: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic caregiver log data.
    n_normal days stable; n_prodromal days with gradual behavioral worsening.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1)
    baselines = {d: rng.uniform(2.0, 4.0) for d in BEHAVIORAL_DIMS}

    rows = []
    for i in range(n_normal + n_prodromal):
        date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        phase = "stable" if i < n_normal else "prodromal"

        step = max(0, i - n_normal + 1)
        drift = {
            "apathy":            step * 0.10,
            "agitation":         step * 0.08,
            "mood":              step * 0.09,
            "compliance":        step * 0.06,
            "social_withdrawal": step * 0.11,
        }

        row = {
            "patient_id": patient_id,
            "date": date,
            "session_index": i,
            "phase": phase,
            "caregiver_id": "C_001",
        }
        for dim in BEHAVIORAL_DIMS:
            val = baselines[dim] + drift[dim] + rng.normal(0, 0.5)
            row[dim] = round(float(np.clip(val, 0, 10)), 2)

        rows.append(row)

    return pd.DataFrame(rows)


class BehavioralTracker:
    """
    Stores caregiver entries and runs CUSUM analysis per behavioral dimension.
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self._entries: List[dict] = []

    def add_entry(self, entry: BehavioralEntry):
        self._entries.append(asdict(entry))

    def load_from_df(self, df: pd.DataFrame):
        """Load historical data from a DataFrame (must contain BEHAVIORAL_DIMS columns)."""
        for _, row in df.iterrows():
            self._entries.append(row.to_dict())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._entries)

    def run_cusum_all(self) -> dict:
        """Run CUSUM on each behavioral dimension. Returns per-dimension results."""
        df = self.to_dataframe()
        detector = CUSUMDetector()
        results = {}
        for dim in BEHAVIORAL_DIMS:
            if dim in df.columns:
                series = df[dim].values.astype(float)
                results[dim] = detector.run(series)
        return results

    def check_threshold_alerts(self) -> dict:
        """Returns dims where the last 7-day average exceeds the clinical alarm threshold."""
        df = self.to_dataframe()
        recent = df.tail(7)
        alerts = {}
        for dim in BEHAVIORAL_DIMS:
            if dim not in recent.columns:
                continue
            avg = float(recent[dim].mean())
            threshold = ALERT_THRESHOLDS[dim]
            if avg >= threshold:
                alerts[dim] = {
                    "7d_average": round(avg, 2),
                    "threshold": threshold,
                    "exceeded_by": round(avg - threshold, 2),
                }
        return alerts

    def generate_report(self) -> dict:
        cusum_results = self.run_cusum_all()
        threshold_alerts = self.check_threshold_alerts()
        df = self.to_dataframe()

        alarmed_dims = [d for d, r in cusum_results.items() if r["first_alarm"] is not None]

        return {
            "patient_id": self.patient_id,
            "total_entries": len(df),
            "date_range": f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}",
            "cusum_alarms": {
                d: {"first_alarm_session": r["first_alarm"]}
                for d, r in cusum_results.items() if r["first_alarm"] is not None
            },
            "threshold_alerts": threshold_alerts,
            "overall_alert": len(alarmed_dims) >= 2 or len(threshold_alerts) > 0,
            "alarmed_dimensions": alarmed_dims,
            "recommendation": (
                "Immediate clinical evaluation recommended."
                if len(alarmed_dims) >= 2
                else "Continue monitoring; schedule routine follow-up."
            ),
        }

    def plot_behavioral_timeline(self, save_path: str = None):
        df = self.to_dataframe()
        cusum_results = self.run_cusum_all()
        detector = CUSUMDetector()

        fig, axes = plt.subplots(len(BEHAVIORAL_DIMS), 2, figsize=(16, 3.5 * len(BEHAVIORAL_DIMS)))

        phase_colors = {"stable": "#4CAF50", "prodromal": "#F44336", None: "#2196F3"}

        for row_idx, dim in enumerate(BEHAVIORAL_DIMS):
            ax_score = axes[row_idx, 0]
            ax_cusum = axes[row_idx, 1]

            # Left: raw score with phase coloring
            if "phase" in df.columns:
                for phase, group in df.groupby("phase"):
                    ax_score.scatter(
                        group["session_index"], group[dim],
                        color=phase_colors.get(phase, "#2196F3"),
                        s=18, label=phase, alpha=0.8
                    )
            else:
                ax_score.plot(df["session_index"], df[dim], color="#2196F3")

            ax_score.axhline(ALERT_THRESHOLDS[dim], color="red", linestyle="--",
                             linewidth=1.2, label=f"Threshold ({ALERT_THRESHOLDS[dim]})")
            ax_score.set_title(f"{dim.replace('_', ' ').title()} — Raw Score")
            ax_score.set_ylim(0, 10.5)
            ax_score.legend(fontsize=7)
            ax_score.grid(alpha=0.3)

            # Right: CUSUM statistic
            res = cusum_results[dim]
            ax_cusum.plot(res["cusum_values"], color="#FF5722", linewidth=1.5)
            ax_cusum.axhline(CUSUM_H, color="black", linestyle="--",
                             linewidth=1.2, label=f"Alarm (H={CUSUM_H})")
            if res["first_alarm"] is not None:
                ax_cusum.axvline(res["first_alarm"], color="purple", linestyle=":",
                                 linewidth=1.5, label=f"1st alarm @ {res['first_alarm']}")
            ax_cusum.set_title(f"{dim.replace('_', ' ').title()} — CUSUM")
            ax_cusum.legend(fontsize=7)
            ax_cusum.grid(alpha=0.3)

        fig.suptitle(f"Behavioral Tracking — Patient: {self.patient_id}", fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[Behavioral] Plot saved: {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    pid = "DS_001"
    df = simulate_behavioral_data(patient_id=pid, n_normal=60, n_prodromal=30)
    df.to_csv(f"data/behavioral_{pid}.csv", index=False)
    print(f"[Behavioral] Simulated {len(df)} days of caregiver logs.")

    tracker = BehavioralTracker(pid)
    tracker.load_from_df(df)

    report = tracker.generate_report()
    print("\n=== Behavioral Tracking Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    tracker.plot_behavioral_timeline(save_path="outputs/behavioral_timeline.png")
