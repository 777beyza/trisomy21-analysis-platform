import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None

import librosa

class AcousticAnalyzer:
    """
    Analyzes speech patterns to detect cognitive hesitation and vocal variety
    using real audio processing (librosa).
    """
    def analyze_speech(self, audio_file_path):
        # Load audio using librosa
        # sr=22050 is default
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # Total duration in seconds
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # Split audio into non-silent intervals (top_db is the threshold)
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        
        # Calculate total non-silent duration
        non_silent_samples = sum([end - start for start, end in non_silent_intervals])
        non_silent_duration = non_silent_samples / sr
        silence_duration = total_duration - non_silent_duration
        
        # Hesitation Index: Percentage of silence/hesitation over total audio
        hesitation_index = (silence_duration / total_duration) * 100 if total_duration > 0 else 0
        
        # Vocal Variety (Proxy for Lexical Diversity)
        # We calculate the standard deviation of the spectral centroid
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        vocal_variety = np.std(centroids) / 1000.0 # Normalize roughly
        vocal_variety = min(max(vocal_variety, 0.1), 1.0) # Clip between 0.1 and 1.0
        
        # Heuristic: Higher hesitation + Lower variety = Higher cognitive load
        cognitive_load_score = min((hesitation_index * 1.5) + ((1 - vocal_variety) * 40), 100)
        
        return {
            "Hesitation Index": round(hesitation_index, 2),
            "Vocal Variety": round(vocal_variety, 2),
            "Cognitive Load Score": round(cognitive_load_score, 2),
            "signal": y,
            "sr": sr,
            "non_silent_intervals": non_silent_intervals
        }

class BehavioralTracker:
    """
    Tracks behavioral shifts using time-series analysis for early detection.
    """
    def __init__(self):
        pass

    def analyze_trends(self, scores):
        """
        scores: list of daily behavioral scores (1-10, where 1 is personality shift/apathy)
        """
        if len(scores) < 3:
            return "Veri Yetersiz", 0
        
        # Simple trend detection (Linear Regression slope proxy)
        x = np.arange(len(scores))
        y = np.array(scores)
        slope = np.polyfit(x, y, 1)[0]
        
        status = "Stabil"
        if slope < -0.2:
            status = "Düşüş Eğilimi (Riskli)"
        elif slope > 0.2:
            status = "İyileşme Eğilimi"
            
        return status, round(slope, 3)

class GeneticRiskProjector:
    """
    Models the progression of Alzheimer's based on APP gene over-expression.
    """
    def calculate_projection(self, current_age, app_factor=1.5):
        """
        app_factor: 1.5 is standard for T21 (3 copies of APP)
        """
        ages = np.arange(current_age, 81)
        # Clinical Model: T21 Alzheimer's risk often follows a Gompertz or Logistic distribution
        # Beta represents the acceleration of amyloid buildup
        beta = 0.15 * (app_factor / 1.5)
        inflection_point = 50 - (app_factor - 1.5) * 10 # Higher APP = Earlier onset
        
        risk_curve = 100 / (1 + np.exp(-beta * (ages - inflection_point)))
        
        return ages, np.round(risk_curve, 2)
