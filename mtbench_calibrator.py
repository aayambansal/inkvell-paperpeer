#!/usr/bin/env python3
"""
MT-Bench Calibration System for Unbiased Peer Review
"""

import json
import numpy as np
from collections import defaultdict
import os

class MTBenchCalibrator:
    def __init__(self, mtbench_path="data/mt_bench"):
        self.mtbench_path = mtbench_path
        self.model_rankings = {}
        self.judgment_patterns = {}
        
    def load_mtbench_data(self):
        """Load MT-Bench judgment data"""
        judgment_data = []
        judgment_dir = os.path.join(self.mtbench_path, "model_judgment")
        
        if not os.path.exists(judgment_dir):
            return judgment_data
            
        for filename in os.listdir(judgment_dir):
            if filename.endswith("_pair.jsonl"):
                filepath = os.path.join(judgment_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            judgment_data.append(json.loads(line))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return judgment_data
    
    def calculate_win_rates(self, judgment_data):
        """Calculate win rates for each model"""
        model_wins = defaultdict(int)
        model_total = defaultdict(int)
        
        for judgment in judgment_data:
            if 'winner' in judgment and 'model_a' in judgment and 'model_b' in judgment:
                model_a = judgment['model_a']
                model_b = judgment['model_b']
                winner = judgment['winner']
                
                model_total[model_a] += 1
                model_total[model_b] += 1
                
                if winner == 'model_a':
                    model_wins[model_a] += 1
                elif winner == 'model_b':
                    model_wins[model_b] += 1
        
        win_rates = {}
        for model in model_total:
            if model_total[model] > 0:
                win_rates[model] = model_wins[model] / model_total[model]
        
        return win_rates
    
    def generate_calibration_scores(self, win_rates):
        """Generate calibration scores based on win rates"""
        if not win_rates:
            return {}
            
        # Convert win rates to 1-5 scale
        min_rate = min(win_rates.values())
        max_rate = max(win_rates.values())
        
        calibration_scores = {}
        for model, rate in win_rates.items():
            if max_rate > min_rate:
                normalized = (rate - min_rate) / (max_rate - min_rate)
                score = 1 + (normalized * 4)  # 1-5 scale
            else:
                score = 3.0  # Default if all equal
            calibration_scores[model] = score
        
        return calibration_scores
    
    def get_unbiased_thresholds(self, win_rates):
        """Calculate statistical thresholds for scoring"""
        if not win_rates:
            return {
                "mean": 4.0,
                "std": 0.5,
                "top_10_percent": 4.5,
                "bottom_10_percent": 3.5
            }
            
        rates = list(win_rates.values())
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        # Map to 1-5 scale with mean at 4.0
        mean_score = 4.0
        std_score = 0.5
        
        top_10 = np.percentile(rates, 90)
        bottom_10 = np.percentile(rates, 10)
        
        # Convert percentiles to score scale
        if len(rates) > 1:
            range_rate = max(rates) - min(rates)
            if range_rate > 0:
                top_10_score = mean_score + ((top_10 - mean_rate) / range_rate) * 2
                bottom_10_score = mean_score + ((bottom_10 - mean_rate) / range_rate) * 2
            else:
                top_10_score = mean_score + 0.5
                bottom_10_score = mean_score - 0.5
        else:
            top_10_score = mean_score + 0.5
            bottom_10_score = mean_score - 0.5
        
        return {
            "mean": mean_score,
            "std": std_score,
            "top_10_percent": max(top_10_score, mean_score + 0.3),
            "bottom_10_percent": min(bottom_10_score, mean_score - 0.3)
        }
    
    def get_calibration_scores(self):
        """Get complete calibration data"""
        print("🔍 Loading MT-Bench data for calibration...")
        
        judgment_data = self.load_mtbench_data()
        if not judgment_data:
            print("⚠️ No MT-Bench data found, using defaults")
            return self._get_default_calibration()
        
        win_rates = self.calculate_win_rates(judgment_data)
        calibration_scores = self.generate_calibration_scores(win_rates)
        thresholds = self.get_unbiased_thresholds(win_rates)
        
        print(f"📊 Analyzed {len(win_rates)} models from MT-Bench")
        
        # Show top models
        sorted_models = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        print("🏆 Top 5 Models by Win Rate:")
        for i, (model, rate) in enumerate(sorted_models, 1):
            print(f"   {i}. {model}: {rate:.3f}")
        
        print("📈 Calibration Statistics:")
        print(f"   Mean Score: {thresholds['mean']:.2f}")
        print(f"   Std Dev: {thresholds['std']:.2f}")
        print(f"   Top 10%: {thresholds['top_10_percent']:.2f}+")
        print(f"   Bottom 10%: {thresholds['bottom_10_percent']:.2f}-")
        
        return {
            "model_scores": calibration_scores,
            "win_rates": win_rates,
            "thresholds": thresholds,
            "distribution": {
                "mean": thresholds["mean"],
                "std": thresholds["std"],
                "range": [thresholds["bottom_10_percent"], thresholds["top_10_percent"]]
            }
        }
    
    def _get_default_calibration(self):
        """Default calibration when MT-Bench data unavailable"""
        return {
            "model_scores": {},
            "win_rates": {},
            "thresholds": {
                "mean": 4.0,
                "std": 0.5,
                "top_10_percent": 4.5,
                "bottom_10_percent": 3.5
            },
            "distribution": {
                "mean": 4.0,
                "std": 0.5,
                "range": [3.5, 4.5]
            }
        }

if __name__ == "__main__":
    calibrator = MTBenchCalibrator()
    scores = calibrator.get_calibration_scores()
    print(f"\nCalibration complete: {len(scores['model_scores'])} models processed")
