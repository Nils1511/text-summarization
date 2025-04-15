#!/usr/bin/env python3
"""
Evaluator module for text summarization.
Handles evaluation of summaries using various metrics.
"""
import logging
import os
from typing import Dict, List, Any, Tuple
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
nltk.download('punkt_tab')

class SummarizationEvaluator:
    """
    Class for evaluating text summarization models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SummarizationEvaluator with configuration.
        
        Args:
            config: Dictionary containing evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.metrics = config["evaluation"]["metrics"]
        self.logger.info(f"Initializing evaluator with metrics: {self.metrics}")
        
        # Create ROUGE scorer for evaluation
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)
        
        # Store evaluation results
        self.results = {}
        
    def evaluate(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Evaluate generated summaries against reference summaries.
        
        Args:
            references: List of reference (ground truth) summaries
            predictions: List of predicted (generated) summaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(references) != len(predictions):
            self.logger.error(f"Mismatched number of references ({len(references)}) and predictions ({len(predictions)})")
            raise ValueError("Number of references and predictions must match")
            
        self.logger.info(f"Evaluating {len(predictions)} summaries")
        
        # Calculate ROUGE scores for each pair
        scores = defaultdict(list)
        for ref, pred in zip(references, predictions):
            score = self.scorer.score(ref, pred)
            for metric in self.metrics:
                scores[metric].append(score[metric].fmeasure)
        
        # Calculate average scores
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        
        self.logger.info(f"Evaluation results: {json.dumps(avg_scores, indent=2)}")
        self.results = {"individual": scores, "average": avg_scores}
        
        return avg_scores
        
    def analyze_errors(self, references: List[str], predictions: List[str], dialogues: List[str]) -> Dict[str, Any]:
        """
        Perform error analysis on generated summaries.
        
        Args:
            references: List of reference (ground truth) summaries
            predictions: List of predicted (generated) summaries
            dialogues: List of input dialogues
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Performing error analysis on generated summaries")
        
        # Ensure we have evaluation scores
        if not self.results:
            self.evaluate(references, predictions)
            
        # Initialize analysis results
        analysis = {
            "length_analysis": self._analyze_length(references, predictions, dialogues),
            "content_analysis": self._analyze_content(references, predictions),
            "challenging_cases": self._find_challenging_cases(references, predictions, dialogues),
            "statistical_analysis": self._statistical_analysis(references, predictions)
        }
        
        self.logger.info("Error analysis completed")
        return analysis
        
    def _analyze_length(self, references: List[str], predictions: List[str], dialogues: List[str]) -> Dict[str, Any]:
        """
        Analyze the relationship between text length and performance.
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            dialogues: List of input dialogues
            
        Returns:
            Dictionary of length analysis results
        """
        # Calculate lengths
        dialogue_lengths = [len(text.split()) for text in dialogues]
        ref_lengths = [len(text.split()) for text in references]
        pred_lengths = [len(text.split()) for text in predictions]
        
        # Calculate compression ratio (summary length / dialogue length)
        compression_ratios = [ref_len / dial_len if dial_len > 0 else 0 
                             for ref_len, dial_len in zip(ref_lengths, dialogue_lengths)]
        
        # Calculate length ratio (predicted length / reference length)
        length_ratios = [pred_len / ref_len if ref_len > 0 else 0 
                        for pred_len, ref_len in zip(pred_lengths, ref_lengths)]
        
        return {
            "dialogue_lengths": {
                "mean": np.mean(dialogue_lengths),
                "median": np.median(dialogue_lengths),
                "min": np.min(dialogue_lengths),
                "max": np.max(dialogue_lengths)
            },
            "reference_lengths": {
                "mean": np.mean(ref_lengths),
                "median": np.median(ref_lengths),
                "min": np.min(ref_lengths),
                "max": np.max(ref_lengths)
            },
            "prediction_lengths": {
                "mean": np.mean(pred_lengths),
                "median": np.median(pred_lengths),
                "min": np.min(pred_lengths),
                "max": np.max(pred_lengths)
            },
            "compression_ratios": {
                "mean": np.mean(compression_ratios),
                "median": np.median(compression_ratios)
            },
            "length_ratios": {
                "mean": np.mean(length_ratios),
                "median": np.median(length_ratios)
            }
        }
        
    def _analyze_content(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """
        Analyze content differences between reference and predicted summaries.
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            
        Returns:
            Dictionary of content analysis results
        """
        # Tokenize summaries
        ref_tokens = [word_tokenize(ref.lower()) for ref in references]
        pred_tokens = [word_tokenize(pred.lower()) for pred in predictions]
        
        # Calculate content overlap
        overlap_scores = []
        missed_content = []
        added_content = []
        
        for ref_toks, pred_toks in zip(ref_tokens, pred_tokens):
            ref_set = set(ref_toks)
            pred_set = set(pred_toks)
            
            # Calculate Jaccard similarity (intersection / union)
            if len(ref_set.union(pred_set)) > 0:
                overlap = len(ref_set.intersection(pred_set)) / len(ref_set.union(pred_set))
            else:
                overlap = 0
                
            overlap_scores.append(overlap)
            
            # Find missed and added content
            missed = ref_set - pred_set
            added = pred_set - ref_set
            
            missed_content.append(missed)
            added_content.append(added)
            
        return {
            "content_overlap": {
                "mean": np.mean(overlap_scores),
                "median": np.median(overlap_scores),
                "min": np.min(overlap_scores),
                "max": np.max(overlap_scores)
            },
            "common_missed_words": self._find_common_elements(missed_content),
            "common_added_words": self._find_common_elements(added_content)
        }
        
    def _find_common_elements(self, list_of_sets: List[set]) -> List[Tuple[str, int]]:
        """
        Find most common elements across a list of sets.
        
        Args:
            list_of_sets: List of sets containing elements
            
        Returns:
            List of (element, count) tuples, sorted by count
        """
        counter = defaultdict(int)
        for s in list_of_sets:
            for elem in s:
                if not elem.isalnum():  # Skip punctuation
                    continue
                counter[elem] += 1
                
        # Sort by count (descending)
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:20]
        
    def _find_challenging_cases(self, references: List[str], predictions: List[str], dialogues: List[str]) -> List[Dict[str, Any]]:
        """
        Find examples where the model performed particularly poorly.
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            dialogues: List of input dialogues
            
        Returns:
            List of challenging cases with scores and analysis
        """
        # Calculate individual ROUGE scores
        scores = []
        for ref, pred in zip(references, predictions):
            rouge_scores = self.scorer.score(ref, pred)
            # Calculate average F1 across metrics
            avg_f1 = np.mean([rouge_scores[metric].fmeasure for metric in self.metrics])
            scores.append(avg_f1)
            
        # Find indices of lowest scoring examples
        worst_indices = np.argsort(scores)[:10]  # Top 10 worst examples
        
        challenging_cases = []
        for idx in worst_indices:
            case = {
                "dialogue": dialogues[idx],
                "reference": references[idx],
                "prediction": predictions[idx],
                "rouge_score": scores[idx],
                "dialogue_length": len(dialogues[idx].split()),
                "reference_length": len(references[idx].split()),
                "prediction_length": len(predictions[idx].split())
            }
            challenging_cases.append(case)
            
        return challenging_cases
        
    def _statistical_analysis(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """
        Perform statistical analysis on evaluation results.
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            
        Returns:
            Dictionary of statistical analysis results
        """
        # Ensure we have individual scores
        if not self.results or "individual" not in self.results:
            self.evaluate(references, predictions)
            
        stats = {}
        for metric in self.metrics:
            scores = np.array(self.results["individual"][metric])
            stats[metric] = {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "q1": np.percentile(scores, 25),
                "q3": np.percentile(scores, 75)
            }
            
        return stats
        
    def visualize_results(self, output_dir: str = "results") -> None:
        """
        Create visualizations of evaluation results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.results or not hasattr(self, "results") or "individual" not in self.results:
            self.logger.warning("No evaluation results available for visualization")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot ROUGE score distributions
        self._plot_score_distributions(output_dir)
        
        # If we have length analysis
        if hasattr(self, "analysis") and "length_analysis" in self.analysis:
            self._plot_length_analysis(output_dir)
            
    def _plot_score_distributions(self, output_dir: str) -> None:
        """
        Plot distributions of ROUGE scores.
        
        Args:
            output_dir: Directory to save visualizations
        """
        plt.figure(figsize=(10, 6))
        
        for i, metric in enumerate(self.metrics):
            scores = self.results["individual"][metric]
            plt.subplot(1, len(self.metrics), i+1)
            plt.hist(scores, bins=20, alpha=0.7)
            plt.axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            plt.title(f'{metric} Distribution')
            plt.xlabel('F1 Score')
            plt.ylabel('Frequency')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rouge_score_distributions.png'))
        plt.close()
        
    def _plot_length_analysis(self, output_dir: str) -> None:
        """
        Plot length analysis visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # This would create plots showing relationship between input/output lengths
        # and performance, if length analysis data is available
        pass
        
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save average metrics
        if self.results and "average" in self.results:
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(self.results["average"], f, indent=2)
                
        # Save full analysis if available
        if hasattr(self, "analysis"):
            # Convert complex objects to serializable format
            analysis_copy = {}
            for key, value in self.analysis.items():
                if key == "challenging_cases":
                    analysis_copy[key] = value  # Already serializable
                else:
                    # Convert numpy values to Python native types
                    analysis_copy[key] = self._make_serializable(value)
                    
            with open(os.path.join(output_dir, "analysis.json"), "w") as f:
                json.dump(analysis_copy, f, indent=2)
                
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj