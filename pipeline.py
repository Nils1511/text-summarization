#!/usr/bin/env python3
"""
Pipeline module for text summarization.
Orchestrates the data flow and summarization process.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import time

import torch
from tqdm import tqdm

from data_loader import DataLoader
from model import SummarizationModel
from evaluator import SummarizationEvaluator
from dialogue_processor import DialogueProcessor

class SummarizationPipeline:
    """
    Class for orchestrating the text summarization pipeline.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SummarizationPipeline with configuration.
        
        Args:
            config: Dictionary containing configuration for all components
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing summarization pipeline")
        
        # Initialize model
        self.model = SummarizationModel(config)
        self.model_name = config["model"]["name"]
        
        # Get device
        self.device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model_obj, self.tokenizer = self.model.load_model()
        
        # Initialize data loader with tokenizer
        self.data_loader = DataLoader(config, self.tokenizer)
        
        # Initialize evaluator
        self.evaluator = SummarizationEvaluator(config)

        self.dialogue_processor = DialogueProcessor()
        
        self.logger.info("Pipeline initialized successfully")
        
    def run_summarization(self, split: str = "test", num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run summarization on the specified dataset split.
        
        Args:
            split: Dataset split to use ('train', 'validation', 'test')
            num_samples: Number of samples to process (None for all)
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Running summarization pipeline on {split} split")
        
        # Load dataset if not already loaded
        if not hasattr(self.data_loader, 'dataset') or self.data_loader.dataset is None:
            self.data_loader.load_dataset()
        
        # Get dataset
        dataset = self.data_loader.dataset[split]
        
        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        self.logger.info(f"Processing {len(dataset)} examples")
        
        # Process in batches
        batch_size = self.config["model"]["batch_size"]
        dialogues = []
        references = []
        predictions = []
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset[i:i+batch_size]
            
            # Get dialogues and reference summaries
            batch_dialogues = batch["dialogue"]
            batch_references = batch["summary"]
            
            # Generate predictions
            batch_predictions = self.model.generate_summary(batch_dialogues)
            
            # Store results
            dialogues.extend(batch_dialogues)
            references.extend(batch_references)
            predictions.extend(batch_predictions)
            
        # Evaluate results
        self.logger.info("Evaluating results")
        metrics = self.evaluator.evaluate(references, predictions)
        
        # Analyze errors
        self.logger.info("Analyzing errors")
        analysis = self.evaluator.analyze_errors(references, predictions, dialogues)
        self.evaluator.analysis = analysis  # Store for later
        
        # Create examples
        examples = self._create_examples(dialogues, references, predictions)
        
        results = {
            "metrics": metrics,
            "analysis": analysis,
            "examples": examples
        }
        
        self.logger.info(f"Pipeline completed with average ROUGE-1: {metrics['rouge1']:.4f}")
        return results
    
    def _create_examples(self, dialogues: List[str], references: List[str], predictions: List[str], n: int = 10) -> List[Dict[str, str]]:
        """
        Create examples for qualitative analysis.
        
        Args:
            dialogues: List of input dialogues
            references: List of reference summaries
            predictions: List of generated summaries
            n: Number of examples to create
            
        Returns:
            List of example dictionaries
        """
        # Select a sample of indices
        import random
        indices = random.sample(range(len(dialogues)), min(n, len(dialogues)))
        
        examples = []
        for i in indices:
            examples.append({
                "dialogue": dialogues[i],
                "reference": references[i],
                "prediction": predictions[i]
            })
            
        return examples
        
    def save_results(self, results: Dict[str, Any], output_dir: str = "results") -> None:
        """
        Save results to disk.
        
        Args:
            results: Dictionary containing results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)
            
        # Save examples
        examples_path = os.path.join(output_dir, "examples.json")
        with open(examples_path, "w") as f:
            json.dump(results["examples"], f, indent=2)
            
        # Save full results (excluding complex objects)
        full_results = {
            "metrics": results["metrics"],
            "examples": results["examples"],
            "model_name": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        full_path = os.path.join(output_dir, "results.json")
        with open(full_path, "w") as f:
            json.dump(full_results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_dir}")
        
        # Let evaluator save its results as well
        self.evaluator.save_results(output_dir)
        self.evaluator.visualize_results(output_dir)
        
    def summarize_text(self, text):
        # For dialogue texts, use specialized processing
        if ":" in text and "\n" in text:  # Simple heuristic to detect dialogues
            dialogue_info = self.dialogue_processor.analyze_dialogue(text)
            # Use structured context for better results
            text_to_summarize = dialogue_info["structured_context"]
        else:
            text_to_summarize = text
            
        return self.model.generate_summary(text_to_summarize)[0]
        
    def run_hyperparameter_tuning(self, 
                                  split: str = "validation", 
                                  num_samples: int = 100,
                                  param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Run basic hyperparameter tuning for summarization.
        
        Args:
            split: Dataset split to use for tuning
            num_samples: Number of samples to use
            param_grid: Dictionary of parameters to tune
            
        Returns:
            Dictionary containing best parameters and results
        """
        self.logger.info("Running hyperparameter tuning")
        
        if param_grid is None:
            # Default parameters to tune
            param_grid = {
                "num_beams": [2, 4, 6],
                "min_length": [10, 15, 20],
                "max_length": [50, 100]
            }
            
        # Load dataset if not already loaded
        if not hasattr(self.data_loader, 'dataset') or self.data_loader.dataset is None:
            self.data_loader.load_dataset()
            
        # Get dataset for tuning
        dataset = self.data_loader.dataset[split]
        
        # Limit samples
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            
        dialogues = dataset["dialogue"]
        references = dataset["summary"]
        
        # Track best parameters and score
        best_score = -1
        best_params = {}
        best_predictions = []
        results = []
        
        # Generate all combinations of parameters
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
            self.logger.info(f"Trying parameters: {param_str}")
            
            # Check if max_length < min_length
            if "max_length" in param_dict and "min_length" in param_dict:
                if param_dict["max_length"] <= param_dict["min_length"]:
                    continue
            
            # Generate summaries with these parameters
            predictions = self.model.generate_summary(
                dialogues, 
                max_length=param_dict.get("max_length", None),
                min_length=param_dict.get("min_length", None),
                num_beams=param_dict.get("num_beams", 4)
            )
            
            # Evaluate
            metrics = self.evaluator.evaluate(references, predictions)
            
            # Use ROUGE-1 as primary metric for tuning
            rouge1 = metrics["rouge1"]
            
            # Track results
            current_result = {
                "params": param_dict,
                "metrics": metrics
            }
            results.append(current_result)
            
            # Update best if better
            if rouge1 > best_score:
                best_score = rouge1
                best_params = param_dict
                best_predictions = predictions
                self.logger.info(f"New best score: {best_score:.4f} with params: {param_str}")
                
        # Final evaluation with best parameters
        self.logger.info(f"Best parameters found: {best_params}")
        self.logger.info(f"Best ROUGE-1 score: {best_score:.4f}")
        
        # Analyze errors for best predictions
        analysis = self.evaluator.analyze_errors(references, best_predictions, dialogues)
        
        tuning_results = {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
            "analysis": analysis
        }
        
        return tuning_results