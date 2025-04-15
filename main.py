#!/usr/bin/env python3
"""
Main script for running the text summarization pipeline.
"""
import argparse
import logging
import os
import sys
import yaml
from typing import Dict, Any

from pipeline import SummarizationPipeline

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Dictionary containing logging configuration
    """
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", None)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Text Summarization Pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["evaluate", "tune", "summarize"], 
        default="evaluate", 
        help="Mode of operation"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "validation", "test"], 
        default="test", 
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None, 
        help="Number of samples to process"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results", 
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        default=None, 
        help="Text to summarize (for summarize mode)"
    )
    
    return parser.parse_args()

def main() -> None:
    """
    Main function to run the summarization pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting text summarization pipeline in {args.mode} mode")
    
    # Initialize pipeline
    pipeline = SummarizationPipeline(config)
    
    # Run appropriate mode
    if args.mode == "evaluate":
        logger.info(f"Evaluating on {args.split} split with {args.samples or 'all'} samples")
        results = pipeline.run_summarization(
            split=args.split,
            num_samples=args.samples
        )
        
        # Save results
        pipeline.save_results(results, output_dir=args.output_dir)
        
    elif args.mode == "tune":
        logger.info("Running hyperparameter tuning")
        tuning_results = pipeline.run_hyperparameter_tuning(
            split=args.split,
            num_samples=args.samples or 100
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        with open(os.path.join(args.output_dir, "tuning_results.json"), "w") as f:
            json.dump(tuning_results["best_params"], f, indent=2)
            
        logger.info(f"Best parameters: {tuning_results['best_params']}")
        logger.info(f"Best ROUGE-1 score: {tuning_results['best_score']:.4f}")
        
    elif args.mode == "summarize":
        if args.text:
            text = args.text
        else:
            # Read from stdin if no text provided
            logger.info("Reading text from stdin")
            text = sys.stdin.read().strip()
            
        if not text:
            logger.error("No text provided for summarization")
            sys.exit(1)
            
        # Generate summary
        summary = pipeline.summarize_text(text)
        print("\nSummary:")
        print("-" * 40)
        print(summary)
        print("-" * 40)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()