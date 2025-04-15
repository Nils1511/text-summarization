#!/usr/bin/env python3
"""
Model module for text summarization.
Handles loading and configuring the summarization model.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Union
import re

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

class SummarizationModel:
    """
    Class for loading and configuring the summarization model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SummarizationModel with configuration.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.model_name = config["model"]["name"]
        self.device_name = config["model"]["device"]
        self.device = torch.device(self.device_name if torch.cuda.is_available() and self.device_name == "cuda" else "cpu")
        
        if self.device.type == "cuda":
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU for inference")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer from Hugging Face.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading model: {self.model_name}")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Handle cases where the model doesn't have pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.sep_token
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            self.logger.info(f"Model loaded successfully: {self.model.__class__.__name__}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1000000:.2f}M")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    def clean_generated_text(self, text):
        """Clean up artifacts in generated text"""
        # Remove strange characters and patterns seen in the outputs
        clean = text.replace("\ufffd", "")
        clean = re.sub(r"[=-]{5,}", "", clean)
        clean = re.sub(r"[@]{5,}", "", clean)
        clean = re.sub(r"[\u25ac]{5,}", "", clean)
        
        # Fix truncated sentences
        if not clean.endswith((".", "!", "?")):
            clean = re.sub(r"\s+\S+$", "", clean)
            if not clean.endswith((".", "!", "?")):
                clean += "."
        
        return clean.strip()
    def generate_summary(
        self, 
        text: Union[str, List[str]], 
        max_length: Optional[int] = None, 
        min_length: Optional[int] = None,
        num_beams: int = 4
    ) -> List[str]:
        """
        Generate summaries for the given text(s).
        
        Args:
            text: Input text or list of texts to summarize
            max_length: Maximum length of generated summary (default: from config)
            min_length: Minimum length of generated summary (default: 1/4 of max_length)
            num_beams: Number of beams for beam search (default: 4)
            
        Returns:
            List of generated summaries
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        if isinstance(text, str):
            text = [text]
            
        if max_length is None:
            max_length = self.config["model"]["max_output_length"]
            
        if min_length is None:
            min_length = max(10, max_length // 4)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config["model"]["max_input_length"],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summaries
        self.logger.info(f"Generating summaries for {len(text)} inputs with beam_size={num_beams}")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,  # Encourage slightly longer summaries
                no_repeat_ngram_size=3,  # Avoid repetition
                early_stopping=True,
                # Add these new parameters:
                do_sample=True,  # Enable sampling
                top_k=50,  # Top-k sampling
                top_p=0.95,  # Nucleus sampling
            )
            
        # Decode generated token IDs to text
        generated_summaries = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        generated_summaries = [self.clean_generated_text(summary) for summary in generated_summaries]
        
        return generated_summaries
        
    def save_model(self, path: str) -> None:
        """
        Save model and tokenizer to disk.
        
        Args:
            path: Directory path to save model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not loaded, nothing to save")
            return
            
        try:
            os.makedirs(path, exist_ok=True)
            
            self.logger.info(f"Saving model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            self.logger.info(f"Model and tokenizer saved successfully to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise