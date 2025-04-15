#!/usr/bin/env python3
"""
Data loader module for text summarization.
Handles loading, preprocessing, and batching of dataset.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from dialogue_processor import DialogueProcessor 

class DataLoader:
    """
    Class for loading and preprocessing data for summarization tasks.
    """
    
    def __init__(self, config: Dict, tokenizer: PreTrainedTokenizer):
        """
        Initialize DataLoader with configuration and tokenizer.
        
        Args:
            config: Dictionary containing dataset configuration
            tokenizer: Pretrained tokenizer for the model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        self.dataset_name = config["dataset"]["name"]
        self.train_split = config["dataset"]["train_split"]
        self.val_split = config["dataset"]["validation_split"]
        self.test_split = config["dataset"]["test_split"]
        
        self.max_input_length = config["model"]["max_input_length"]
        self.max_output_length = config["model"]["max_output_length"]
        
        self.logger.info(f"Initializing DataLoader for dataset: {self.dataset_name}")
        self.dataset = None
        
    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Load dataset from Hugging Face datasets library.
        
        Returns:
            Dictionary containing train, validation, and test splits
        """
        self.logger.info(f"Loading dataset: {self.dataset_name}")
        try:
            dataset = load_dataset(self.dataset_name, trust_remote_code=True)
            self.logger.info(f"Dataset loaded successfully with splits: {list(dataset.keys())}")
            
            self.dataset = {
                "train": dataset[self.train_split],
                "validation": dataset[self.val_split],
                "test": dataset[self.test_split]
            }
            
            # Log dataset statistics
            for split, data in self.dataset.items():
                self.logger.info(f"{split} split size: {len(data)}")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def preprocess_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Preprocess data by tokenizing inputs and outputs.
        
        Args:
            examples: Dictionary of examples with 'dialogue' and 'summary' fields
            
        Returns:
            Dictionary of tokenized inputs and outputs
        """
        # Tokenize dialogues (input)
        inputs = self.tokenizer(
            examples["dialogue"],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        
        # Tokenize summaries (output)
        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(
                examples["summary"],
                padding="max_length",
                truncation=True,
                max_length=self.max_output_length,
                return_tensors="pt"
            )
        
        # Replace padding token id with -100 so it's ignored in loss calculation
        outputs_ids = outputs["input_ids"].masked_fill(
            outputs["input_ids"] == self.tokenizer.pad_token_id,
            -100
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": outputs_ids,
            # Keep original text for evaluation
            "dialogue": examples["dialogue"],
            "summary": examples["summary"]
        }
    
    def prepare_dataset(self) -> Dict[str, Dataset]:
        """
        Apply preprocessing to all dataset splits.
        
        Returns:
            Dictionary containing preprocessed train, validation, and test splits
        """
        if self.dataset is None:
            self.load_dataset()
        
        self.logger.info("Preprocessing dataset for model input")
        
        processed_dataset = {}
        for split, data in self.dataset.items():
            self.logger.info(f"Preprocessing {split} split")
            processed_dataset[split] = data.map(
                self.preprocess_function,
                batched=True,
                remove_columns=data.column_names
            )
            
            # Log sample counts
            self.logger.info(f"Processed {split} split size: {len(processed_dataset[split])}")
        
        return processed_dataset

    def preprocess_dialogue(self, dialogue):
        """Process dialogue text to highlight speaker transitions"""
        # Convert irregular separators to consistent format
        # processed = dialogue.replace("\r\n", "\n")
        
        # # Add special tokens around speaker names
        # lines = processed.split("\n")
        # structured_dialogue = []
        
        # for line in lines:
        #     if ":" in line:
        #         parts = line.split(":", 1)
        #         if len(parts) == 2:
        #             speaker, content = parts
        #             # Add special tokens to highlight speaker changes
        #             structured_line = f"<speaker>{speaker.strip()}</speaker>: {content.strip()}"
        #             structured_dialogue.append(structured_line)
        #     else:
        #         structured_dialogue.append(line)
                
        # return "\n".join(structured_dialogue)

        # structured = DialogueProcessor().create_structured_context(dialogue, DialogueProcessor().extract_speakers(dialogue))
        # return structured

        processor = DialogueProcessor()
        speakers = processor.extract_speakers(dialogue)
        context = processor.create_structured_context(dialogue, speakers)
        
        # Prompt-style framing
        prompt = f"Summarize the following conversation between {', '.join(speakers)}:\n\n{context}"
        return prompt


    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Apply preprocessing to a batch of examples.
        
        Args:
            examples: Dictionary of examples with 'dialogue' and 'summary' fields
            
        Returns:
            Dictionary of preprocessed examples
        """
        examples["dialogue"] = [self.preprocess_dialogue(d) for d in examples["dialogue"]]
        return self.preprocess_data(examples)
    
    def get_sample(self, split: str = "test", n: int = 5) -> List[Dict[str, str]]:
        """
        Get a sample of examples from the dataset for inspection.
        
        Args:
            split: Dataset split to sample from ('train', 'validation', 'test')
            n: Number of examples to sample
            
        Returns:
            List of sampled examples with 'dialogue' and 'summary' fields
        """
        if self.dataset is None:
            self.load_dataset()
            
        samples = []
        data = self.dataset[split]
        indices = np.random.choice(len(data), min(n, len(data)), replace=False)
        
        for idx in indices:
            samples.append({
                "dialogue": data[idx]["dialogue"],
                "summary": data[idx]["summary"]
            })
            
        return samples
    
    def get_data_loader(self, split: str, batch_size: Optional[int] = None) -> Dataset:
        """
        Get a PyTorch DataLoader for the specified split.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            batch_size: Batch size for DataLoader, uses config value if not specified
            
        Returns:
            PyTorch DataLoader for the specified split
        """
        if batch_size is None:
            batch_size = self.config["model"]["batch_size"]
            
        if self.dataset is None:
            self.prepare_dataset()
            
        return self.dataset[split].with_format("torch")