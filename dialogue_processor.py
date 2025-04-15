#!/usr/bin/env python3
"""
Specialized processor for dialogue summarization tasks.
"""
import re
from typing import Dict, List, Any, Optional

class DialogueProcessor:
    """Class for dialogue-specific processing logic"""
    
    def __init__(self):
        self.speaker_pattern = re.compile(r"([^:]+?):")
    
    def extract_speakers(self, dialogue):
        """Extract unique speakers from dialogue"""
        return list(set(self.speaker_pattern.findall(dialogue)))
    
    def extract_key_actions(self, dialogue, speakers):
        """Extract key actions for each speaker"""
        actions = {speaker: [] for speaker in speakers}
        lines = dialogue.split("\n")
        
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    speaker, content = parts[0].strip(), parts[1].strip()
                    if speaker in actions:
                        # Look for action verbs and significant content
                        actions[speaker].append(content)
        
        return actions
    
    def analyze_dialogue(self, dialogue):
        """Extract structured information from dialogue"""
        speakers = self.extract_speakers(dialogue)
        actions = self.extract_key_actions(dialogue, speakers)
        
        # Basic dialogue stats
        return {
            "speakers": speakers,
            "speaker_actions": actions,
            "num_turns": len(dialogue.split("\n")),
            "structured_context": self.create_structured_context(dialogue, speakers)
        }
    
    def create_structured_context(self, dialogue, speakers):
        """Create structured context to help the model"""
        context = f"This is a conversation between {', '.join(speakers)}.\n\n"
        context += dialogue
        return context