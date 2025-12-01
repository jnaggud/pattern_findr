#!/usr/bin/env python3
"""
API Configuration Management for Pattern_FindR
Handles secure storage and retrieval of API keys.
"""

import os
import json
from typing import Optional

class APIConfig:
    def __init__(self):
        self.config_file = "api_config.json"
        self.config_data = self._load_config()
    
    def _load_config(self) -> dict:
        """Load API configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_config(self):
        """Save API configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving API config: {e}")
            return False
    
    def get_openai_key(self) -> Optional[str]:
        """Get stored OpenAI API key."""
        return self.config_data.get('openai_api_key')
    
    def set_openai_key(self, api_key: str) -> bool:
        """Store OpenAI API key."""
        if api_key and api_key.startswith('sk-'):
            self.config_data['openai_api_key'] = api_key
            return self._save_config()
        return False
    
    def clear_openai_key(self) -> bool:
        """Clear stored OpenAI API key."""
        if 'openai_api_key' in self.config_data:
            del self.config_data['openai_api_key']
            return self._save_config()
        return False
    
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is stored."""
        key = self.get_openai_key()
        return key is not None and len(key) > 0
