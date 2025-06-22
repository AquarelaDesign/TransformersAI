import json
import os
from datetime import datetime

class ConfigManager:
    def __init__(self):
        self.config_dir = 'config'
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_training_config(self, config):
        """Salva configuração de treinamento"""
        filename = f"{self.config_dir}/training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def load_config(self, filename):
        """Carrega configuração de arquivo"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_available_configs(self):
        """Lista configurações disponíveis"""
        configs = []
        
        if os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    configs.append(filename)
        
        return configs
