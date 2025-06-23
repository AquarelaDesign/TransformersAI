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
    
    def get_previous_configs(self):
        """Retorna lista de configurações de treinamentos anteriores"""
        try:
            configs = []
            training_data_dir = 'training_data'
            
            if os.path.exists(training_data_dir):
                for filename in os.listdir(training_data_dir):
                    if (filename.startswith('training_config_') or filename.startswith('retrain_config_')) and filename.endswith('.json'):
                        file_path = os.path.join(training_data_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                
                            # Extrair informações relevantes
                            config_info = {
                                'filename': filename,
                                'timestamp': config.get('timestamp', 'N/A'),
                                'base_model': config.get('base_model', 'N/A'),
                                'epochs': config.get('epochs', 'N/A'),
                                'total_texts': config.get('total_texts', 0),
                                'total_sources': config.get('total_sources', 0),
                                'sources_preview': config.get('web_sources', [])[:3],
                                'has_email': bool(config.get('email_config')),
                                'file_size': os.path.getsize(file_path),
                                'is_retrain': filename.startswith('retrain_config_')
                            }
                            configs.append(config_info)
                        except Exception as e:
                            print(f"Erro ao ler {filename}: {e}")
            
            # Ordenar por timestamp (mais recente primeiro)
            configs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return configs
            
        except Exception as e:
            print(f"Erro ao buscar configurações: {e}")
            return []
