# ...existing code from data_collector.py...
# (Mover o código existente do data_collector.py para aqui)

import os
import json
from data_collector import DataCollector as BaseDataCollector

class DataCollector(BaseDataCollector):
    """
    Versão estendida do DataCollector com funcionalidades específicas
    para o sistema modular
    """
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.config_manager = config_manager
    
    def get_collected_data_info(self):
        """Retorna informações sobre os dados coletados"""
        try:
            info = {
                'total_files': 0,
                'total_texts': 0,
                'sources': []
            }
            
            # Verificar dados nos diretórios de treinamento
            training_data_dir = 'training_data'
            if os.path.exists(training_data_dir):
                for filename in os.listdir(training_data_dir):
                    if filename.startswith('collected_data_') and filename.endswith('.json'):
                        info['total_files'] += 1
                        try:
                            with open(os.path.join(training_data_dir, filename), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'texts' in data:
                                    info['total_texts'] += len(data['texts'])
                                if 'sources' in data:
                                    for source in data['sources']:
                                        if source not in info['sources']:
                                            info['sources'].append(source)
                        except Exception as e:
                            print(f"Erro ao ler {filename}: {e}")
                            continue
            
            # Também verificar diretório data (compatibilidade com versões antigas)
            if os.path.exists('data'):
                for filename in os.listdir('data'):
                    if filename.endswith('.json'):
                        info['total_files'] += 1
                        try:
                            with open(os.path.join('data', filename), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                info['total_texts'] += len(data)
                                
                                # Contar fontes
                                for item in data:
                                    source = item.get('source', 'Desconhecido')
                                    if source not in info['sources']:
                                        info['sources'].append(source)
                        except:
                            continue
            
            return info
        except Exception as e:
            return {'error': str(e)}
