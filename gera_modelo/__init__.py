"""
Gera Modelo - Sistema de Treinamento de Modelos de IA
Estrutura modular para facilitar manutenção
"""

from .app import create_gera_modelo_app
from .core import *
from .routes import *
from .services import *

__version__ = "1.0.0"
__author__ = "TransformersAI"

# Instâncias globais (serão inicializadas no app.py)
config_manager = None
data_collector = None
model_trainer = None

def initialize_components():
    """Inicializa as instâncias globais dos componentes"""
    global config_manager, data_collector, model_trainer
    
    if config_manager is None:
        from .core.config_manager import ConfigManager
        from .core.data_collector import DataCollector
        from .core.model_trainer import ModelTrainer
        
        config_manager = ConfigManager()
        data_collector = DataCollector(config_manager)
        model_trainer = ModelTrainer(config_manager)
    
    return config_manager, data_collector, model_trainer
