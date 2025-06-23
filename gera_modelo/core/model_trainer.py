# ...existing code from model_trainer.py...
# (Mover o código existente do model_trainer.py para aqui)

from model_trainer import ModelTrainer as BaseModelTrainer

class ModelTrainer(BaseModelTrainer):
    """
    Versão estendida do ModelTrainer com funcionalidades específicas
    para o sistema modular
    """
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.config_manager = config_manager
