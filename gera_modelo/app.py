from flask import Flask
import os
import logging
import warnings

# Configurar logging e suprimir avisos desnecessários
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*loss_type.*")

def create_gera_modelo_app():
    """Cria e configura a aplicação Flask do sistema de treinamento"""
    
    # Inicializar componentes
    from .core.config_manager import ConfigManager
    from .core.data_collector import DataCollector
    from .core.model_trainer import ModelTrainer
    from .services.training_service import TrainingService
    from .services.chat_service import ChatService
    from .services.data_service import DataService
    from .routes.main_routes import register_main_routes
    from .routes.training_routes import register_training_routes
    from .routes.data_routes import register_data_routes
    
    # Criar instâncias dos componentes
    config_manager = ConfigManager()
    data_collector = DataCollector(config_manager)
    model_trainer = ModelTrainer(config_manager)
    
    # Criar serviços
    training_service = TrainingService(config_manager, data_collector, model_trainer)
    chat_service = ChatService(config_manager, data_collector, model_trainer)
    data_service = DataService(config_manager, data_collector)
    
    # Dicionário de serviços para passar para as rotas
    services = {
        'config_manager': config_manager,
        'data_collector': data_collector,
        'model_trainer': model_trainer,
        'training_service': training_service,
        'chat_service': chat_service,
        'data_service': data_service
    }
    
    # Criar aplicação Flask
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Registrar rotas
    register_main_routes(app, services)
    register_training_routes(app, services)
    register_data_routes(app, services)
    
    return app, services

if __name__ == '__main__':
    app, services = create_gera_modelo_app()
    app.run(debug=True)
