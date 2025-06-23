from flask import Flask
from flask_cors import CORS
import atexit
import torch
from .chatbot import ChatBot
from .routes.chat_routes import register_chat_routes
from .routes.admin_routes import register_admin_routes
from .routes.api_routes import register_api_routes
from data_collector import DataCollector

def create_chat_app():
    """Cria e configura a aplicação Flask do chat"""
    
    # Inicializar chatbot
    chatbot = ChatBot()
    
    # Tentar carregar modelo
    print("Iniciando sistema de chat...")
    try:
        if not chatbot.load_latest_model():
            print("Carregamento direto falhou. Tentando em background...")
            chatbot.try_load_model_async()
    except Exception as e:
        print(f"Erro na inicialização do modelo: {e}")
        print("Sistema funcionará apenas com respostas padrão.")
    
    # Criar aplicação Flask
    app = Flask(__name__, 
                template_folder='../chat_templates',
                static_folder='../chat_static')
    
    # Configurar CORS
    CORS(app, origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5001",
        "http://127.0.0.1:5001"
    ])
    
    # Instanciar coletor de dados
    data_collector = DataCollector()
    
    # Registrar rotas
    register_chat_routes(app, chatbot)
    register_admin_routes(app, chatbot)
    register_api_routes(app, chatbot)
    
    # Função de limpeza
    def cleanup_on_exit():
        """Limpa recursos antes de sair"""
        try:
            print("Limpando recursos...")
            if hasattr(chatbot, 'model') and chatbot.model is not None:
                del chatbot.model
            if hasattr(chatbot, 'tokenizer') and chatbot.tokenizer is not None:
                del chatbot.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Limpeza concluída.")
        except Exception as e:
            print(f"Erro na limpeza: {e}")
    
    # Registrar função de limpeza
    atexit.register(cleanup_on_exit)
    
    return app, chatbot

if __name__ == '__main__':
    app, chatbot = create_chat_app()
    
    try:
        print("Iniciando Chat Widget...")
        print("Acesse: http://localhost:5001/chat")
        print("Admin: http://localhost:5001/admin")
        print("CORS configurado para localhost:5000 e localhost:5001")
        print(f"Status do modelo: {'Carregado' if chatbot.model_loaded else 'Modo fallback'}")
        
        app.run(
            debug=False,
            port=5001, 
            host='0.0.0.0',
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
