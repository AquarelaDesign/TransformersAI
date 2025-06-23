"""
Chat Widget - Versão modular
Arquivo principal que utiliza a estrutura modular
"""

from chat_widget.app import create_chat_app

if __name__ == '__main__':
    # Criar aplicação usando a estrutura modular
    app, chatbot = create_chat_app()
    
    try:
        print("=== CHAT WIDGET MODULAR ===")
        print("Acesse: http://localhost:5001/chat")
        print("Admin: http://localhost:5001/admin")
        print("CORS configurado para localhost:5000 e localhost:5001")
        print(f"Status do modelo: {'Carregado' if chatbot.model_loaded else 'Modo fallback'}")
        print("Estrutura modular carregada com sucesso!")
        
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
