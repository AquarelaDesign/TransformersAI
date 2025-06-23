"""
Gera Modelo - Versão modular
Arquivo principal que utiliza a estrutura modular
"""

from gera_modelo.app import create_gera_modelo_app

if __name__ == '__main__':
    # Criar aplicação usando a estrutura modular
    app, services = create_gera_modelo_app()
    
    try:
        print("=== GERA MODELO - SISTEMA MODULAR ===")
        print("Acesse: http://localhost:5000")
        print("Estrutura modular carregada com sucesso!")
        print("\nComponentes carregados:")
        print(f"  - ConfigManager: {services['config_manager'].__class__.__name__}")
        print(f"  - DataCollector: {services['data_collector'].__class__.__name__}")
        print(f"  - ModelTrainer: {services['model_trainer'].__class__.__name__}")
        print(f"  - TrainingService: {services['training_service'].__class__.__name__}")
        print(f"  - ChatService: {services['chat_service'].__class__.__name__}")
        print(f"  - DataService: {services['data_service'].__class__.__name__}")
        
        app.run(debug=True)
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
