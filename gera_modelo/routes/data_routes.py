from flask import jsonify

def register_data_routes(app, services):
    """Registra rotas relacionadas aos dados"""
    
    data_service = services['data_service']
    chat_service = services['chat_service']
    
    @app.route('/api/collected_data_info')
    def collected_data_info():
        """Retorna informações sobre os dados coletados"""
        return jsonify(data_service.get_collected_data_info())
    
    @app.route('/api/previous_configs')
    def get_previous_configs():
        """Retorna lista de configurações de treinamentos anteriores"""
        return jsonify(data_service.get_previous_configs())
    
    @app.route('/api/chat_training_data_info')
    def get_chat_training_data_info():
        """Retorna informações sobre dados de treinamento do chat"""
        result = chat_service.get_chat_training_data_info()
        
        if 'error' in result:
            return jsonify(result), 503
        
        return jsonify(result)
