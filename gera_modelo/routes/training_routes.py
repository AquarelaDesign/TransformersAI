from flask import request, jsonify

def register_training_routes(app, services):
    """Registra rotas relacionadas ao treinamento de modelos"""
    
    training_service = services['training_service']
    model_trainer = services['model_trainer']
    
    @app.route('/api/start_training', methods=['POST'])
    def start_training():
        """Inicia o processo de treinamento"""
        config = request.json
        return jsonify(training_service.start_training(config))
    
    @app.route('/api/training_status')
    def training_status():
        return jsonify(model_trainer.get_status())
    
    @app.route('/api/retrain_model', methods=['POST'])
    def retrain_model():
        """Retreina o modelo usando configurações de treinamentos anteriores"""
        data = request.json
        previous_config_file = data.get('config_file')
        use_chat_data = data.get('use_chat_data', True)
        merge_strategy = data.get('merge_strategy', 'append')
        
        if not previous_config_file:
            return jsonify({'error': 'Arquivo de configuração não especificado'}), 400
        
        result = training_service.retrain_model(previous_config_file, use_chat_data, merge_strategy)
        
        if result.get('status') == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
    
    @app.route('/api/train_with_chat_data', methods=['POST'])
    def train_with_chat_data():
        """Treina modelo usando dados de conversas do chat"""
        data = request.json
        base_config_file = data.get('base_config_file')
        training_options = data.get('training_options', {})
        
        chat_service = services['chat_service']
        result = chat_service.train_with_chat_data(base_config_file, training_options)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
