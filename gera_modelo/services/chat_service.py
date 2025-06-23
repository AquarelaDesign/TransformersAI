import requests
import threading
import json
import os
from datetime import datetime

class ChatService:
    def __init__(self, config_manager, data_collector, model_trainer):
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.model_trainer = model_trainer
        self.chat_service_url = 'http://localhost:5001'
    
    def get_chat_training_data_info(self):
        """Retorna informações sobre dados de treinamento do chat"""
        try:
            response = requests.get(f'{self.chat_service_url}/chat/training_data')
            
            if response.ok:
                return response.json()
            else:
                return {
                    'error': 'Serviço de chat não disponível',
                    'details': 'Execute o chat_widget.py primeiro'
                }
        except Exception as e:
            return {'error': str(e)}
    
    def train_with_chat_data(self, base_config_file=None, training_options=None):
        """Treina modelo usando dados de conversas do chat"""
        try:
            if training_options is None:
                training_options = {}
            
            # Verificar se há dados de chat disponíveis
            chat_data_dir = 'chat_training_data'
            if not os.path.exists(chat_data_dir) or not os.listdir(chat_data_dir):
                return {
                    'error': 'Nenhum dado de conversa encontrado',
                    'details': 'Execute algumas conversas no chat primeiro'
                }
            
            # Preparar configuração
            config = self._prepare_chat_training_config(base_config_file, training_options)
            
            # Coletar e processar dados do chat
            chat_data_result = self._collect_chat_training_data(config, training_options)
            
            if chat_data_result['status'] == 'error':
                return chat_data_result
            
            # Salvar e iniciar treinamento
            return self._save_and_start_chat_training(config, chat_data_result['data'], training_options)
            
        except Exception as e:
            print(f"Erro no treinamento com dados do chat: {e}")
            return {'error': f'Erro no treinamento: {str(e)}'}
    
    def _prepare_chat_training_config(self, base_config_file, training_options):
        """Prepara configuração para treinamento com dados do chat"""
        if base_config_file:
            # Usar configuração existente como base
            config_path = f'training_data/{base_config_file}'
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    base_config = json.load(f)
            else:
                raise FileNotFoundError('Configuração base não encontrada')
        else:
            # Configuração padrão
            base_config = {
                'base_model': 'microsoft/DialoGPT-small',
                'epochs': 3,
                'batch_size': 1,
                'max_length': 256,
                'learning_rate': 5e-5
            }
        
        # Atualizar com opções específicas
        new_config = {**base_config}
        new_config.update({
            'training_type': 'chat_data_only',
            'chat_data_source': 'chat_training_data',
            'include_ai_responses': training_options.get('include_ai_responses', True),
            'include_human_responses': training_options.get('include_human_responses', True),
            'filter_by_satisfaction': training_options.get('filter_by_satisfaction', False),
            'min_satisfaction': training_options.get('min_satisfaction', 3),
            'timestamp': datetime.now().isoformat()
        })
        
        return new_config
    
    def _collect_chat_training_data(self, config, training_options):
        """Coleta e processa dados do chat para treinamento"""
        all_texts = []
        total_sources = set()
        conversation_count = 0
        chat_data_dir = config['chat_data_source']
        
        for filename in os.listdir(chat_data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(chat_data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                    
                    # Filtrar por satisfação se solicitado
                    if (config['filter_by_satisfaction'] and 
                        chat_data.get('satisfaction', 0) < config['min_satisfaction']):
                        continue
                    
                    conversation_count += 1
                    
                    if 'training_data' in chat_data:
                        for item in chat_data['training_data']:
                            if 'input' in item and 'output' in item:
                                # Filtrar por tipo de resposta
                                interaction_type = item.get('interaction_type', 'ai_response')
                                
                                include_item = False
                                if interaction_type == 'ai_response' and config['include_ai_responses']:
                                    include_item = True
                                elif interaction_type == 'human_response' and config['include_human_responses']:
                                    include_item = True
                                
                                if include_item:
                                    # Formato para treinamento
                                    training_text = f"Usuário: {item['input']}\nAssistente: {item['output']}"
                                    all_texts.append(training_text)
                                    total_sources.add(f"Chat: {chat_data['conversation_id']}")
                
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
        
        # Verificar se há dados suficientes
        if len(all_texts) == 0:
            return {
                'status': 'error',
                'error': 'Nenhum dado válido encontrado para treinamento',
                'details': 'Verifique os filtros aplicados ou se há conversas salvas'
            }
        
        # Remover duplicatas
        unique_texts = list(set(all_texts))
        print(f"Dados do chat coletados: {len(unique_texts)} textos únicos de {conversation_count} conversas")
        
        if len(unique_texts) < 5:
            return {
                'status': 'error',
                'error': f'Dados insuficientes: apenas {len(unique_texts)} textos únicos encontrados',
                'suggestion': 'Execute mais conversas no chat ou reduza os filtros'
            }
        
        return {
            'status': 'success',
            'data': {
                'unique_texts': unique_texts,
                'total_sources': total_sources,
                'conversation_count': conversation_count,
                'original_count': len(all_texts)
            }
        }
    
    def _save_and_start_chat_training(self, config, data, training_options):
        """Salva dados e inicia treinamento com dados do chat"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_texts = data['unique_texts']
        total_sources = data['total_sources']
        
        # Salvar dados coletados
        data_filename = f'training_data/chat_training_data_{timestamp}.json'
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'total_texts': len(unique_texts),
            'sources': list(total_sources),
            'config_used': config,
            'texts': unique_texts,
            'chat_training_info': {
                'conversations_processed': data['conversation_count'],
                'original_texts_count': data['original_count'],
                'unique_texts_count': len(unique_texts),
                'training_options': training_options
            }
        }
        
        os.makedirs('training_data', exist_ok=True)
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
        
        # Salvar configuração
        config_filename = f'training_data/chat_training_config_{timestamp}.json'
        config['collected_data'] = [data_filename]
        config['total_texts'] = len(unique_texts)
        config['total_sources'] = len(total_sources)
        
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Iniciar treinamento
        training_thread = threading.Thread(
            target=self._run_training_process,
            args=(config, config_filename),
            daemon=True
        )
        training_thread.start()
        
        return {
            'status': 'success',
            'message': 'Treinamento com dados do chat iniciado',
            'config_file': config_filename,
            'data_file': data_filename,
            'stats': {
                'total_texts': len(unique_texts),
                'conversations_used': data['conversation_count'],
                'total_sources': len(total_sources),
                'training_options': training_options
            }
        }
    
    def _run_training_process(self, config, config_filename):
        """Executa processo de treinamento"""
        try:
            print(f"\n=== INICIANDO TREINAMENTO COM DADOS DO CHAT ===")
            print(f"Arquivo de configuração: {config_filename}")
            print(f"Total de textos: {config.get('total_texts', 'N/A')}")
            
            self.model_trainer.update_status('training', 0, 'Iniciando treinamento com dados do chat...')
            self.model_trainer.train_model(config)
            
            print(f"=== TREINAMENTO COM DADOS DO CHAT CONCLUÍDO ===")
            
        except Exception as e:
            print(f"\nERRO NO TREINAMENTO COM DADOS DO CHAT: {str(e)}")
            self.model_trainer.update_status('error', 0, f'Erro durante o treinamento: {str(e)}')
