from flask import Flask, render_template, request, jsonify
from data_collector import DataCollector
from model_trainer import ModelTrainer
from config_manager import ConfigManager
import threading
import os
import logging
import warnings
import json
from datetime import datetime

# Configurar logging e suprimir avisos desnecessários
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*loss_type.*")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Inicializar componentes
config_manager = ConfigManager()
data_collector = DataCollector(config_manager)
model_trainer = ModelTrainer(config_manager)

def run_training_process(config, config_filename):
    """Função para executar o processo de treinamento em thread separada"""
    try:
        print(f"\n=== INICIANDO PROCESSO DE TREINAMENTO ===")
        print(f"Arquivo de configuração: {config_filename}")
        print(f"Total de textos: {config.get('total_texts', 'N/A')}")
        
        # Atualizar status inicial
        model_trainer.update_status('training', 0, 'Iniciando treinamento...')
        
        # Iniciar treinamento do modelo
        model_trainer.train_model(config)
        
        print(f"=== TREINAMENTO CONCLUÍDO ===")
        
    except Exception as e:
        print(f"\nERRO NO PROCESSO DE TREINAMENTO: {str(e)}")
        model_trainer.update_status('error', 0, f'Erro durante o treinamento: {str(e)}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Inicia o processo de treinamento (corrigida para evitar arquivos vazios)"""
    try:
        config = request.json
        
        # Validar configuração
        if not config.get('web_sources') and not config.get('email_config'):
            return jsonify({
                'status': 'error', 
                'message': 'Pelo menos uma fonte de dados deve ser configurada'
            })
        
        # Validar configurações de email se fornecidas
        if config.get('email_config'):
            email_config = config['email_config']
            required_fields = ['provider', 'username', 'password']
            
            for field in required_fields:
                if not email_config.get(field):
                    return jsonify({
                        'status': 'error',
                        'message': f'Campo obrigatório não informado: {field}'
                    })
            
            # Configurar servidores baseado no provedor
            provider = email_config['provider'].lower()
            if provider == 'office365':
                email_config.update({
                    'imap_server': 'outlook.office365.com',
                    'imap_port': 993,
                    'smtp_server': 'smtp.office365.com',
                    'smtp_port': 587,
                    'use_ssl': True,
                    'use_starttls': True
                })
            elif provider == 'gmail':
                email_config.update({
                    'imap_server': 'imap.gmail.com',
                    'imap_port': 993,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'use_ssl': True,
                    'use_starttls': True
                })
            elif provider == 'custom':
                # Verificar se configurações customizadas foram fornecidas
                custom_fields = ['imap_server', 'smtp_server']
                for field in custom_fields:
                    if not email_config.get(field):
                        return jsonify({
                            'status': 'error',
                            'message': f'Para provedor customizado, {field} é obrigatório'
                        })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Provedor de email não suportado. Use: office365, gmail ou custom'
                })
        
        # Iniciar coleta de dados em thread separada
        def collect_and_train():
            try:
                print("\n=== INICIANDO COLETA DE DADOS ===")
                
                # Coletar dados primeiro
                all_texts = []
                total_sources = set()
                
                # Coleta web
                if config.get('web_sources'):
                    for url in config['web_sources']:
                        try:
                            web_texts = data_collector.collect_web_data(url, config.get('keywords', []))
                            if web_texts and len(web_texts) > 0:
                                all_texts.extend(web_texts)
                                total_sources.add(url)
                                print(f"Coletados {len(web_texts)} textos de {url}")
                        except Exception as e:
                            print(f"Erro ao coletar de {url}: {e}")
                
                # Coleta email
                if config.get('email_config'):
                    try:
                        email_texts = data_collector.collect_email_data(config['email_config'])
                        if email_texts and len(email_texts) > 0:
                            all_texts.extend(email_texts)
                            total_sources.add(f"Email: {config['email_config']['username']}")
                            print(f"Coletados {len(email_texts)} textos de email")
                    except Exception as e:
                        print(f"Erro ao coletar emails: {e}")
                
                # Verificar se há dados válidos antes de continuar
                if not all_texts or len(all_texts) == 0:
                    print("Nenhum texto coletado - abortando treinamento")
                    model_trainer.update_status('error', 0, 'Nenhum dado foi coletado para treinamento. Verifique as fontes configuradas.')
                    return
                
                # Remover textos vazios ou muito curtos
                valid_texts = [text.strip() for text in all_texts if text.strip() and len(text.strip()) > 10]
                
                if len(valid_texts) == 0:
                    print("Nenhum texto válido encontrado - abortando")
                    model_trainer.update_status('error', 0, 'Nenhum texto válido foi encontrado. Os textos coletados estão vazios ou são muito curtos.')
                    return
                
                print(f"Textos válidos para treinamento: {len(valid_texts)}")
                
                # Salvar configuração apenas se houver dados válidos
                config['total_texts'] = len(valid_texts)
                config['total_sources'] = len(total_sources)
                config_manager.save_training_config(config)
                
                # Salvar dados coletados
                data_collector.save_collected_data(valid_texts, list(total_sources), config)
                
                print("\n=== INICIANDO TREINAMENTO DO MODELO ===")
                model_trainer.train_model(config)
                
            except Exception as e:
                print(f"\nERRO NO PROCESSO: {str(e)}")
                model_trainer.update_status('error', 0, f'Erro durante o processo: {str(e)}')
        
        thread = threading.Thread(target=collect_and_train)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Treinamento iniciado com sucesso!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/training_status')
def training_status():
    return jsonify(model_trainer.get_status())

@app.route('/api/collected_data_info')
def collected_data_info():
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
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/retrain_model', methods=['POST'])
def retrain_model():
    """Retreina o modelo usando configurações de treinamentos anteriores"""
    try:
        data = request.json
        previous_config_file = data.get('config_file')
        use_chat_data = data.get('use_chat_data', True)
        merge_strategy = data.get('merge_strategy', 'append')
        
        if not previous_config_file:
            return jsonify({'error': 'Arquivo de configuração não especificado'}), 400
        
        # Carregar configuração anterior
        config_path = f'training_data/{previous_config_file}'
        if not os.path.exists(config_path):
            return jsonify({'error': 'Arquivo de configuração não encontrada'}), 404
        
        with open(config_path, 'r', encoding='utf-8') as f:
            previous_config = json.load(f)
        
        print(f"Carregando configuração anterior: {previous_config_file}")
        
        # Preparar nova configuração baseada na anterior
        new_config = {
            'base_model': previous_config.get('base_model', 'microsoft/DialoGPT-small'),
            'epochs': previous_config.get('epochs', 3),
            'batch_size': previous_config.get('batch_size', 1),
            'max_length': previous_config.get('max_length', 256),
            'learning_rate': previous_config.get('learning_rate', 5e-5),
            'web_sources': previous_config.get('web_sources', []),
            'keywords': previous_config.get('keywords', []),
            'email_config': previous_config.get('email_config'),
            'retrain_mode': True,
            'previous_config': previous_config_file,
            'merge_strategy': merge_strategy
        }
        
        # Coletar dados existentes
        all_texts = []
        total_sources = set()
        
        # 1. Dados dos treinamentos anteriores
        if merge_strategy in ['append', 'merge']:
            # Buscar arquivos de dados coletados anteriormente
            training_data_dir = 'training_data'
            if os.path.exists(training_data_dir):
                for filename in os.listdir(training_data_dir):
                    if filename.startswith('collected_data_') and filename.endswith('.json'):
                        file_path = os.path.join(training_data_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_data = json.load(f)
                                if 'texts' in file_data:
                                    all_texts.extend(file_data['texts'])
                                    if 'sources' in file_data:
                                        total_sources.update(file_data['sources'])
                                    print(f"Carregados {len(file_data['texts'])} textos de {filename}")
                        except Exception as e:
                            print(f"Erro ao carregar {filename}: {e}")
        
        # 2. Dados do chat (se solicitado)
        if use_chat_data:
            chat_data_dir = 'chat_training_data'
            if os.path.exists(chat_data_dir):
                chat_texts = []
                for filename in os.listdir(chat_data_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(chat_data_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                chat_data = json.load(f)
                                if 'training_data' in chat_data:
                                    for item in chat_data['training_data']:
                                        if 'input' in item and 'output' in item:
                                            chat_text = f"Usuário: {item['input']}\nAssistente: {item['output']}"
                                            chat_texts.append(chat_text)
                                            total_sources.add(f"Chat: {chat_data['conversation_id']}")
                        except Exception as e:
                            print(f"Erro ao processar {filename}: {e}")
                
                all_texts.extend(chat_texts)
                print(f"Adicionados {len(chat_texts)} textos de conversas do chat")
        
        # 3. Coletar novos dados (se especificado)
        if new_config['web_sources'] or new_config['email_config']:
            print("Coletando novos dados...")
            
            # Coleta web
            if new_config['web_sources']:
                for url in new_config['web_sources']:
                    try:
                        web_texts = data_collector.collect_web_data(url, new_config['keywords'])
                        if web_texts:
                            all_texts.extend(web_texts)
                            total_sources.add(url)
                            print(f"Coletados {len(web_texts)} textos de {url}")
                    except Exception as e:
                        print(f"Erro ao coletar de {url}: {e}")
            
            # Coleta email
            if new_config['email_config']:
                try:
                    email_texts = data_collector.collect_email_data(new_config['email_config'])
                    if email_texts:
                        all_texts.extend(email_texts)
                        total_sources.add(f"Email: {new_config['email_config']['username']}")
                        print(f"Coletados {len(email_texts)} textos de email")
                except Exception as e:
                    print(f"Erro ao coletar emails: {e}")
        
        # Verificar se há dados suficientes
        if len(all_texts) == 0:
            print("Nenhum dado coletado - não gerando arquivo de configuração")
            return jsonify({
                'error': 'Nenhum dado foi coletado para o retreinamento',
                'details': 'Verifique as fontes de dados ou configurações anteriores'
            }), 400
        
        # Remover duplicatas e textos inválidos
        valid_texts = list(set([text.strip() for text in all_texts if text.strip() and len(text.strip()) > 10]))
        print(f"Total de textos únicos e válidos: {len(valid_texts)}")
        
        if len(valid_texts) < 10:
            return jsonify({
                'error': f'Dados insuficientes para retreinamento. Encontrados apenas {len(valid_texts)} textos únicos.',
                'suggestion': 'Adicione mais fontes de dados ou use configurações de treinamentos anteriores com mais dados.'
            }), 400
        
        # Salvar dados e configuração apenas se houver conteúdo válido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Atualizar configuração com dados válidos
        new_config['collected_data'] = []
        new_config['total_texts'] = len(valid_texts)
        new_config['total_sources'] = len(total_sources)
        new_config['timestamp'] = datetime.now().isoformat()
        
        # Salvar dados coletados
        data_filename = f'training_data/retrain_collected_data_{timestamp}.json'
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'total_texts': len(valid_texts),
            'sources': list(total_sources),
            'config_used': new_config,
            'texts': valid_texts,
            'retrain_info': {
                'previous_config': previous_config_file,
                'merge_strategy': merge_strategy,
                'used_chat_data': use_chat_data,
                'original_texts_count': len(all_texts),
                'unique_texts_count': len(valid_texts)
            }
        }
        
        os.makedirs('training_data', exist_ok=True)
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
        
        print(f"Dados salvos em: {data_filename}")
        new_config['collected_data'].append(data_filename)
        
        # Salvar configuração de treinamento
        config_filename = f'training_data/retrain_config_{timestamp}.json'
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
        
        print(f"Configuração salva em: {config_filename}")
        
        # Iniciar treinamento
        training_thread = threading.Thread(
            target=run_training_process, 
            args=(new_config, config_filename),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Retreinamento iniciado com base na configuração anterior',
            'config_file': config_filename,
            'data_file': data_filename,
            'stats': {
                'total_texts': len(valid_texts),
                'total_sources': len(total_sources),
                'previous_config': previous_config_file,
                'merge_strategy': merge_strategy,
                'used_chat_data': use_chat_data
            }
        })
        
    except Exception as e:
        print(f"Erro no retreinamento: {e}")
        return jsonify({'error': f'Erro no retreinamento: {str(e)}'}), 500

@app.route('/api/previous_configs')
def get_previous_configs():
    """Retorna lista de configurações de treinamentos anteriores"""
    try:
        configs = []
        training_data_dir = 'training_data'
        
        if os.path.exists(training_data_dir):
            for filename in os.listdir(training_data_dir):
                if (filename.startswith('training_config_') or filename.startswith('retrain_config_')) and filename.endswith('.json'):
                    file_path = os.path.join(training_data_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            
                        # Extrair informações relevantes
                        config_info = {
                            'filename': filename,
                            'timestamp': config.get('timestamp', 'N/A'),
                            'base_model': config.get('base_model', 'N/A'),
                            'epochs': config.get('epochs', 'N/A'),
                            'total_texts': config.get('total_texts', 0),
                            'total_sources': config.get('total_sources', 0),
                            'sources_preview': config.get('web_sources', [])[:3],
                            'has_email': bool(config.get('email_config')),
                            'file_size': os.path.getsize(file_path),
                            'is_retrain': filename.startswith('retrain_config_')
                        }
                        configs.append(config_info)
                    except Exception as e:
                        print(f"Erro ao ler {filename}: {e}")
        
        # Ordenar por timestamp (mais recente primeiro)
        configs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'configs': configs,
            'total_configs': len(configs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_with_chat_data', methods=['POST'])
def train_with_chat_data():
    """Treina modelo usando dados de conversas do chat"""
    try:
        data = request.json
        base_config_file = data.get('base_config_file')  # Configuração base opcional
        training_options = data.get('training_options', {})
        
        # Verificar se há dados de chat disponíveis
        chat_data_dir = 'chat_training_data'
        if not os.path.exists(chat_data_dir) or not os.listdir(chat_data_dir):
            return jsonify({
                'error': 'Nenhum dado de conversa encontrado',
                'details': 'Execute algumas conversas no chat primeiro'
            }), 400
        
        # Preparar configuração
        if base_config_file:
            # Usar configuração existente como base
            config_path = f'training_data/{base_config_file}'
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    base_config = json.load(f)
            else:
                return jsonify({'error': 'Configuração base não encontrada'}), 404
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
            'chat_data_source': chat_data_dir,
            'include_ai_responses': training_options.get('include_ai_responses', True),
            'include_human_responses': training_options.get('include_human_responses', True),
            'filter_by_satisfaction': training_options.get('filter_by_satisfaction', False),
            'min_satisfaction': training_options.get('min_satisfaction', 3),
            'timestamp': datetime.now().isoformat()
        })
        
        # Coletar e processar dados do chat
        all_texts = []
        total_sources = set()
        conversation_count = 0
        
        for filename in os.listdir(chat_data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(chat_data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                    
                    # Filtrar por satisfação se solicitado
                    if (new_config['filter_by_satisfaction'] and 
                        chat_data.get('satisfaction', 0) < new_config['min_satisfaction']):
                        continue
                    
                    conversation_count += 1
                    
                    if 'training_data' in chat_data:
                        for item in chat_data['training_data']:
                            if 'input' in item and 'output' in item:
                                # Filtrar por tipo de resposta
                                interaction_type = item.get('interaction_type', 'ai_response')
                                
                                include_item = False
                                if interaction_type == 'ai_response' and new_config['include_ai_responses']:
                                    include_item = True
                                elif interaction_type == 'human_response' and new_config['include_human_responses']:
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
            return jsonify({
                'error': 'Nenhum dado válido encontrado para treinamento',
                'details': 'Verifique os filtros aplicados ou se há conversas salvas'
            }), 400
        
        # Remover duplicatas
        unique_texts = list(set(all_texts))
        print(f"Dados do chat coletados: {len(unique_texts)} textos únicos de {conversation_count} conversas")
        
        if len(unique_texts) < 5:
            return jsonify({
                'error': f'Dados insuficientes: apenas {len(unique_texts)} textos únicos encontrados',
                'suggestion': 'Execute mais conversas no chat ou reduza os filtros'
            }), 400
        
        # Salvar dados coletados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_filename = f'training_data/chat_training_data_{timestamp}.json'
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'total_texts': len(unique_texts),
            'sources': list(total_sources),
            'config_used': new_config,
            'texts': unique_texts,
            'chat_training_info': {
                'conversations_processed': conversation_count,
                'original_texts_count': len(all_texts),
                'unique_texts_count': len(unique_texts),
                'training_options': training_options
            }
        }
        
        os.makedirs('training_data', exist_ok=True)
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
        
        # Salvar configuração
        config_filename = f'training_data/chat_training_config_{timestamp}.json'
        new_config['collected_data'] = [data_filename]
        new_config['total_texts'] = len(unique_texts)
        new_config['total_sources'] = len(total_sources)
        
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
        
        # Iniciar treinamento
        training_thread = threading.Thread(
            target=run_training_process,
            args=(new_config, config_filename),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Treinamento com dados do chat iniciado',
            'config_file': config_filename,
            'data_file': data_filename,
            'stats': {
                'total_texts': len(unique_texts),
                'conversations_used': conversation_count,
                'total_sources': len(total_sources),
                'training_options': training_options
            }
        })
        
    except Exception as e:
        print(f"Erro no treinamento com dados do chat: {e}")
        return jsonify({'error': f'Erro no treinamento: {str(e)}'}), 500

@app.route('/api/chat_training_data_info')
def get_chat_training_data_info():
    """Retorna informações sobre dados de treinamento do chat"""
    try:
        # Fazer requisição para o serviço de chat
        import requests
        response = requests.get('http://localhost:5001/chat/training_data')
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': 'Serviço de chat não disponível',
                'details': 'Execute o chat_widget.py primeiro'
            }), 503
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
