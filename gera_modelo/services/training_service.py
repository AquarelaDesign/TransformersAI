import threading
import json
import os
from datetime import datetime

class TrainingService:
    def __init__(self, config_manager, data_collector, model_trainer):
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.model_trainer = model_trainer
    
    def start_training(self, config):
        """Inicia o processo de treinamento (versão modular)"""
        try:
            # Validar configuração
            if not config.get('web_sources') and not config.get('email_config'):
                return {
                    'status': 'error', 
                    'message': 'Pelo menos uma fonte de dados deve ser configurada'
                }
            
            # Validar configurações de email se fornecidas
            if config.get('email_config'):
                validation_result = self._validate_email_config(config['email_config'])
                if validation_result['status'] == 'error':
                    return validation_result
            
            # Iniciar coleta e treinamento em thread separada
            thread = threading.Thread(target=self._collect_and_train, args=(config,))
            thread.daemon = True
            thread.start()
            
            return {'status': 'success', 'message': 'Treinamento iniciado com sucesso!'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _validate_email_config(self, email_config):
        """Valida configurações de email"""
        required_fields = ['provider', 'username', 'password']
        
        for field in required_fields:
            if not email_config.get(field):
                return {
                    'status': 'error',
                    'message': f'Campo obrigatório não informado: {field}'
                }
        
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
            custom_fields = ['imap_server', 'smtp_server']
            for field in custom_fields:
                if not email_config.get(field):
                    return {
                        'status': 'error',
                        'message': f'Para provedor customizado, {field} é obrigatório'
                    }
        else:
            return {
                'status': 'error',
                'message': 'Provedor de email não suportado. Use: office365, gmail ou custom'
            }
        
        return {'status': 'success'}
    
    def _collect_and_train(self, config):
        """Coleta dados e treina modelo (processo interno)"""
        try:
            print("\n=== INICIANDO COLETA DE DADOS ===")
            
            # Coletar dados primeiro
            all_texts = []
            total_sources = set()
            
            # Coleta web
            if config.get('web_sources'):
                for url in config['web_sources']:
                    try:
                        web_texts = self.data_collector.collect_web_data(url, config.get('keywords', []))
                        if web_texts and len(web_texts) > 0:
                            all_texts.extend(web_texts)
                            total_sources.add(url)
                            print(f"✅ Coletados {len(web_texts)} textos de {url}")
                        else:
                            print(f"⚠️ Nenhum texto válido coletado de {url}")
                    except Exception as e:
                        print(f"❌ Erro ao coletar de {url}: {e}")
            
            # Coleta email
            if config.get('email_config'):
                try:
                    email_texts = self.data_collector.collect_email_data(config['email_config'])
                    if email_texts and len(email_texts) > 0:
                        all_texts.extend(email_texts)
                        total_sources.add(f"Email: {config['email_config']['username']}")
                        print(f"✅ Coletados {len(email_texts)} textos de email")
                    else:
                        print("⚠️ Nenhum texto de email coletado")
                except Exception as e:
                    print(f"❌ Erro ao coletar emails: {e}")
            
            print(f"\n📊 RESUMO DA COLETA:")
            print(f"   - Total de textos brutos: {len(all_texts)}")
            print(f"   - Total de fontes: {len(total_sources)}")
            
            # Verificar se há dados válidos
            if not all_texts or len(all_texts) == 0:
                print("❌ Nenhum texto coletado - abortando treinamento")
                self.model_trainer.update_status('error', 0, 'Nenhum dado foi coletado para treinamento. Verifique as fontes configuradas.')
                return
            
            # Processar e validar textos
            valid_texts = self._process_texts(all_texts)
            
            if len(valid_texts) < 5:
                print("❌ Poucos textos únicos - abortando")
                self.model_trainer.update_status('error', 0, f'Dados insuficientes: apenas {len(valid_texts)} textos únicos válidos.')
                return
            
            print(f"\n✅ DADOS VÁLIDOS PARA TREINAMENTO: {len(valid_texts)} textos")
            
            # Salvar dados
            config['total_texts'] = len(valid_texts)
            config['total_sources'] = len(total_sources)
            config_filename = self.config_manager.save_training_config(config)
            
            data_filename = self.data_collector.save_collected_data(valid_texts, list(total_sources), config)
            
            if not data_filename:
                print("❌ Erro ao salvar dados coletados")
                self.model_trainer.update_status('error', 0, 'Erro ao salvar dados coletados.')
                return
            
            print(f"\n💾 Dados salvos em: {data_filename}")
            print(f"⚙️ Configuração salva em: {config_filename}")
            
            print("\n🚀 INICIANDO TREINAMENTO DO MODELO...")
            self.model_trainer.train_model(config)
            
        except Exception as e:
            print(f"\n❌ ERRO NO PROCESSO: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model_trainer.update_status('error', 0, f'Erro durante o processo: {str(e)}')
    
    def _process_texts(self, all_texts):
        """Processa e filtra textos válidos"""
        # Remover textos vazios ou muito curtos
        valid_texts = []
        for text in all_texts:
            if isinstance(text, str):
                clean_text = text.strip()
                if len(clean_text) >= 25:  # Mínimo 25 caracteres
                    valid_texts.append(clean_text)
        
        print(f"   - Textos válidos (≥25 chars): {len(valid_texts)}")
        
        # Remover duplicatas
        unique_texts = list(set(valid_texts))
        print(f"   - Textos únicos: {len(unique_texts)}")
        
        # Mostrar amostra dos dados
        if unique_texts:
            print("\n🔍 AMOSTRA DOS DADOS COLETADOS:")
            for i, text in enumerate(unique_texts[:3]):
                print(f"   {i+1}. {text[:100]}... ({len(text)} chars)")
        
        return unique_texts
    
    def retrain_model(self, previous_config_file, use_chat_data=True, merge_strategy='append'):
        """Retreina modelo usando configurações anteriores"""
        try:
            # Carregar configuração anterior
            config_path = f'training_data/{previous_config_file}'
            if not os.path.exists(config_path):
                return {'error': 'Arquivo de configuração não encontrada', 'status': 'error'}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                previous_config = json.load(f)
            
            print(f"Carregando configuração anterior: {previous_config_file}")
            
            # Preparar nova configuração
            new_config = self._prepare_retrain_config(previous_config, use_chat_data, merge_strategy)
            
            # Coletar dados para retreinamento
            collected_data = self._collect_retrain_data(new_config)
            
            if collected_data['status'] == 'error':
                return collected_data
            
            # Salvar e iniciar treinamento
            return self._save_and_start_retrain(new_config, collected_data['data'], previous_config_file, use_chat_data, merge_strategy)
            
        except Exception as e:
            print(f"Erro no retreinamento: {e}")
            return {'error': f'Erro no retreinamento: {str(e)}', 'status': 'error'}
    
    def _prepare_retrain_config(self, previous_config, use_chat_data, merge_strategy):
        """Prepara configuração para retreinamento"""
        return {
            'base_model': previous_config.get('base_model', 'microsoft/DialoGPT-small'),
            'epochs': previous_config.get('epochs', 3),
            'batch_size': previous_config.get('batch_size', 1),
            'max_length': previous_config.get('max_length', 256),
            'learning_rate': previous_config.get('learning_rate', 5e-5),
            'web_sources': previous_config.get('web_sources', []),
            'keywords': previous_config.get('keywords', []),
            'email_config': previous_config.get('email_config'),
            'retrain_mode': True,
            'previous_config': previous_config,
            'merge_strategy': merge_strategy,
            'use_chat_data': use_chat_data
        }
    
    def _collect_retrain_data(self, config):
        """Coleta dados para retreinamento"""
        all_texts = []
        total_sources = set()
        
        # 1. Dados dos treinamentos anteriores
        if config['merge_strategy'] in ['append', 'merge']:
            self._load_previous_training_data(all_texts, total_sources)
        
        # 2. Dados do chat
        if config['use_chat_data']:
            self._load_chat_data(all_texts, total_sources)
        
        # 3. Novos dados das fontes
        if config['web_sources'] or config['email_config']:
            self._collect_new_data(config, all_texts, total_sources)
        
        # Validar dados coletados
        if len(all_texts) == 0:
            return {
                'status': 'error',
                'error': 'Nenhum dado foi coletado para o retreinamento',
                'details': 'Verifique as fontes de dados ou configurações anteriores'
            }
        
        # Processar textos
        valid_texts = self._process_texts(all_texts)
        
        if len(valid_texts) < 10:
            return {
                'status': 'error',
                'error': f'Dados insuficientes para retreinamento. Encontrados apenas {len(valid_texts)} textos únicos.',
                'suggestion': 'Adicione mais fontes de dados ou use configurações de treinamentos anteriores com mais dados.'
            }
        
        return {
            'status': 'success',
            'data': {
                'valid_texts': valid_texts,
                'total_sources': total_sources,
                'stats': {
                    'original_count': len(all_texts),
                    'valid_count': len(valid_texts),
                    'sources_count': len(total_sources)
                }
            }
        }
    
    def _load_previous_training_data(self, all_texts, total_sources):
        """Carrega dados de treinamentos anteriores"""
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
    
    def _load_chat_data(self, all_texts, total_sources):
        """Carrega dados de conversas do chat"""
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
    
    def _collect_new_data(self, config, all_texts, total_sources):
        """Coleta novos dados das fontes configuradas"""
        print("Coletando novos dados...")
        
        # Coleta web
        if config['web_sources']:
            for url in config['web_sources']:
                try:
                    web_texts = self.data_collector.collect_web_data(url, config['keywords'])
                    if web_texts:
                        all_texts.extend(web_texts)
                        total_sources.add(url)
                        print(f"Coletados {len(web_texts)} textos de {url}")
                except Exception as e:
                    print(f"Erro ao coletar de {url}: {e}")
        
        # Coleta email
        if config['email_config']:
            try:
                email_texts = self.data_collector.collect_email_data(config['email_config'])
                if email_texts:
                    all_texts.extend(email_texts)
                    total_sources.add(f"Email: {config['email_config']['username']}")
                    print(f"Coletados {len(email_texts)} textos de email")
            except Exception as e:
                print(f"Erro ao coletar emails: {e}")
    
    def _save_and_start_retrain(self, new_config, data, previous_config_file, use_chat_data, merge_strategy):
        """Salva dados e inicia retreinamento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        valid_texts = data['valid_texts']
        total_sources = data['total_sources']
        
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
                'original_texts_count': data['stats']['original_count'],
                'unique_texts_count': len(valid_texts)
            }
        }
        
        os.makedirs('training_data', exist_ok=True)
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
        
        print(f"Dados salvos em: {data_filename}")
        new_config['collected_data'] = [data_filename]
        new_config['total_texts'] = len(valid_texts)
        new_config['total_sources'] = len(total_sources)
        
        # Salvar configuração
        config_filename = f'training_data/retrain_config_{timestamp}.json'
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
        
        print(f"Configuração salva em: {config_filename}")
        
        # Iniciar treinamento
        training_thread = threading.Thread(
            target=self._run_training_process,
            args=(new_config, config_filename),
            daemon=True
        )
        training_thread.start()
        
        return {
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
        }
    
    def _run_training_process(self, config, config_filename):
        """Executa processo de treinamento"""
        try:
            print(f"\n=== INICIANDO PROCESSO DE TREINAMENTO ===")
            print(f"Arquivo de configuração: {config_filename}")
            print(f"Total de textos: {config.get('total_texts', 'N/A')}")
            
            self.model_trainer.update_status('training', 0, 'Iniciando treinamento...')
            self.model_trainer.train_model(config)
            
            print(f"=== TREINAMENTO CONCLUÍDO ===")
            
        except Exception as e:
            print(f"\nERRO NO PROCESSO DE TREINAMENTO: {str(e)}")
            self.model_trainer.update_status('error', 0, f'Erro durante o treinamento: {str(e)}')
