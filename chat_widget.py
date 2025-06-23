from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from datetime import datetime
import uuid
import threading
import time
from data_collector import DataCollector

class ChatBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.conversations = {}
        self.human_queue = []
        self.human_agents = {}  # Armazena informações dos agentes {agent_id: {name, status, etc}}
        
    def load_latest_model(self, background=False):
        """Carrega o modelo mais recente treinado"""
        try:
            models_dir = './models'
            if not os.path.exists(models_dir):
                print("Diretório de modelos não encontrado. Sistema funcionará apenas com respostas padrão.")
                return False
                
            # Encontrar o modelo mais recente
            model_folders = [f for f in os.listdir(models_dir) if f.startswith('trained_')]
            if not model_folders:
                print("Nenhum modelo treinado encontrado. Sistema funcionará apenas com respostas padrão.")
                return False
                
            latest_model = max(model_folders, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            model_path = os.path.join(models_dir, latest_model)
            
            print(f"Carregando modelo: {model_path}")
            
            # Carregar com tratamento de erro melhorado
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Força float32 para compatibilidade
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.model_loaded = True
                print("Modelo carregado com sucesso!")
                return True
                
            except Exception as model_error:
                print(f"Erro ao carregar arquivos do modelo: {model_error}")
                print("Sistema continuará com respostas padrão.")
                return False
            
        except Exception as e:
            print(f"Erro no carregamento do modelo: {e}")
            print("Sistema continuará com respostas padrão.")
            return False
    
    def try_load_model_async(self):
        """Tenta carregar modelo em background de forma segura"""
        def load_model_safe():
            try:
                self.load_latest_model(background=True)
            except Exception as e:
                print(f"Erro no carregamento assíncrono: {e}")
        
        # Usar thread normal (não daemon) com timeout
        model_thread = threading.Thread(target=load_model_safe)
        model_thread.daemon = False  # Não usar daemon
        model_thread.start()
        
        # Aguardar por um tempo razoável
        model_thread.join(timeout=30)  # 30 segundos timeout
        
        if model_thread.is_alive():
            print("Carregamento do modelo demorou mais que o esperado. Continuando com respostas padrão.")

    def generate_response(self, message, conversation_id=None):
        """Gera resposta usando o modelo treinado (melhorada)"""
        if not self.model_loaded:
            print("Modelo não carregado, usando respostas padrão")
            return self._get_fallback_response(message)
        
        try:
            # Primeiro tentar resposta inteligente baseada em padrões
            intelligent_response = self._get_intelligent_response(message)
            if intelligent_response:
                return intelligent_response
            
            # Se não encontrou padrão, tentar modelo treinado
            return self._generate_ai_response(message, conversation_id)
            
        except Exception as e:
            print(f"Erro na geração de resposta: {e}")
            return self._get_fallback_response(message)
    
    def _get_intelligent_response(self, message):
        """Respostas inteligentes baseadas em padrões (melhorada)"""
        message_lower = message.lower().strip()
        
        # Remover pontuação para melhor detecção
        import re
        clean_message = re.sub(r'[^\w\s]', '', message_lower)
        
        # Saudações
        if any(word in clean_message for word in ['ola', 'oi', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'hello', 'hey']):
            return 'Olá! Seja bem-vindo. Como posso ajudá-lo hoje?'
        
        # Agradecimentos
        if any(word in clean_message for word in ['obrigado', 'obrigada', 'valeu', 'thanks', 'brigado']):
            return 'De nada! Fico feliz em ajudar. Há mais alguma coisa que posso fazer por você?'
        
        # Perguntas sobre produtos/serviços específicos
        if 'domit' in clean_message or 'dormit' in clean_message:
            return 'O Domit é nosso produto para melhorar a qualidade do sono. Gostaria de saber mais detalhes sobre horários da integração ou tem alguma dúvida específica?'
        
        # Competência e integrações
        if any(word in clean_message for word in ['competencia', 'integracao', 'shorenia', 'europa']):
            return 'Sobre nossa competência com integrações na Shorenia e Europa, posso fornecer informações detalhadas. Que tipo de informação específica você precisa?'
        
        # Perguntas sobre preços
        if any(word in clean_message for word in ['preço', 'valor', 'custo', 'quanto custa', 'caro', 'barato', 'preco']):
            return 'Para informações detalhadas sobre preços, posso conectá-lo com nossa equipe comercial. Gostaria que eu transferisse para um atendente?'
        
        # Problemas técnicos
        if any(word in clean_message for word in ['problema', 'erro', 'bug', 'não funciona', 'defeito', 'nao funciona']):
            return 'Entendo que você está enfrentando um problema. Pode me fornecer mais detalhes sobre o que está acontecendo? Assim posso ajudá-lo melhor.'
        
        # Produtos e serviços gerais
        if any(word in clean_message for word in ['produto', 'servico', 'oferece', 'vende', 'disponivel', 'serviço']):
            return 'Temos diversos produtos e serviços disponíveis. Pode me dizer qual área específica te interessa? Assim posso ser mais preciso na resposta.'
        
        # Informações de contato
        if any(word in clean_message for word in ['contato', 'telefone', 'email', 'endereco', 'localizacao', 'endereço']):
            return 'Posso ajudá-lo com informações de contato. Prefere que eu transfira para um atendente ou você gostaria de informações específicas?'
        
        # Ajuda geral
        if any(word in clean_message for word in ['ajuda', 'help', 'socorro', 'duvida', 'dúvida']):
            return 'Claro! Estou aqui para ajudar. Pode me contar qual é sua dúvida ou necessidade?'
        
        # Despedidas
        if any(word in clean_message for word in ['tchau', 'bye', 'ate logo', 'até logo', 'obrigado', 'era so isso']):
            return 'Foi um prazer ajudá-lo! Se precisar de mais alguma coisa, estarei aqui. Tenha um ótimo dia!'
        
        # FAQ comum
        if any(word in clean_message for word in ['horario', 'horário', 'funcionamento', 'atendimento']):
            return 'Nosso horário de atendimento é de segunda a sexta, das 8h às 18h. Posso ajudá-lo com algo específico?'
        
        return None  # Não encontrou padrão conhecido
    
    def _get_fallback_response(self, message):
        """Resposta padrão quando modelo não está disponível (melhorada)"""
        message_lower = message.lower().strip()
        
        # Respostas mais específicas baseadas no contexto
        if any(word in message_lower for word in ['domit', 'dormit']):
            return "Sobre o Domit, posso conectá-lo com nossa equipe especializada para mais detalhes. Gostaria que eu transferisse?"
        
        elif any(word in message_lower for word in ['preço', 'valor', 'custo', 'preco']):
            return "Para informações sobre preços, nossa equipe comercial pode ajudá-lo melhor. Posso transferir para um atendente?"
        
        elif any(word in message_lower for word in ['problema', 'erro', 'não funciona', 'nao funciona']):
            return "Entendo que você tem um problema. Nossa equipe técnica pode ajudá-lo melhor. Gostaria que eu transferisse para um especialista?"
        
        elif any(word in message_lower for word in ['produto', 'serviço', 'servico']):
            return "Temos vários produtos e serviços. Para informações detalhadas, posso conectá-lo com nossa equipe. Qual sua preferência?"
        
        else:
            return "Entendo sua pergunta. Para dar a melhor resposta possível, posso transferi-lo para um de nossos atendentes especializados. Gostaria?"
    
    def _generate_ai_response(self, message, conversation_id):
        """Gera resposta usando modelo treinado (mais conservadora)"""
        try:
            print(f"Tentando gerar resposta IA para: {message}")
            
            # Sempre usar fallback primeiro se o padrão for reconhecido
            intelligent_response = self._get_intelligent_response(message)
            if intelligent_response:
                print(f"Usando resposta inteligente: {intelligent_response}")
                return intelligent_response
            
            # Se chegou aqui, usar resposta padrão
            fallback = self._get_fallback_response(message)
            print(f"Usando resposta fallback: {fallback}")
            return fallback
            
        except Exception as e:
            print(f"Erro no modelo AI: {e}")
            return self._get_fallback_response(message)

    def save_conversation_data(self, conversation_id):
        """Salva conversa para futuros treinamentos (com informações completas do agente)"""
        if conversation_id not in self.conversations:
            return
        
        conversation = self.conversations[conversation_id]
        
        # Preparar dados para treinamento - incluindo diálogos com atendentes
        training_data = []
        chat_history = []
        
        # Obter informações completas do agente
        agent_id = conversation.get('assigned_agent')
        agent_name = conversation.get('agent_name', agent_id)
        
        if agent_id and agent_id in self.human_agents:
            agent_name = self.human_agents[agent_id]['name']
        
        # Processar todas as mensagens da conversa
        for i, msg in enumerate(conversation['messages']):
            if msg['type'] == 'user':
                # Buscar resposta correspondente (bot ou agent)
                response_msg = None
                if i + 1 < len(conversation['messages']):
                    next_msg = conversation['messages'][i + 1]
                    if next_msg['type'] in ['bot', 'agent']:
                        response_msg = next_msg
                
                if response_msg:
                    # Determinar o tipo de interação
                    interaction_type = 'ai_response' if response_msg['type'] == 'bot' else 'human_response'
                    
                    training_item = {
                        'input': msg['content'],
                        'output': response_msg['content'],
                        'interaction_type': interaction_type,
                        'rating': response_msg.get('rating', 'neutral'),
                        'timestamp': msg['timestamp'],
                        'response_timestamp': response_msg['timestamp'],
                        'agent_id': response_msg.get('agent_id') if response_msg['type'] == 'agent' else None,
                        'agent_name': agent_name if response_msg['type'] == 'agent' else None
                    }
                    training_data.append(training_item)
            
            # Manter histórico completo
            chat_history.append({
                'type': msg['type'],
                'content': msg['content'],
                'timestamp': msg['timestamp'],
                'agent_id': msg.get('agent_id'),
                'agent_name': agent_name if msg.get('agent_id') == agent_id else None
            })
        
        # Calcular métricas da conversa e tempo
        ai_interactions = len([item for item in training_data if item['interaction_type'] == 'ai_response'])
        human_interactions = len([item for item in training_data if item['interaction_type'] == 'human_response'])
        
        # Métricas de tempo detalhadas
        timing_metrics = conversation.get('timing_metrics', {})
        conversation_duration = self._calculate_duration(conversation)
        human_time_seconds = timing_metrics.get('total_human_time_seconds', 0)
        
        # Salvar arquivo de treinamento
        os.makedirs('chat_training_data', exist_ok=True)
        filename = f"chat_training_data/conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        conversation_summary = {
            'conversation_id': conversation_id,
            'start_time': conversation['start_time'],
            'end_time': datetime.now().isoformat(),
            'client_data': conversation.get('client_data', {}),
            'total_messages': len(conversation['messages']),
            'satisfaction': conversation.get('satisfaction', 'unknown'),
            'transferred_to_human': conversation.get('transferred_to_human', False),
            'assigned_agent': agent_id,
            'agent_name': agent_name,  # Nome completo do agente
            'agent_start_time': conversation.get('agent_start_time'),
            'ended_by': conversation.get('ended_by', 'unknown'),
            'client_history_available': len(conversation.get('client_history', [])) > 0,
            'timing_metrics': {
                'conversation_duration_minutes': conversation_duration,
                'human_transfer_time': timing_metrics.get('human_transfer_time'),
                'human_start_time': timing_metrics.get('human_start_time'),
                'human_end_time': timing_metrics.get('human_end_time'),
                'total_human_time_seconds': human_time_seconds,
                'human_time_minutes': round(human_time_seconds / 60, 1),
                'waiting_time_before_human': self._calculate_waiting_time(timing_metrics),
                'ai_only_time_minutes': conversation_duration - (human_time_seconds / 60) if conversation_duration > 0 else 0
            },
            'metrics': {
                'ai_interactions': ai_interactions,
                'human_interactions': human_interactions,
                'total_interactions': len(training_data),
                'conversation_duration_minutes': conversation_duration
            },
            'training_data': training_data,
            'full_chat_history': chat_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_summary, f, ensure_ascii=False, indent=2)
        
        print(f"Conversa salva para treinamento: {filename}")
        print(f"  - IA: {ai_interactions} interações")
        print(f"  - Humano: {human_interactions} interações")
        print(f"  - Agente: {agent_name} ({agent_id})")
        print(f"  - Tempo humano: {round(human_time_seconds / 60, 1)} minutos")
        print(f"  - Cliente: {conversation.get('client_data', {}).get('email', 'Sem email')}")
        
        return filename

    def _calculate_duration(self, conversation):
        """Calcula duração da conversa em minutos"""
        try:
            start = datetime.fromisoformat(conversation['start_time'])
            end = datetime.now()
            if 'end_time' in conversation:
                end = datetime.fromisoformat(conversation['end_time'])
            
            duration = (end - start).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0
    
    def _calculate_waiting_time(self, timing_metrics):
        """Calcula tempo de espera antes do atendimento humano"""
        try:
            transfer_time = timing_metrics.get('human_transfer_time')
            start_time = timing_metrics.get('human_start_time')
            
            if transfer_time and start_time:
                transfer_dt = datetime.fromisoformat(transfer_time)
                start_dt = datetime.fromisoformat(start_time)
                waiting_seconds = (start_dt - transfer_dt).total_seconds()
                return round(waiting_seconds / 60, 1)  # Em minutos
        except:
            pass
        return 0

# Instância global do chatbot
chatbot = ChatBot()

# Tentar carregar modelo na inicialização (sem daemon thread)
print("Iniciando sistema de chat...")
try:
    # Tentar carregamento direto primeiro (mais rápido)
    if not chatbot.load_latest_model():
        print("Carregamento direto falhou. Tentando em background...")
        chatbot.try_load_model_async()
except Exception as e:
    print(f"Erro na inicialização do modelo: {e}")
    print("Sistema funcionará apenas com respostas padrão.")

# Aplicação Flask para o chat
chat_app = Flask(__name__, 
                 template_folder='chat_templates',
                 static_folder='chat_static')

# Configurar CORS para permitir requisições do frontend
CORS(chat_app, origins=[
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5001",
    "http://127.0.0.1:5001"
])

# Instanciar coletor de dados
data_collector = DataCollector()

@chat_app.route('/chat')
def chat_widget():
    """Página do widget de chat"""
    return render_template('chat_widget.html')

@chat_app.route('/chat/start', methods=['POST'])
def start_conversation():
    """Inicia nova conversa com dados do cliente (melhorada com histórico)"""
    data = request.json or {}
    client_data = data.get('client_data', {})
    
    conversation_id = str(uuid.uuid4())
    
    # Validar e normalizar email
    client_email = client_data.get('email', '').strip().lower()
    if client_email and '@' not in client_email:
        client_email = ''  # Email inválido
    
    # Buscar conversas anteriores do cliente para referência
    previous_conversations = []
    is_returning_client = False
    
    if client_email:
        previous_conversations = get_client_history_by_email(client_email)
        is_returning_client = len(previous_conversations) > 0
    
    chatbot.conversations[conversation_id] = {
        'id': conversation_id,
        'start_time': datetime.now().isoformat(),
        'messages': [],
        'status': 'active',
        'transferred_to_human': False,
        'client_data': {
            'name': client_data.get('name', ''),
            'email': client_email,
            'phone': client_data.get('phone', ''),
            'collected_at': datetime.now().isoformat()
        },
        'timing_metrics': {
            'conversation_start': datetime.now().isoformat(),
            'human_transfer_time': None,
            'human_start_time': None,
            'human_end_time': None,
            'total_human_time_seconds': 0,
            'response_times': []
        },
        'client_history': previous_conversations if is_returning_client else []
    }
    
    # Mensagem de boas-vindas personalizada
    welcome_message = 'Seja bem-vindo ao nosso atendimento! Como posso ajudá-lo hoje?'
    
    if is_returning_client:
        client_name = client_data.get('name', 'Cliente')
        history_count = len(previous_conversations)
        last_contact = format_date(previous_conversations[0].get('date'))
        
        welcome_message = f'Olá {client_name}! É um prazer tê-lo de volta. '
        welcome_message += f'Vejo que você já conversou conosco {history_count} vez(es), '
        welcome_message += f'sendo o último contato {last_contact}. '
        welcome_message += 'Como posso ajudá-lo hoje?'
    
    return jsonify({
        'conversation_id': conversation_id,
        'message': welcome_message,
        'status': 'connected',
        'returning_client': is_returning_client,
        'client_history_count': len(previous_conversations) if is_returning_client else 0
    })

def get_client_history_by_email(email):
    """Busca histórico completo de conversas por email do cliente"""
    if not email or '@' not in email:
        return []
    
    email = email.strip().lower()
    history = []
    
    try:
        # 1. Buscar em conversas ativas/recentes na memória
        for conv_id, conv in chatbot.conversations.items():
            conv_email = conv.get('client_data', {}).get('email', '').strip().lower()
            if conv_email == email and conv_id != conv.get('id'):  # Evitar conversa atual
                history.append({
                    'conversation_id': conv_id,
                    'date': conv.get('start_time'),
                    'end_date': conv.get('end_time'),
                    'summary': get_conversation_summary(conv),
                    'satisfaction': conv.get('satisfaction', 'unknown'),
                    'transferred_to_human': conv.get('transferred_to_human', False),
                    'total_messages': len(conv.get('messages', [])),
                    'human_time_minutes': round(conv.get('timing_metrics', {}).get('total_human_time_seconds', 0) / 60, 1),
                    'status': conv.get('status', 'unknown'),
                    'source': 'memory'
                })
        
        # 2. Buscar em arquivos de chat salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        # Verificar email do cliente
                        saved_email = saved_conv.get('client_data', {}).get('email', '').strip().lower()
                        
                        if saved_email == email:
                            # Calcular tempo humano do arquivo
                            human_time_seconds = saved_conv.get('timing_metrics', {}).get('total_human_time_seconds', 0)
                            
                            history.append({
                                'conversation_id': saved_conv.get('conversation_id', filename.replace('.json', '')),
                                'date': saved_conv.get('start_time'),
                                'end_date': saved_conv.get('end_time'),
                                'summary': f"Conversa salva - {saved_conv.get('total_messages', 0)} mensagens",
                                'satisfaction': saved_conv.get('satisfaction', 'unknown'),
                                'transferred_to_human': saved_conv.get('transferred_to_human', False),
                                'total_messages': saved_conv.get('total_messages', 0),
                                'human_time_minutes': round(human_time_seconds / 60, 1),
                                'assigned_agent': saved_conv.get('assigned_agent'),
                                'source': 'file'
                            })
                            
                    except Exception as e:
                        print(f"Erro ao processar {filename} para histórico: {e}")
        
        # 3. Ordenar por data (mais recente primeiro)
        history.sort(key=lambda x: x['date'] or '', reverse=True)
        
        print(f"Histórico encontrado para {email}: {len(history)} conversas")
        return history
        
    except Exception as e:
        print(f"Erro ao buscar histórico para {email}: {e}")
        return []

def format_date(date_str):
    """Formata data para exibição amigável"""
    try:
        if not date_str:
            return "Data não disponível"
        
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - date_obj
        
        if diff.days == 0:
            return "Hoje"
        elif diff.days == 1:
            return "Ontem"
        elif diff.days < 7:
            return f"{diff.days} dias atrás"
        else:
            return date_obj.strftime("%d/%m/%Y")
    except:
        return date_str

def get_conversation_summary(conversation):
    """Gera resumo de uma conversa"""
    messages = conversation.get('messages', [])
    if not messages:
        return "Conversa sem mensagens"
    
    user_messages = [msg for msg in messages if msg['type'] == 'user']
    if user_messages:
        first_message = user_messages[0]['content']
        if len(first_message) > 100:
            return first_message[:100] + "..."
        return first_message
    
    return f"Conversa com {len(messages)} mensagens"

@chat_app.route('/chat/message', methods=['POST'])
def send_message():
    """Processa mensagem do usuário (corrigida)"""
    data = request.json
    conversation_id = data.get('conversation_id')
    user_message = data.get('message', '').strip()
    
    if not conversation_id or not user_message:
        return jsonify({'error': 'Dados inválidos'}), 400
    
    if conversation_id not in chatbot.conversations:
        return jsonify({'error': 'Conversa não encontrada'}), 404
    
    conversation = chatbot.conversations[conversation_id]
    
    # Adicionar mensagem do usuário
    conversation['messages'].append({
        'type': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"[CHAT] Mensagem recebida: {user_message}")
    print(f"[CHAT] Conversa ID: {conversation_id}")
    print(f"[CHAT] Status atual: {conversation.get('status', 'indefinido')}")
    
    # Verificar se tem agente humano atendendo
    if conversation.get('assigned_agent'):
        print(f"[CHAT] Conversa sendo atendida por agente: {conversation['assigned_agent']}")
        return jsonify({
            'response': 'Sua mensagem foi enviada para o atendente. Aguarde a resposta.',
            'type': 'system',
            'human_agent': True
        })
    
    # Verificar se já foi transferido para humano mas ainda não tem agente
    if conversation.get('transferred_to_human', False):
        print("[CHAT] Conversa na fila de atendimento humano")
        queue_position = len([item for item in chatbot.human_queue if not item.get('assigned_agent')])
        return jsonify({
            'response': f'Você está na fila para atendimento humano (posição {queue_position}). Um agente entrará em contato em breve.',
            'type': 'system',
            'queue_position': queue_position
        })
    
    # Verificar se precisa transferir para humano (palavras-chave)
    transfer_keywords = ['atendente', 'humano', 'pessoa', 'falar com alguém', 'transferir', 'não resolve', 'operador', 'falar com pessoa']
    should_transfer = any(keyword in user_message.lower() for keyword in transfer_keywords)
    
    if should_transfer:
        print("[CHAT] Transferindo para atendimento humano por palavra-chave")
        return handle_human_transfer(conversation_id)
    
    # Gerar resposta da IA (sempre usar resposta inteligente)
    print("[CHAT] Gerando resposta da IA...")
    bot_response = chatbot.generate_response(user_message, conversation_id)
    print(f"[CHAT] IA respondeu: {bot_response}")
    
    # Adicionar resposta da IA
    conversation['messages'].append({
        'type': 'bot',
        'content': bot_response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Preparar resposta
    response_data = {
        'response': bot_response,
        'type': 'bot',
        'suggestions': get_suggestions(user_message)
    }
    
    # Verificar se deve mostrar opção de transferência (após algumas interações)
    user_messages_count = len([msg for msg in conversation['messages'] if msg['type'] == 'user'])
    if user_messages_count >= 3:  # Após 3 mensagens do usuário
        response_data['show_transfer_option'] = True
    
    return jsonify(response_data)

@chat_app.route('/chat/transfer', methods=['POST'])
def request_human_transfer():
    """Solicita transferência para atendente humano"""
    data = request.json
    conversation_id = data.get('conversation_id')
    
    if conversation_id not in chatbot.conversations:
        return jsonify({'error': 'Conversa não encontrada'}), 404
    
    return handle_human_transfer(conversation_id)

def handle_human_transfer(conversation_id):
    """Gerencia transferência para atendimento humano (com controle de tempo)"""
    if conversation_id not in chatbot.conversations:
        return jsonify({'error': 'Conversa não encontrada'}), 404
        
    conversation = chatbot.conversations[conversation_id]
    
    # Marcar como transferida e registrar tempo
    conversation['transferred_to_human'] = True
    conversation['timing_metrics']['human_transfer_time'] = datetime.now().isoformat()
    
    # Verificar se já não está na fila
    already_in_queue = any(item['conversation_id'] == conversation_id for item in chatbot.human_queue)
    
    if not already_in_queue:
        # Adicionar à fila de atendimento humano
        queue_item = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'user_info': conversation.get('client_data', {}).get('name') or f'Cliente #{conversation_id[-6:]}',
            'client_email': conversation.get('client_data', {}).get('email', ''),
            'waiting_time': '0m'
        }
        chatbot.human_queue.append(queue_item)
        print(f"[TRANSFER] Conversa {conversation_id} adicionada à fila")
    
    # Adicionar mensagem do sistema
    conversation['messages'].append({
        'type': 'system',
        'content': 'Transferindo para atendente humano. Aguarde um momento...',
        'timestamp': datetime.now().isoformat()
    })
    
    queue_position = len([item for item in chatbot.human_queue if not item.get('assigned_agent')])
    
    return jsonify({
        'response': f'Você foi transferido para nossa equipe de atendimento. Posição na fila: {queue_position}. Um agente humano entrará em contato em breve.',
        'type': 'system',
        'transferred': True,
        'queue_position': queue_position
    })

def get_suggestions(user_message):
    """Gera sugestões baseadas na mensagem do usuário (melhorada)"""
    suggestions = []
    
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ['preço', 'valor', 'custo', 'quanto custa']):
        suggestions = ['Ver tabela de preços', 'Solicitar orçamento', 'Falar sobre descontos']
    elif any(word in message_lower for word in ['problema', 'erro', 'bug', 'não funciona']):
        suggestions = ['Reportar problema', 'Ver soluções comuns', 'Contato técnico']
    elif any(word in message_lower for word in ['produto', 'serviço', 'oferece']):
        suggestions = ['Ver catálogo', 'Especificações técnicas', 'Comparar produtos']
    elif any(word in message_lower for word in ['contato', 'telefone', 'email', 'endereço']):
        suggestions = ['Informações de contato', 'Horário de funcionamento', 'Localização']
    elif any(word in message_lower for word in ['ajuda', 'suporte', 'dúvida']):
        suggestions = ['Central de ajuda', 'FAQ', 'Falar com atendente']
    else:
        suggestions = ['Como posso ajudar?', 'Falar com atendente', 'Ver FAQ']
    
    return suggestions[:3]  # Máximo 3 sugestões

@chat_app.route('/admin/conversations')
def admin_conversations():
    """Painel admin para monitorar conversas"""
    active_conversations = len([c for c in chatbot.conversations.values() if c['status'] == 'active'])
    human_queue_size = len(chatbot.human_queue)
    
    return jsonify({
        'active_conversations': active_conversations,
        'human_queue': human_queue_size,
        'model_loaded': chatbot.model_loaded,
        'total_conversations': len(chatbot.conversations)
    })

@chat_app.route('/admin')
def admin_panel():
    """Painel administrativo para atendentes"""
    return render_template('admin_panel.html')

@chat_app.route('/admin/queue')
def get_queue():
    """Retorna fila de atendimento com informações do cliente"""
    try:
        print(f"[ADMIN] Consultando fila. Total na fila: {len(chatbot.human_queue)}")
        
        # Filtrar conversas por status e calcular tempo de espera
        queue = []
        for item in chatbot.human_queue:
            if not item.get('assigned_agent'):
                # Calcular tempo de espera
                start_time = datetime.fromisoformat(item['timestamp'])
                waiting_seconds = (datetime.now() - start_time).total_seconds()
                waiting_time = f"{int(waiting_seconds // 60)}min" if waiting_seconds >= 60 else f"{int(waiting_seconds)}s"
                
                # Buscar informações adicionais da conversa
                conversation_id = item['conversation_id']
                conversation = chatbot.conversations.get(conversation_id, {})
                client_data = conversation.get('client_data', {})
                
                # Verificar se é cliente recorrente
                client_email = client_data.get('email', '')
                is_returning_client = False
                previous_conversations_count = 0
                
                if client_email:
                    previous_conversations = get_client_history_by_email(client_email)
                    previous_conversations_count = len(previous_conversations)
                    is_returning_client = previous_conversations_count > 0
                
                queue_item = {
                    'conversation_id': conversation_id,
                    'timestamp': item['timestamp'],
                    'user_info': item.get('user_info', 'Cliente'),
                    'client_email': client_email,
                    'client_name': client_data.get('name', ''),
                    'client_phone': client_data.get('phone', ''),
                    'is_returning_client': is_returning_client,
                    'previous_conversations_count': previous_conversations_count,
                    'waiting_time': waiting_time,
                    'priority': 'high' if is_returning_client else 'normal'
                }
                queue.append(queue_item)
        
        # Ordenar fila - clientes recorrentes primeiro
        queue.sort(key=lambda x: (not x['is_returning_client'], x['timestamp']))
        
        # Conversas ativas com agentes
        active_conversations = []
        for conv_id, conv in chatbot.conversations.items():
            if conv.get('assigned_agent') and conv.get('status') == 'active':
                agent_id = conv['assigned_agent']
                agent_name = conv.get('agent_name', agent_id)
                
                if agent_id in chatbot.human_agents:
                    agent_name = chatbot.human_agents[agent_id]['name']
                
                active_conversations.append({
                    'conversation_id': conv_id,
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'start_time': conv['start_time'],
                    'agent_start_time': conv.get('agent_start_time'),
                    'client_email': conv.get('client_data', {}).get('email', ''),
                    'client_name': conv.get('client_data', {}).get('name', '')
                })
        
        print(f"[ADMIN] Retornando {len(queue)} na fila e {len(active_conversations)} ativas")
        
        return jsonify({
            'queue': queue,
            'active_conversations': active_conversations,
            'agents_online': len(chatbot.human_agents)
        })
    except Exception as e:
        print(f"[ADMIN] Erro ao buscar fila: {e}")
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/agent_login', methods=['POST'])
def agent_login():
    """Registra/atualiza informações do agente"""
    try:
        data = request.json
        agent_id = data.get('agent_id')
        agent_name = data.get('agent_name', agent_id)
        
        if not agent_id:
            return jsonify({'success': False, 'message': 'ID do agente é obrigatório'})
        
        # Registrar ou atualizar agente
        chatbot.human_agents[agent_id] = {
            'name': agent_name,
            'login_time': datetime.now().isoformat(),
            'status': 'available',
            'active_conversations': 0
        }
        
        print(f"[AGENT] Agente {agent_name} ({agent_id}) fez login")
        
        return jsonify({
            'success': True,
            'agent_info': chatbot.human_agents[agent_id]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/admin/conversation/<conversation_id>')
def get_conversation(conversation_id):
    """Retorna detalhes de uma conversa específica com histórico do cliente"""
    try:
        if conversation_id in chatbot.conversations:
            conversation = chatbot.conversations[conversation_id]
            
            # Adicionar informações do agente se disponível
            agent_id = conversation.get('assigned_agent')
            if agent_id and agent_id in chatbot.human_agents:
                conversation['agent_info'] = chatbot.human_agents[agent_id]
            
            # Incluir histórico do cliente se disponível
            client_email = conversation.get('client_data', {}).get('email', '')
            if client_email:
                conversation['client_full_history'] = get_client_history_by_email(client_email)
            
            return jsonify({'conversation': conversation})
        else:
            return jsonify({'error': 'Conversa não encontrada'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/send_message', methods=['POST'])
def admin_send_message():
    """Atendente envia mensagem para o cliente (melhorada para textos longos)"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        message = data.get('message', '').strip()
        agent_id = data.get('agent_id')
        msg_type = data.get('type', 'agent')
        
        if conversation_id not in chatbot.conversations:
            return jsonify({'success': False, 'message': 'Conversa não encontrada'})
        
        conversation = chatbot.conversations[conversation_id]
        
        # Verificar se o agente tem permissão
        if conversation.get('assigned_agent') != agent_id and msg_type != 'system':
            return jsonify({'success': False, 'message': 'Acesso negado'})
        
        # Verificar tamanho da mensagem (limite mais alto)
        if len(message) > 5000:  # Limite aumentado para 5000 caracteres
            return jsonify({'success': False, 'message': 'Mensagem muito longa (máximo 5000 caracteres)'})
        
        # Adicionar mensagem com timestamp preciso
        new_message = {
            'type': msg_type,
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id if msg_type == 'agent' else None,
            'character_count': len(message)  # Para estatísticas
        }
        
        conversation['messages'].append(new_message)
        
        print(f"Mensagem do agente enviada ({len(message)} chars): {message[:100]}...")  # Debug
        
        return jsonify({'success': True, 'message': new_message})
    except Exception as e:
        print(f"Erro ao enviar mensagem do agente: {e}")  # Debug
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/admin/accept', methods=['POST'])
def accept_conversation():
    """Atendente aceita uma conversa da fila (com controle de tempo)"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        agent_id = data.get('agent_id')
        agent_name = data.get('agent_name', agent_id)
        
        print(f"[ADMIN] Tentativa de aceitar conversa {conversation_id} por agente {agent_name} ({agent_id})")
        
        if conversation_id not in chatbot.conversations:
            print(f"[ADMIN] Conversa {conversation_id} não encontrada")
            return jsonify({'success': False, 'message': 'Conversa não encontrada'})
        
        # Atualizar informações do agente
        if agent_id not in chatbot.human_agents:
            chatbot.human_agents[agent_id] = {
                'name': agent_name,
                'login_time': datetime.now().isoformat(),
                'status': 'busy',
                'active_conversations': 0
            }
        
        chatbot.human_agents[agent_id]['status'] = 'busy'
        chatbot.human_agents[agent_id]['active_conversations'] += 1
        
        # Marcar conversa como aceita e iniciar cronômetro humano
        conversation = chatbot.conversations[conversation_id]
        human_start_time = datetime.now()
        
        conversation['assigned_agent'] = agent_id
        conversation['agent_name'] = agent_name
        conversation['agent_start_time'] = human_start_time.isoformat()
        conversation['status'] = 'active'
        
        # Atualizar métricas de tempo
        conversation['timing_metrics']['human_start_time'] = human_start_time.isoformat()
        
        # Remover da fila
        original_queue_size = len(chatbot.human_queue)
        chatbot.human_queue = [item for item in chatbot.human_queue 
                              if item['conversation_id'] != conversation_id]
        
        print(f"[ADMIN] Fila reduzida de {original_queue_size} para {len(chatbot.human_queue)}")
        
        # Buscar histórico do cliente se disponível
        client_email = conversation.get('client_data', {}).get('email', '')
        client_history_message = ""
        
        if client_email:
            previous_conversations = get_client_history_by_email(client_email)
            if previous_conversations:
                history_count = len(previous_conversations)
                last_conversation = previous_conversations[0]
                
                client_history_message = f"\n\n📋 HISTÓRICO DO CLIENTE ({client_email}):\n"
                client_history_message += f"• {history_count} conversa(s) anterior(es)\n"
                client_history_message += f"• Último contato: {format_date(last_conversation.get('date'))}\n"
                
                if last_conversation.get('transferred_to_human'):
                    client_history_message += f"• Última conversa: Transferida para humano\n"
                    if last_conversation.get('human_time_minutes', 0) > 0:
                        client_history_message += f"• Tempo de atendimento anterior: {last_conversation['human_time_minutes']} min\n"
                
                if last_conversation.get('satisfaction') != 'unknown':
                    client_history_message += f"• Satisfação anterior: {last_conversation['satisfaction']}\n"
                
                if last_conversation.get('summary'):
                    client_history_message += f"• Último assunto: {last_conversation['summary'][:100]}...\n"
        
        # Adicionar mensagem do sistema com informações do agente e histórico
        system_message = f"Atendente {agent_name} assumiu esta conversa. Como posso ajudá-lo?"
        if client_history_message:
            system_message += client_history_message
        
        conversation['messages'].append({
            'type': 'system',
            'content': system_message,
            'timestamp': human_start_time.isoformat(),
            'agent_info': {
                'agent_id': agent_id,
                'agent_name': agent_name,
                'client_history_shown': bool(client_history_message)
            }
        })
        
        print(f"[ADMIN] Conversa {conversation_id} aceita com sucesso por {agent_name}")
        
        return jsonify({
            'success': True,
            'agent_start_time': human_start_time.isoformat(),
            'agent_name': agent_name,
            'client_history': previous_conversations if client_email else []
        })
    except Exception as e:
        print(f"[ADMIN] Erro ao aceitar conversa: {e}")
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/admin/end_conversation', methods=['POST'])
def admin_end_conversation():
    """Atendente encerra uma conversa (com cálculo de tempo total)"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        agent_id = data.get('agent_id')
        
        if conversation_id not in chatbot.conversations:
            return jsonify({'success': False, 'message': 'Conversa não encontrada'})
        
        conversation = chatbot.conversations[conversation_id]
        
        # Verificar permissão
        if conversation.get('assigned_agent') != agent_id:
            return jsonify({'success': False, 'message': 'Acesso negado'})
        
        # Obter nome do agente
        agent_name = conversation.get('agent_name', agent_id)
        if agent_id in chatbot.human_agents:
            agent_name = chatbot.human_agents[agent_id]['name']
            chatbot.human_agents[agent_id]['active_conversations'] -= 1
            if chatbot.human_agents[agent_id]['active_conversations'] <= 0:
                chatbot.human_agents[agent_id]['status'] = 'available'
        
        # Calcular tempo total de atendimento humano
        human_end_time = datetime.now()
        human_start_time_str = conversation.get('timing_metrics', {}).get('human_start_time')
        
        total_human_time_seconds = 0
        if human_start_time_str:
            try:
                human_start_time = datetime.fromisoformat(human_start_time_str)
                total_human_time_seconds = (human_end_time - human_start_time).total_seconds()
            except Exception as e:
                print(f"Erro ao calcular tempo humano: {e}")
        
        # Atualizar métricas finais
        conversation['timing_metrics']['human_end_time'] = human_end_time.isoformat()
        conversation['timing_metrics']['total_human_time_seconds'] = total_human_time_seconds
        
        # Marcar como encerrada
        conversation['status'] = 'completed'
        conversation['end_time'] = human_end_time.isoformat()
        conversation['ended_by'] = 'agent'
        
        # Adicionar mensagem de encerramento com métricas
        human_time_minutes = round(total_human_time_seconds / 60, 1)
        conversation['messages'].append({
            'type': 'system',
            'content': f'Conversa encerrada por {agent_name}. Tempo de atendimento: {human_time_minutes} minutos.',
            'timestamp': human_end_time.isoformat()
        })
        
        # Salvar dados da conversa
        saved_file = chatbot.save_conversation_data(conversation_id)
        
        return jsonify({
            'success': True,
            'saved_file': saved_file,
            'human_time_minutes': human_time_minutes,
            'agent_name': agent_name,
            'metrics': {
                'total_human_time_seconds': total_human_time_seconds,
                'human_time_minutes': human_time_minutes,
                'total_messages': len(conversation.get('messages', [])),
                'client_email': conversation.get('client_data', {}).get('email', ''),
                'agent_name': agent_name
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/admin/client_history/<email>')
def get_client_history_admin(email):
    """Retorna histórico completo de um cliente por email (para admin)"""
    try:
        if not email or '@' not in email:
            return jsonify({'error': 'Email inválido'}), 400
        
        history = get_client_history_by_email(email.strip().lower())
        
        # Calcular estatísticas do cliente
        total_conversations = len(history)
        total_human_time = sum(conv.get('human_time_minutes', 0) for conv in history)
        transferred_count = sum(1 for conv in history if conv.get('transferred_to_human', False))
        
        # Satisfação média
        satisfactions = [conv.get('satisfaction') for conv in history if conv.get('satisfaction') not in ['unknown', None]]
        avg_satisfaction = None
        if satisfactions:
            numeric_satisfactions = []
            for sat in satisfactions:
                if isinstance(sat, (int, float)):
                    numeric_satisfactions.append(sat)
                elif isinstance(sat, str) and sat.isdigit():
                    numeric_satisfactions.append(int(sat))
            
            if numeric_satisfactions:
                avg_satisfaction = round(sum(numeric_satisfactions) / len(numeric_satisfactions), 1)
        
        client_stats = {
            'total_conversations': total_conversations,
            'total_human_time_minutes': round(total_human_time, 1),
            'avg_human_time_per_conversation': round(total_human_time / total_conversations, 1) if total_conversations > 0 else 0,
            'transferred_to_human_count': transferred_count,
            'transfer_rate': round((transferred_count / total_conversations) * 100, 1) if total_conversations > 0 else 0,
            'avg_satisfaction': avg_satisfaction,
            'last_contact': history[0]['date'] if history else None
        }
        
        return jsonify({
            'client_email': email,
            'conversations': history,
            'stats': client_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/time_metrics')
def get_time_metrics():
    """Retorna métricas de tempo de atendimento"""
    try:
        all_conversations = []
        
        # Coletar de conversas na memória
        for conv in chatbot.conversations.values():
            if conv.get('status') == 'completed' and conv.get('assigned_agent'):
                all_conversations.append(conv)
        
        # Coletar de arquivos salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        if saved_conv.get('assigned_agent'):
                            all_conversations.append(saved_conv)
                            
                    except Exception as e:
                        print(f"Erro ao processar {filename} para métricas: {e}")
        
        # Calcular métricas
        if not all_conversations:
            return jsonify({
                'total_conversations_with_human': 0,
                'avg_human_time_minutes': 0,
                'min_human_time_minutes': 0,
                'max_human_time_minutes': 0,
                'total_human_hours': 0,
                'conversations_by_agent': {}
            })
        
        human_times = []
        agent_stats = {}
        
        for conv in all_conversations:
            human_time_seconds = conv.get('timing_metrics', {}).get('total_human_time_seconds', 0)
            if human_time_seconds > 0:
                human_time_minutes = human_time_seconds / 60
                human_times.append(human_time_minutes)
                
                agent_id = conv.get('assigned_agent')
                if agent_id:
                    if agent_id not in agent_stats:
                        agent_stats[agent_id] = {
                            'total_conversations': 0,
                            'total_time_minutes': 0,
                            'avg_time_minutes': 0
                        }
                    
                    agent_stats[agent_id]['total_conversations'] += 1
                    agent_stats[agent_id]['total_time_minutes'] += human_time_minutes
        
        # Calcular médias por agente
        for agent_id in agent_stats:
            stats = agent_stats[agent_id]
            stats['avg_time_minutes'] = round(stats['total_time_minutes'] / stats['total_conversations'], 1)
            stats['total_time_minutes'] = round(stats['total_time_minutes'], 1)
        
        metrics = {
            'total_conversations_with_human': len(human_times),
            'avg_human_time_minutes': round(sum(human_times) / len(human_times), 1) if human_times else 0,
            'min_human_time_minutes': round(min(human_times), 1) if human_times else 0,
            'max_human_time_minutes': round(max(human_times), 1) if human_times else 0,
            'total_human_hours': round(sum(human_times) / 60, 1),
            'conversations_by_agent': agent_stats
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/agents')
def get_agents_status():
    """Retorna status de todos os agentes"""
    try:
        agents_list = []
        
        for agent_id, agent_info in chatbot.human_agents.items():
            agents_list.append({
                'agent_id': agent_id,
                'name': agent_info['name'],
                'status': agent_info['status'],
                'login_time': agent_info['login_time'],
                'active_conversations': agent_info['active_conversations']
            })
        
        return jsonify({
            'agents': agents_list,
            'total_agents': len(agents_list),
            'available_agents': len([a for a in agents_list if a['status'] == 'available']),
            'busy_agents': len([a for a in agents_list if a['status'] == 'busy'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/clients')
def get_all_clients():
    """Retorna lista de todos os clientes com estatísticas"""
    try:
        clients = {}
        
        # Coletar emails de conversas na memória
        for conv in chatbot.conversations.values():
            email = conv.get('client_data', {}).get('email', '').strip().lower()
            if email and '@' in email:
                if email not in clients:
                    clients[email] = {
                        'email': email,
                        'name': conv.get('client_data', {}).get('name', ''),
                        'phone': conv.get('client_data', {}).get('phone', ''),
                        'conversations': [],
                        'last_contact': None
                    }
                
                clients[email]['conversations'].append({
                    'conversation_id': conv['id'],
                    'date': conv.get('start_time'),
                    'status': conv.get('status', 'unknown'),
                    'human_time_minutes': round(conv.get('timing_metrics', {}).get('total_human_time_seconds', 0) / 60, 1)
                })
        
        # Coletar emails de arquivos salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        email = saved_conv.get('client_data', {}).get('email', '').strip().lower()
                        if email and '@' in email:
                            if email not in clients:
                                clients[email] = {
                                    'email': email,
                                    'name': saved_conv.get('client_data', {}).get('name', ''),
                                    'phone': saved_conv.get('client_data', {}).get('phone', ''),
                                    'conversations': [],
                                    'last_contact': None
                                }
                            
                            human_time_seconds = saved_conv.get('timing_metrics', {}).get('total_human_time_seconds', 0)
                            clients[email]['conversations'].append({
                                'conversation_id': saved_conv.get('conversation_id'),
                                'date': saved_conv.get('start_time'),
                                'status': 'completed',
                                'human_time_minutes': round(human_time_seconds / 60, 1)
                            })
                            
                    except Exception as e:
                        print(f"Erro ao processar {filename} para lista de clientes: {e}")
        
        # Calcular estatísticas e último contato para cada cliente
        client_list = []
        for email, client_data in clients.items():
            conversations = client_data['conversations']
            conversations.sort(key=lambda x: x['date'] or '', reverse=True)
            
            total_conversations = len(conversations)
            total_human_time = sum(conv['human_time_minutes'] for conv in conversations)
            last_contact = conversations[0]['date'] if conversations else None
            
            client_summary = {
                'email': email,
                'name': client_data['name'],
                'phone': client_data['phone'],
                'total_conversations': total_conversations,
                'total_human_time_minutes': round(total_human_time, 1),
                'avg_human_time_minutes': round(total_human_time / total_conversations, 1) if total_conversations > 0 else 0,
                'last_contact': last_contact,
                'recent_conversations': conversations[:3]
            }
            
            client_list.append(client_summary)
        
        client_list.sort(key=lambda x: x['last_contact'] or '', reverse=True)
        
        return jsonify({
            'clients': client_list,
            'total_clients': len(client_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/stats')
def admin_stats():
    """Estatísticas para o painel administrativo"""
    try:
        queue_size = len([item for item in chatbot.human_queue if not item.get('assigned_agent')])
        active_conversations = len([conv for conv in chatbot.conversations.values() 
                                  if conv.get('assigned_agent') and conv.get('status') == 'active'])
        
        total_conversations = len(chatbot.conversations)
        
        print(f"[STATS] Fila: {queue_size}, Ativas: {active_conversations}, Total: {total_conversations}")
        
        response = jsonify({
            'queue_size': queue_size,
            'active_conversations': active_conversations,
            'total_conversations': total_conversations,
            'model_loaded': chatbot.model_loaded,
            'model_status': 'loaded' if chatbot.model_loaded else 'fallback_mode'
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
        return response
    except Exception as e:
        print(f"[STATS] Erro: {e}")
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/stats', methods=['OPTIONS'])
def admin_stats_options():
    """Handle preflight OPTIONS request"""
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@chat_app.route('/chat/poll/<conversation_id>')
def poll_messages(conversation_id):
    """Polling para receber novas mensagens (usado pelo cliente)"""
    try:
        if conversation_id not in chatbot.conversations:
            return jsonify({'error': 'Conversa não encontrada'}), 404
        
        conversation = chatbot.conversations[conversation_id]
        
        new_messages = []
        last_check = request.args.get('last_check', '0')
        
        try:
            last_check_time = datetime.fromisoformat(last_check) if last_check != '0' else datetime.min
        except:
            last_check_time = datetime.min
        
        for msg in conversation['messages']:
            try:
                msg_time = datetime.fromisoformat(msg['timestamp'])
                if msg_time > last_check_time and msg['type'] in ['agent', 'system']:
                    new_messages.append(msg)
            except:
                continue
        
        return jsonify({
            'new_messages': new_messages,
            'conversation_ended': conversation.get('status') == 'completed',
            'last_update': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/history')
def get_conversation_history():
    """Retorna histórico de conversas encerradas"""
    try:
        history = []
        
        # Buscar em conversas encerradas na memória
        for conv_id, conv in chatbot.conversations.items():
            if conv.get('status') == 'completed':
                history.append({
                    'conversation_id': conv_id,
                    'client_data': conv.get('client_data', {}),
                    'start_time': conv.get('start_time'),
                    'end_time': conv.get('end_time'),
                    'satisfaction': conv.get('satisfaction', 'unknown'),
                    'total_messages': len(conv.get('messages', [])),
                    'assigned_agent': conv.get('assigned_agent'),
                    'transferred_to_human': conv.get('transferred_to_human', False)
                })
        
        # Buscar em arquivos salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        history.append({
                            'conversation_id': saved_conv.get('conversation_id'),
                            'client_data': saved_conv.get('client_data', {}),
                            'start_time': saved_conv.get('start_time'),
                            'end_time': saved_conv.get('end_time'),
                            'satisfaction': saved_conv.get('satisfaction', 'unknown'),
                            'total_messages': saved_conv.get('total_messages', 0),
                            'assigned_agent': saved_conv.get('assigned_agent'),
                            'transferred_to_human': saved_conv.get('transferred_to_human', False),
                            'from_file': True
                        })
                        
                    except Exception as e:
                        print(f"Erro ao processar {filename}: {e}")
        
        # Ordenar por data de encerramento (mais recente primeiro)
        history.sort(key=lambda x: x.get('end_time', ''), reverse=True)
        
        return jsonify({'conversations': history[:50]})  # Últimas 50 conversas
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/history/<conversation_id>')
def get_history_conversation(conversation_id):
    """Retorna conversa específica do histórico"""
    try:
        # Primeiro verificar se está na memória
        if conversation_id in chatbot.conversations:
            conversation = chatbot.conversations[conversation_id]
            return jsonify({'conversation': conversation})
        
        # Buscar em arquivos salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json') and conversation_id in filename:
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        if saved_conv.get('conversation_id') == conversation_id:
                            conversation = {
                                'id': conversation_id,
                                'start_time': saved_conv.get('start_time'),
                                'end_time': saved_conv.get('end_time'),
                                'client_data': saved_conv.get('client_data', {}),
                                'messages': saved_conv.get('full_chat_history', []),
                                'status': 'completed',
                                'satisfaction': saved_conv.get('satisfaction'),
                                'assigned_agent': saved_conv.get('assigned_agent'),
                                'transferred_to_human': saved_conv.get('transferred_to_human', False)
                            }
                            return jsonify({'conversation': conversation})
                            
                    except Exception as e:
                        print(f"Erro ao ler arquivo {filename}: {e}")
        
        return jsonify({'error': 'Conversa não encontrada'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Função para limpeza adequada na saída
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
import atexit
atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    try:
        print("Iniciando Chat Widget...")
        print("Acesse: http://localhost:5001/chat")
        print("Admin: http://localhost:5001/admin")
        print("CORS configurado para localhost:5000 e localhost:5001")
        print(f"Status do modelo: {'Carregado' if chatbot.model_loaded else 'Modo fallback'}")
        
        chat_app.run(
            debug=False,
            port=5001, 
            host='0.0.0.0',
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        cleanup_on_exit()
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
        cleanup_on_exit()
