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
        self.human_agents = {}
        
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
        """Salva conversa para futuros treinamentos (melhorada com dados do cliente)"""
        if conversation_id not in self.conversations:
            return
        
        conversation = self.conversations[conversation_id]
        
        # Preparar dados para treinamento - incluindo diálogos com atendentes
        training_data = []
        chat_history = []
        
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
                        'agent_id': response_msg.get('agent_id') if response_msg['type'] == 'agent' else None
                    }
                    training_data.append(training_item)
            
            # Manter histórico completo
            chat_history.append({
                'type': msg['type'],
                'content': msg['content'],
                'timestamp': msg['timestamp'],
                'agent_id': msg.get('agent_id')
            })
        
        # Calcular métricas da conversa
        ai_interactions = len([item for item in training_data if item['interaction_type'] == 'ai_response'])
        human_interactions = len([item for item in training_data if item['interaction_type'] == 'human_response'])
        
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
            'assigned_agent': conversation.get('assigned_agent'),
            'agent_start_time': conversation.get('agent_start_time'),
            'ended_by': conversation.get('ended_by', 'unknown'),
            'metrics': {
                'ai_interactions': ai_interactions,
                'human_interactions': human_interactions,
                'total_interactions': len(training_data),
                'conversation_duration_minutes': self._calculate_duration(conversation)
            },
            'training_data': training_data,
            'full_chat_history': chat_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_summary, f, ensure_ascii=False, indent=2)
        
        print(f"Conversa salva para treinamento: {filename}")
        print(f"  - IA: {ai_interactions} interações")
        print(f"  - Humano: {human_interactions} interações")
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
    """Inicia nova conversa com dados do cliente"""
    data = request.json or {}
    client_data = data.get('client_data', {})
    
    conversation_id = str(uuid.uuid4())
    
    chatbot.conversations[conversation_id] = {
        'id': conversation_id,
        'start_time': datetime.now().isoformat(),
        'messages': [],
        'status': 'active',
        'transferred_to_human': False,
        'client_data': {
            'name': client_data.get('name', ''),
            'email': client_data.get('email', ''),
            'phone': client_data.get('phone', ''),
            'collected_at': datetime.now().isoformat()
        }
    }
    
    # Buscar conversas anteriores do cliente para referência
    previous_conversations = get_client_history(client_data.get('email', ''))
    if previous_conversations:
        chatbot.conversations[conversation_id]['client_history'] = previous_conversations
    
    return jsonify({
        'conversation_id': conversation_id,
        'message': 'Seja bem-vindo ao nosso atendimento! Como posso ajudá-lo hoje?',
        'status': 'connected'
    })

def get_client_history(email):
    """Busca histórico de conversas do cliente por email"""
    if not email:
        return []
    
    history = []
    try:
        # Buscar em conversas ativas
        for conv_id, conv in chatbot.conversations.items():
            if (conv.get('client_data', {}).get('email') == email and 
                conv.get('status') == 'completed'):
                history.append({
                    'conversation_id': conv_id,
                    'date': conv.get('start_time'),
                    'summary': get_conversation_summary(conv),
                    'satisfaction': conv.get('satisfaction', 'unknown')
                })
        
        # Buscar em arquivos de chat salvos
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(chat_data_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            saved_conv = json.load(f)
                        
                        # Verificar se é do mesmo cliente (pode não ter client_data em conversas antigas)
                        saved_email = None
                        if 'full_chat_history' in saved_conv:
                            for msg in saved_conv['full_chat_history']:
                                if msg.get('type') == 'system' and 'email' in msg.get('content', '').lower():
                                    # Tentar extrair email de mensagens do sistema
                                    pass
                        
                        # Por enquanto, adicionar conversas recentes como referência geral
                        if saved_conv.get('start_time'):
                            start_date = datetime.fromisoformat(saved_conv['start_time'])
                            if (datetime.now() - start_date).days <= 30:  # Últimos 30 dias
                                history.append({
                                    'conversation_id': saved_conv.get('conversation_id', filename),
                                    'date': saved_conv.get('start_time'),
                                    'summary': f"Conversa anterior - {saved_conv.get('total_messages', 0)} mensagens",
                                    'satisfaction': saved_conv.get('satisfaction', 'unknown')
                                })
                    except Exception as e:
                        print(f"Erro ao processar {filename}: {e}")
        
        # Ordenar por data (mais recente primeiro) e limitar a 5
        history.sort(key=lambda x: x['date'], reverse=True)
        return history[:5]
        
    except Exception as e:
        print(f"Erro ao buscar histórico do cliente: {e}")
        return []

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

# ...existing code...

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
    """Gerencia transferência para atendimento humano (corrigida)"""
    if conversation_id not in chatbot.conversations:
        return jsonify({'error': 'Conversa não encontrada'}), 404
        
    conversation = chatbot.conversations[conversation_id]
    
    # Marcar como transferida
    conversation['transferred_to_human'] = True
    
    # Verificar se já não está na fila
    already_in_queue = any(item['conversation_id'] == conversation_id for item in chatbot.human_queue)
    
    if not already_in_queue:
        # Adicionar à fila de atendimento humano
        queue_item = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'user_info': f'Cliente #{conversation_id[-6:]}',
            'waiting_time': '0m'
        }
        chatbot.human_queue.append(queue_item)
        print(f"[TRANSFER] Conversa {conversation_id} adicionada à fila")
        print(f"[TRANSFER] Fila atual: {len(chatbot.human_queue)} conversas")
    
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

@chat_app.route('/chat/rate', methods=['POST'])
def rate_message():
    """Avalia resposta do bot"""
    data = request.json
    conversation_id = data.get('conversation_id')
    message_index = data.get('message_index')
    rating = data.get('rating')  # 'positive', 'negative', 'neutral'
    
    if conversation_id in chatbot.conversations:
        messages = chatbot.conversations[conversation_id]['messages']
        if 0 <= message_index < len(messages):
            messages[message_index]['rating'] = rating
            
            return jsonify({'status': 'success'})
    
    return jsonify({'error': 'Avaliação não pôde ser salva'}), 400

@chat_app.route('/chat/end', methods=['POST'])
def end_conversation():
    """Finaliza conversa e salva dados"""
    data = request.json
    conversation_id = data.get('conversation_id')
    satisfaction = data.get('satisfaction', 'unknown')
    
    if conversation_id in chatbot.conversations:
        chatbot.conversations[conversation_id]['satisfaction'] = satisfaction
        chatbot.save_conversation_data(conversation_id)
        
        return jsonify({'status': 'success', 'message': 'Obrigado pelo feedback!'})
    
    return jsonify({'error': 'Conversa não encontrada'}), 404

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
    """Retorna fila de atendimento e conversas ativas (corrigida)"""
    try:
        print(f"[ADMIN] Consultando fila. Total na fila: {len(chatbot.human_queue)}")
        print(f"[ADMIN] Total de conversas: {len(chatbot.conversations)}")
        
        # Filtrar conversas por status e calcular tempo de espera
        queue = []
        for item in chatbot.human_queue:
            if not item.get('assigned_agent'):
                # Calcular tempo de espera
                start_time = datetime.fromisoformat(item['timestamp'])
                waiting_seconds = (datetime.now() - start_time).total_seconds()
                waiting_time = f"{int(waiting_seconds // 60)}min" if waiting_seconds >= 60 else f"{int(waiting_seconds)}s"
                
                queue_item = {
                    'conversation_id': item['conversation_id'],
                    'timestamp': item['timestamp'],
                    'user_info': item.get('user_info', 'Cliente'),
                    'waiting_time': waiting_time
                }
                queue.append(queue_item)
        
        # Conversas ativas com agentes
        active_conversations = []
        for conv_id, conv in chatbot.conversations.items():
            if conv.get('assigned_agent') and conv.get('status') == 'active':
                active_conversations.append({
                    'conversation_id': conv_id,
                    'agent_id': conv['assigned_agent'],
                    'start_time': conv['start_time'],
                    'agent_start_time': conv.get('agent_start_time')
                })
        
        print(f"[ADMIN] Retornando {len(queue)} na fila e {len(active_conversations)} ativas")
        
        return jsonify({
            'queue': queue,
            'active_conversations': active_conversations
        })
    except Exception as e:
        print(f"[ADMIN] Erro ao buscar fila: {e}")
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/accept', methods=['POST'])
def accept_conversation():
    """Atendente aceita uma conversa da fila (corrigida)"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        agent_id = data.get('agent_id')
        
        print(f"[ADMIN] Tentativa de aceitar conversa {conversation_id} por agente {agent_id}")
        
        if conversation_id not in chatbot.conversations:
            print(f"[ADMIN] Conversa {conversation_id} não encontrada")
            return jsonify({'success': False, 'message': 'Conversa não encontrada'})
        
        # Marcar conversa como aceita
        conversation = chatbot.conversations[conversation_id]
        conversation['assigned_agent'] = agent_id
        conversation['agent_start_time'] = datetime.now().isoformat()
        conversation['status'] = 'active'  # Garantir que está ativa
        
        # Remover da fila
        original_queue_size = len(chatbot.human_queue)
        chatbot.human_queue = [item for item in chatbot.human_queue 
                              if item['conversation_id'] != conversation_id]
        
        print(f"[ADMIN] Fila reduzida de {original_queue_size} para {len(chatbot.human_queue)}")
        
        # Adicionar mensagem do sistema
        conversation['messages'].append({
            'type': 'system',
            'content': 'Um atendente humano assumiu esta conversa. Como posso ajudá-lo?',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"[ADMIN] Conversa {conversation_id} aceita com sucesso por {agent_id}")
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"[ADMIN] Erro ao aceitar conversa: {e}")
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/admin/conversation/<conversation_id>')
def get_conversation(conversation_id):
    """Retorna detalhes de uma conversa específica"""
    try:
        if conversation_id in chatbot.conversations:
            conversation = chatbot.conversations[conversation_id]
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

@chat_app.route('/admin/end_conversation', methods=['POST'])
def admin_end_conversation():
    """Atendente encerra uma conversa (melhorada)"""
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
        
        # Marcar como encerrada
        conversation['status'] = 'completed'
        conversation['end_time'] = datetime.now().isoformat()
        conversation['ended_by'] = 'agent'
        
        # Adicionar mensagem de encerramento
        conversation['messages'].append({
            'type': 'system',
            'content': 'Conversa encerrada pelo atendente.',
            'timestamp': datetime.now().isoformat()
        })
        
        # Salvar dados da conversa e retornar nome do arquivo
        saved_file = chatbot.save_conversation_data(conversation_id)
        
        return jsonify({
            'success': True,
            'saved_file': saved_file,
            'metrics': conversation.get('metrics', {})
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@chat_app.route('/chat/training_data')
def get_chat_training_data():
    """Retorna informações sobre dados de treinamento do chat"""
    try:
        training_data_info = {
            'total_conversations': 0,
            'total_interactions': 0,
            'ai_interactions': 0,
            'human_interactions': 0,
            'files': []
        }
        
        chat_data_dir = 'chat_training_data'
        if os.path.exists(chat_data_dir):
            for filename in os.listdir(chat_data_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(chat_data_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        training_data_info['total_conversations'] += 1
                        
                        if 'metrics' in data:
                            metrics = data['metrics']
                            training_data_info['ai_interactions'] += metrics.get('ai_interactions', 0)
                            training_data_info['human_interactions'] += metrics.get('human_interactions', 0)
                            training_data_info['total_interactions'] += metrics.get('total_interactions', 0)
                        
                        # Informações do arquivo
                        file_info = {
                            'filename': filename,
                            'conversation_id': data.get('conversation_id', 'N/A'),
                            'start_time': data.get('start_time', 'N/A'),
                            'satisfaction': data.get('satisfaction', 'unknown'),
                            'transferred_to_human': data.get('transferred_to_human', False),
                            'metrics': data.get('metrics', {}),
                            'file_size': os.path.getsize(file_path)
                        }
                        training_data_info['files'].append(file_info)
                        
                    except Exception as e:
                        print(f"Erro ao processar {filename}: {e}")
        
        # Ordenar arquivos por data
        training_data_info['files'].sort(key=lambda x: x['start_time'], reverse=True)
        
        return jsonify(training_data_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/admin/stats')
def admin_stats():
    """Estatísticas para o painel administrativo (corrigida)"""
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
        
        # Adicionar headers CORS
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
        return response
    except Exception as e:
        print(f"[STATS] Erro: {e}")
        return jsonify({'error': str(e)}), 500

# Adicionar rota OPTIONS para preflight requests
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
        
        # Verificar se há mensagens novas do agente
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
                            # Reconstruir formato de conversa
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

@chat_app.route('/admin/retrain', methods=['POST'])
def start_retraining():
    """Inicia processo de retreinamento com dados coletados"""
    try:
        data = request.json or {}
        use_all_data = data.get('use_all_data', False)
        
        # Verificar se há dados suficientes
        chat_data_dir = 'chat_training_data'
        if not os.path.exists(chat_data_dir):
            return jsonify({'error': 'Nenhum dado de treinamento encontrado'}), 400
        
        training_files = [f for f in os.listdir(chat_data_dir) if f.endswith('.json')]
        if len(training_files) < 5:
            return jsonify({'error': 'Dados insuficientes para retreinamento (mínimo 5 conversas)'}), 400
        
        # Preparar dados para retreinamento
        training_data = []
        total_interactions = 0
        
        for filename in training_files:
            try:
                file_path = os.path.join(chat_data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                # Filtrar apenas interações bem avaliadas se não usar todos os dados
                for interaction in conv_data.get('training_data', []):
                    if use_all_data or interaction.get('rating', 'neutral') in ['positive', 'neutral']:
                        training_data.append({
                            'input': interaction['input'],
                            'output': interaction['output'],
                            'interaction_type': interaction['interaction_type'],
                            'rating': interaction.get('rating', 'neutral')
                        })
                        total_interactions += 1
                        
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
        
        if total_interactions < 10:
            return jsonify({'error': 'Interações insuficientes para retreinamento (mínimo 10)'}), 400
        
        # Salvar dados preparados para retreinamento
        retrain_file = f"retrain_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        retrain_path = os.path.join(chat_data_dir, retrain_file)
        
        with open(retrain_path, 'w', encoding='utf-8') as f:
            json.dump({
                'created_at': datetime.now().isoformat(),
                'total_interactions': total_interactions,
                'total_conversations': len(training_files),
                'use_all_data': use_all_data,
                'training_data': training_data
            }, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Dados preparados para retreinamento: {total_interactions} interações de {len(training_files)} conversas',
            'retrain_file': retrain_file,
            'total_interactions': total_interactions,
            'total_conversations': len(training_files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/chat/training_data')
def get_training_data():
    """Retorna dados de treinamento coletados para o dashboard"""
    try:
        chat_data_dir = 'chat_training_data'
        
        if not os.path.exists(chat_data_dir):
            return jsonify({
                'total_conversations': 0,
                'total_interactions': 0,
                'ai_interactions': 0,
                'human_interactions': 0,
                'files': []
            })
        
        files_data = []
        total_conversations = 0
        total_interactions = 0
        ai_interactions = 0
        human_interactions = 0
        
        for filename in os.listdir(chat_data_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(chat_data_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        conv_data = json.load(f)
                    
                    total_conversations += 1
                    
                    # Contar interações
                    file_ai_count = 0
                    file_human_count = 0
                    file_total_count = 0
                    
                    for interaction in conv_data.get('training_data', []):
                        file_total_count += 1
                        if interaction.get('interaction_type') == 'ai_response':
                            file_ai_count += 1
                        elif interaction.get('interaction_type') == 'human_response':
                            file_human_count += 1
                    
                    total_interactions += file_total_count
                    ai_interactions += file_ai_count
                    human_interactions += file_human_count
                    
                    # Dados do arquivo para exibição
                    files_data.append({
                        'conversation_id': conv_data.get('conversation_id'),
                        'start_time': conv_data.get('start_time'),
                        'end_time': conv_data.get('end_time'),
                        'satisfaction': conv_data.get('satisfaction', 0),
                        'transferred_to_human': conv_data.get('transferred_to_human', False),
                        'metrics': {
                            'total_interactions': file_total_count,
                            'ai_interactions': file_ai_count,
                            'human_interactions': file_human_count
                        }
                    })
                    
                except Exception as e:
                    print(f"Erro ao processar arquivo {filename}: {e}")
                    continue
        
        # Ordenar por data de início (mais recente primeiro)
        files_data.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return jsonify({
            'total_conversations': total_conversations,
            'total_interactions': total_interactions,
            'ai_interactions': ai_interactions,
            'human_interactions': human_interactions,
            'files': files_data
        })
        
    except Exception as e:
        print(f"Erro ao carregar dados de treinamento: {e}")
        return jsonify({
            'error': str(e),
            'total_conversations': 0,
            'total_interactions': 0,
            'ai_interactions': 0,
            'human_interactions': 0,
            'files': []
        }), 500

@chat_app.route('/api/collect_web_data', methods=['POST'])
def collect_web_data():
    """Rota para coletar dados de URLs web"""
    try:
        data = request.json or {}
        urls = data.get('urls', [])
        clean_data = data.get('clean_data', True)
        save_backup = data.get('save_backup', True)
        
        if not urls:
            return jsonify({'error': 'Nenhuma URL fornecida'}), 400
        
        # Validar URLs
        valid_urls = []
        for url in urls:
            url = url.strip()
            if url and (url.startswith('http://') or url.startswith('https://')):
                valid_urls.append(url)
        
        if not valid_urls:
            return jsonify({'error': 'Nenhuma URL válida fornecida'}), 400
        
        # Executar coleta
        results = data_collector.collect_web_data(valid_urls, clean_data, save_backup)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'Coleta concluída: {results["successful_collections"]} sucessos, {results["failed_collections"]} falhas'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_app.route('/api/training_data_status')
def get_training_data_status():
    """Obter status dos dados de treinamento coletados"""
    try:
        stats = data_collector.get_statistics()
        
        return jsonify({
            'collector_stats': stats,
            'training_data_available': len(data_collector.get_collected_data()) > 0,
            'data_collector_working': True  # Agora funciona
        })
        
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
        
        # Configurar Flask para não usar threads desnecessárias
        chat_app.run(
            debug=False,  # Desabilitar debug para evitar problemas com threads
            port=5001, 
            host='0.0.0.0',
            threaded=True,
            use_reloader=False  # Evitar reloader que pode causar problemas
        )
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        cleanup_on_exit()
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
        cleanup_on_exit()
