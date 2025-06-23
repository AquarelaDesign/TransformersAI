import json
import os
from datetime import datetime

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

def get_suggestions(user_message):
    """Gera sugestões baseadas na mensagem do usuário"""
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
    
    return suggestions[:3]

def get_client_history_by_email(email, chatbot):
    """Busca histórico completo de conversas por email do cliente"""
    if not email or '@' not in email:
        return []
    
    email = email.strip().lower()
    history = []
    
    try:
        # 1. Buscar em conversas ativas/recentes na memória
        for conv_id, conv in chatbot.conversations.items():
            conv_email = conv.get('client_data', {}).get('email', '').strip().lower()
            if conv_email == email and conv_id != conv.get('id'):
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
                        
                        saved_email = saved_conv.get('client_data', {}).get('email', '').strip().lower()
                        
                        if saved_email == email:
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

def get_conversation_messages(conversation_id, chatbot):
    """Busca mensagens de uma conversa específica"""
    try:
        # Primeiro, verificar conversas ativas
        if conversation_id in chatbot.conversations:
            return chatbot.conversations[conversation_id].get('messages', [])
        
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
                            return saved_conv.get('full_chat_history', [])
                    except Exception as e:
                        print(f"Erro ao ler mensagens de {filename}: {e}")
        
        return []
    except Exception as e:
        print(f"Erro ao buscar mensagens da conversa {conversation_id}: {e}")
        return []

def save_conversation_to_file(conversation, human_agents):
    """Salva conversa para arquivo de treinamento"""
    try:
        conversation_id = conversation.get('id', 'unknown')
        
        # Preparar dados para treinamento
        training_data = []
        chat_history = []
        
        # Obter informações do agente
        agent_id = conversation.get('assigned_agent')
        agent_name = conversation.get('agent_name', agent_id)
        
        if agent_id and agent_id in human_agents:
            agent_name = human_agents[agent_id]['name']
        
        # Processar mensagens
        for i, msg in enumerate(conversation['messages']):
            if msg['type'] == 'user':
                response_msg = None
                if i + 1 < len(conversation['messages']):
                    next_msg = conversation['messages'][i + 1]
                    if next_msg['type'] in ['bot', 'agent']:
                        response_msg = next_msg
                
                if response_msg:
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
            
            chat_history.append({
                'type': msg['type'],
                'content': msg['content'],
                'timestamp': msg['timestamp'],
                'agent_id': msg.get('agent_id'),
                'agent_name': agent_name if msg.get('agent_id') == agent_id else None
            })
        
        # Calcular métricas
        ai_interactions = len([item for item in training_data if item['interaction_type'] == 'ai_response'])
        human_interactions = len([item for item in training_data if item['interaction_type'] == 'human_response'])
        
        timing_metrics = conversation.get('timing_metrics', {})
        human_time_seconds = timing_metrics.get('total_human_time_seconds', 0)
        
        # Criar resumo da conversa
        conversation_summary = {
            'conversation_id': conversation_id,
            'start_time': conversation['start_time'],
            'end_time': datetime.now().isoformat(),
            'client_data': conversation.get('client_data', {}),
            'total_messages': len(conversation['messages']),
            'satisfaction': conversation.get('satisfaction', 'unknown'),
            'transferred_to_human': conversation.get('transferred_to_human', False),
            'assigned_agent': agent_id,
            'agent_name': agent_name,
            'agent_start_time': conversation.get('agent_start_time'),
            'ended_by': conversation.get('ended_by', 'unknown'),
            'client_history_available': len(conversation.get('client_history', [])) > 0,
            'timing_metrics': {
                'total_human_time_seconds': human_time_seconds,
                'human_time_minutes': round(human_time_seconds / 60, 1),
            },
            'metrics': {
                'ai_interactions': ai_interactions,
                'human_interactions': human_interactions,
                'total_interactions': len(training_data),
            },
            'training_data': training_data,
            'full_chat_history': chat_history
        }
        
        # Salvar arquivo
        os.makedirs('chat_training_data', exist_ok=True)
        filename = f"chat_training_data/conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_summary, f, ensure_ascii=False, indent=2)
        
        print(f"Conversa salva para treinamento: {filename}")
        print(f"  - IA: {ai_interactions} interações")
        print(f"  - Humano: {human_interactions} interações")
        print(f"  - Agente: {agent_name} ({agent_id})")
        print(f"  - Tempo humano: {round(human_time_seconds / 60, 1)} minutos")
        print(f"  - Cliente: {conversation.get('client_data', {}).get('email', 'Sem email')}")
        
        return filename
        
    except Exception as e:
        print(f"Erro ao salvar conversa: {e}")
        return None
