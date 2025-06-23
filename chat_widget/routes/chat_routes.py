from flask import jsonify
from datetime import datetime

def handle_human_transfer(conversation_id, chatbot):
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