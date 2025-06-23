from flask import request, jsonify, render_template
from datetime import datetime
from ..utils import get_client_history_by_email, format_date, get_conversation_messages

def register_admin_routes(app, chatbot):
    """Registra rotas administrativas para atendentes"""
    
    @app.route('/admin')
    def admin_panel():
        """Painel administrativo para atendentes"""
        return render_template('admin_panel.html')

    @app.route('/admin/conversations')
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

    @app.route('/admin/queue')
    def get_queue():
        """Retorna fila de atendimento com informações do cliente"""
        try:
            print(f"[ADMIN] Consultando fila. Total na fila: {len(chatbot.human_queue)}")
            
            queue = []
            for item in chatbot.human_queue:
                if not item.get('assigned_agent'):
                    # Calcular tempo de espera
                    start_time = datetime.fromisoformat(item['timestamp'])
                    waiting_seconds = (datetime.now() - start_time).total_seconds()
                    waiting_time = f"{int(waiting_seconds // 60)}min" if waiting_seconds >= 60 else f"{int(waiting_seconds)}s"
                    
                    # Buscar informações da conversa
                    conversation_id = item['conversation_id']
                    conversation = chatbot.conversations.get(conversation_id, {})
                    client_data = conversation.get('client_data', {})
                    
                    # Verificar se é cliente recorrente
                    client_email = client_data.get('email', '')
                    is_returning_client = False
                    previous_conversations_count = 0
                    client_preview_history = []
                    
                    if client_email:
                        previous_conversations = get_client_history_by_email(client_email, chatbot)
                        previous_conversations_count = len(previous_conversations)
                        is_returning_client = previous_conversations_count > 0
                        
                        # Preparar preview do histórico
                        for prev_conv in previous_conversations[:2]:
                            client_preview_history.append({
                                'date': format_date(prev_conv.get('date')),
                                'summary': prev_conv.get('summary', 'Conversa anterior')[:50] + '...',
                                'transferred_to_human': prev_conv.get('transferred_to_human', False)
                            })
                    
                    queue_item = {
                        'conversation_id': conversation_id,
                        'timestamp': item['timestamp'],
                        'user_info': item.get('user_info', 'Cliente'),
                        'client_email': client_email,
                        'client_name': client_data.get('name', ''),
                        'client_phone': client_data.get('phone', ''),
                        'is_returning_client': is_returning_client,
                        'previous_conversations_count': previous_conversations_count,
                        'client_history_preview': client_preview_history,
                        'waiting_time': waiting_time,
                        'priority': 'high' if is_returning_client else 'normal'
                    }
                    queue.append(queue_item)
            
            # Ordenar fila - clientes recorrentes primeiro
            queue.sort(key=lambda x: (not x['is_returning_client'], x['timestamp']))
            
            # Conversas ativas com nomes corretos dos agentes
            active_conversations = []
            for conv_id, conv in chatbot.conversations.items():
                if conv.get('assigned_agent') and conv.get('status') == 'active':
                    agent_id = conv['assigned_agent']
                    agent_name = conv.get('agent_name', agent_id)
                    
                    if agent_id in chatbot.human_agents:
                        agent_name = chatbot.human_agents[agent_id]['name']
                        conv['agent_name'] = agent_name
                    
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

    @app.route('/admin/agent_login', methods=['POST'])
    def agent_login():
        """Registra/atualiza informações do agente"""
        try:
            data = request.json
            agent_id = data.get('agent_id')
            agent_name = data.get('agent_name', agent_id)
            
            if not agent_id:
                return jsonify({'success': False, 'message': 'ID do agente é obrigatório'})
            
            if not agent_name or agent_name == agent_id:
                agent_name = f"Agente {agent_id}"
            
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

    @app.route('/admin/accept', methods=['POST'])
    def accept_conversation():
        """Atendente aceita uma conversa da fila"""
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            agent_id = data.get('agent_id')
            agent_name = data.get('agent_name', agent_id)
            
            print(f"[ADMIN] Tentativa de aceitar conversa {conversation_id} por agente {agent_name} ({agent_id})")
            
            if conversation_id not in chatbot.conversations:
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
            
            # Marcar conversa como aceita
            conversation = chatbot.conversations[conversation_id]
            human_start_time = datetime.now()
            
            conversation['assigned_agent'] = agent_id
            conversation['agent_name'] = agent_name
            conversation['agent_start_time'] = human_start_time.isoformat()
            conversation['status'] = 'active'
            conversation['timing_metrics']['human_start_time'] = human_start_time.isoformat()
            
            # Remover da fila
            original_queue_size = len(chatbot.human_queue)
            chatbot.human_queue = [item for item in chatbot.human_queue 
                                  if item['conversation_id'] != conversation_id]
            
            print(f"[ADMIN] Fila reduzida de {original_queue_size} para {len(chatbot.human_queue)}")
            
            # Buscar histórico do cliente
            client_email = conversation.get('client_data', {}).get('email', '')
            client_history_message = ""
            
            if client_email:
                previous_conversations = get_client_history_by_email(client_email, chatbot)
                if previous_conversations:
                    history_count = len(previous_conversations)
                    last_conversation = previous_conversations[0]
                    
                    client_history_message = f"\n\n📋 HISTÓRICO DO CLIENTE ({client_email}):\n"
                    client_history_message += f"• {history_count} conversa(s) anterior(es)\n"
                    client_history_message += f"• Último contato: {format_date(last_conversation.get('date'))}\n"
                    
                    if last_conversation.get('transferred_to_human'):
                        client_history_message += f"• Última conversa: Transferida para humano\n"
                        if last_conversation.get('human_time_minutes', 0) > 0:
                            client_history_message += f"• Tempo anterior: {last_conversation['human_time_minutes']} min\n"
                    
                    if last_conversation.get('satisfaction') != 'unknown':
                        client_history_message += f"• Satisfação anterior: {last_conversation['satisfaction']}\n"
                    
                    if last_conversation.get('summary'):
                        client_history_message += f"• Último assunto: {last_conversation['summary'][:100]}...\n"
            
            # Adicionar mensagem do sistema
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

    @app.route('/admin/send_message', methods=['POST'])
    def admin_send_message():
        """Atendente envia mensagem para o cliente"""
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            message = data.get('message', '').strip()
            agent_id = data.get('agent_id')
            msg_type = data.get('type', 'agent')
            
            if conversation_id not in chatbot.conversations:
                return jsonify({'success': False, 'message': 'Conversa não encontrada'})
            
            conversation = chatbot.conversations[conversation_id]
            
            # Verificar permissão
            if conversation.get('assigned_agent') != agent_id and msg_type != 'system':
                return jsonify({'success': False, 'message': 'Acesso negado'})
            
            # Verificar tamanho da mensagem
            if len(message) > 5000:
                return jsonify({'success': False, 'message': 'Mensagem muito longa (máximo 5000 caracteres)'})

            # Obter nome do agente
            agent_name = conversation.get('agent_name', agent_id)
            if agent_id in chatbot.human_agents:
                agent_name = chatbot.human_agents[agent_id]['name']
            
            # Adicionar mensagem
            new_message = {
                'type': msg_type,
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id if msg_type == 'agent' else None,
                'character_count': len(message)
            }
            
            conversation['messages'].append(new_message)
            
            print(f"Mensagem do agente enviada ({len(message)} chars): {message[:100]}...")
            
            return jsonify({'success': True, 'message': new_message})
        except Exception as e:
            print(f"Erro ao enviar mensagem do agente: {e}")
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/admin/end_conversation', methods=['POST'])
    def admin_end_conversation():
        """Atendente encerra uma conversa"""
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
            
            # Calcular tempo de atendimento
            human_end_time = datetime.now()
            human_start_time_str = conversation.get('timing_metrics', {}).get('human_start_time')
            
            total_human_time_seconds = 0
            if human_start_time_str:
                try:
                    human_start_time = datetime.fromisoformat(human_start_time_str)
                    total_human_time_seconds = (human_end_time - human_start_time).total_seconds()
                except Exception as e:
                    print(f"Erro ao calcular tempo humano: {e}")
            
            # Atualizar métricas
            conversation['timing_metrics']['human_end_time'] = human_end_time.isoformat()
            conversation['timing_metrics']['total_human_time_seconds'] = total_human_time_seconds
            conversation['status'] = 'completed'
            conversation['end_time'] = human_end_time.isoformat()
            conversation['ended_by'] = 'agent'
            
            # Adicionar mensagem de encerramento
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

    @app.route('/admin/conversation/<conversation_id>')
    def get_conversation(conversation_id):
        """Retorna detalhes de uma conversa específica"""
        try:
            if conversation_id in chatbot.conversations:
                conversation = chatbot.conversations[conversation_id]
                
                # Adicionar informações do agente
                agent_id = conversation.get('assigned_agent')
                agent_name = conversation.get('agent_name', agent_id)
                
                if agent_id and agent_id in chatbot.human_agents:
                    conversation['agent_info'] = chatbot.human_agents[agent_id]
                    agent_name = chatbot.human_agents[agent_id].get('name', agent_name)
                    conversation['agent_name'] = agent_name
                
                # Incluir histórico do cliente
                client_email = conversation.get('client_data', {}).get('email', '')
                if client_email:
                    full_history = get_client_history_by_email(client_email, chatbot)
                    conversation['client_full_history'] = full_history
                    
                    # Adicionar mensagens do histórico
                    conversation['client_history_messages'] = []
                    
                    for hist_conv in full_history[:3]:
                        hist_messages = get_conversation_messages(hist_conv.get('conversation_id'), chatbot)
                        if hist_messages:
                            conversation['client_history_messages'].append({
                                'conversation_date': hist_conv.get('date'),
                                'conversation_id': hist_conv.get('conversation_id'),
                                'messages': hist_messages[:10],
                                'total_messages': len(hist_messages),
                                'summary': hist_conv.get('summary', 'Conversa anterior')
                            })
                
                return jsonify({'conversation': conversation})
            else:
                return jsonify({'error': 'Conversa não encontrada'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
