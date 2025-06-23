from flask import request, jsonify
from datetime import datetime
import json
import os
from ..utils import get_client_history_by_email

def register_api_routes(app, chatbot):
    """Registra rotas da API e métricas"""
    
    @app.route('/admin/stats')
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

    @app.route('/admin/stats', methods=['OPTIONS'])
    def admin_stats_options():
        """Handle preflight OPTIONS request"""
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    @app.route('/admin/time_metrics')
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

    @app.route('/admin/client_history/<email>')
    def get_client_history_admin(email):
        """Retorna histórico completo de um cliente por email"""
        try:
            if not email or '@' not in email:
                return jsonify({'error': 'Email inválido'}), 400
            
            history = get_client_history_by_email(email.strip().lower(), chatbot)
            
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

    @app.route('/admin/agents')
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

    @app.route('/admin/clients')
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
            
            # Calcular estatísticas para cada cliente
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

    @app.route('/admin/history')
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

    @app.route('/admin/history/<conversation_id>')
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
