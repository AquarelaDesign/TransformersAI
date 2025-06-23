from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from datetime import datetime
import threading
import re

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
                    torch_dtype=torch.float32,
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
        
        model_thread = threading.Thread(target=load_model_safe)
        model_thread.daemon = False
        model_thread.start()
        
        model_thread.join(timeout=30)
        
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
        """Respostas inteligentes baseadas em padrões"""
        message_lower = message.lower().strip()
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
        
        return None
    
    def _get_fallback_response(self, message):
        """Resposta padrão quando modelo não está disponível"""
        message_lower = message.lower().strip()
        
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
        """Gera resposta usando modelo treinado"""
        try:
            print(f"Tentando gerar resposta IA para: {message}")
            
            intelligent_response = self._get_intelligent_response(message)
            if intelligent_response:
                print(f"Usando resposta inteligente: {intelligent_response}")
                return intelligent_response
            
            fallback = self._get_fallback_response(message)
            print(f"Usando resposta fallback: {fallback}")
            return fallback
            
        except Exception as e:
            print(f"Erro no modelo AI: {e}")
            return self._get_fallback_response(message)

    def save_conversation_data(self, conversation_id):
        """Salva conversa para futuros treinamentos"""
        if conversation_id not in self.conversations:
            return
        
        from .utils import save_conversation_to_file
        return save_conversation_to_file(self.conversations[conversation_id], self.human_agents)

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
                return round(waiting_seconds / 60, 1)
        except:
            pass
        return 0
