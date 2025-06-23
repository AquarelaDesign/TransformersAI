"""
Chat Widget - Sistema de Chat com IA e Atendimento Humano
Estrutura modular para facilitar manutenção
"""

from .app import create_chat_app
from .chatbot import ChatBot
from .routes import *
from .utils import *

__version__ = "1.0.0"
__author__ = "TransformersAI"

# Instância global do chatbot (será inicializada no app.py)
chatbot = None

def initialize_chatbot():
    """Inicializa a instância global do chatbot"""
    global chatbot
    if chatbot is None:
        chatbot = ChatBot()
    return chatbot
