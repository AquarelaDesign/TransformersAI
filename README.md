# 🤖 TransformersAI - Sistema de Atendimento Inteligente

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistema completo de atendimento ao cliente com IA e escalação para humanos**

[Demo](#demo) • [Instalação](#instalação) • [Documentação](#documentação) • [Contribuir](#contribuindo)

</div>

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [API Documentation](#api-documentation)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuindo](#contribuindo)
- [Roadmap](#roadmap)
- [License](#license)

## 🚀 Sobre o Projeto

O **TransformersAI** é um sistema completo de atendimento ao cliente que combina inteligência artificial com atendimento humano, proporcionando uma experiência fluida e eficiente para clientes e operadores.

### 🎯 Objetivos

- **Automatizar** 80% dos atendimentos com IA
- **Escalar** para humanos quando necessário
- **Aprender** continuamente com interações
- **Otimizar** tempo de resposta e satisfação
- **Integrar** facilmente com sites existentes

### 🌟 Diferenciais

- ✅ **IA Híbrida**: Combina respostas automáticas com atendimento humano
- ✅ **Auto-aprendizado**: Modelo retreina com dados coletados
- ✅ **Interface Moderna**: Design responsivo e intuitivo
- ✅ **Fácil Integração**: Widget JavaScript simples
- ✅ **Dashboard Completo**: Métricas e controle em tempo real

## 🔧 Funcionalidades

### 💬 Chat Widget
- **Interface responsiva** para sites
- **Coleta de dados** do cliente (nome, email, telefone)
- **Respostas automáticas** da IA
- **Escalação inteligente** para humanos
- **Avaliação de satisfação** integrada
- **Histórico** de conversas

### 👨‍💼 Painel Administrativo
- **Fila de atendimento** em tempo real
- **Chat em tempo real** com clientes
- **Respostas rápidas** personalizáveis
- **Informações do cliente** centralizadas
- **Histórico completo** de conversas
- **Métricas de performance**

### 🧠 Sistema de IA
- **Modelo Transformers** (GPT/BERT)
- **Fine-tuning** com dados coletados
- **Retreinamento automático** via interface
- **Backup de modelos** antes de retreinar
- **Classificação de intenções**
- **Geração de respostas** contextual

### 📊 Dashboard Principal
- **Métricas em tempo real** do sistema
- **Estatísticas de performance**
- **Gestão de dados** de treinamento
- **Sistema de retreinamento** via interface
- **Coleta de dados web** automatizada
- **Logs e monitoramento**

## 🛠 Tecnologias Utilizadas

### Backend
- **Python 3.8+** - Linguagem principal
- **Flask** - Framework web
- **Transformers (Hugging Face)** - Modelos de IA
- **PyTorch** - Deep Learning
- **BeautifulSoup** - Web scraping
- **SQLite** - Banco de dados
- **Requests** - Requisições HTTP

### Frontend
- **HTML5/CSS3** - Estrutura e estilo
- **JavaScript (ES6+)** - Interatividade
- **Bootstrap 5.3** - Framework CSS
- **Bootstrap Icons** - Ícones
- **Fetch API** - Comunicação com backend

### Ferramentas de Desenvolvimento
- **Git** - Controle de versão
- **VS Code** - IDE recomendada
- **GitHub** - Repositório e colaboração
- **Virtual Environment** - Isolamento de dependências

## 🏗 Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat Widget   │────│   Flask API     │────│  IA Transformer │
│   (Frontend)    │    │   (Backend)     │    │     (Model)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Admin Panel    │    │   Data Storage  │    │ Training Data   │
│  (Dashboard)    │    │   (JSON/SQLite) │    │ (Auto-Collect)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔄 Fluxo de Dados

1. **Cliente** acessa o chat widget
2. **IA** processa a mensagem e gera resposta
3. **Sistema** decide se escala para humano
4. **Dados** são coletados para retreinamento
5. **Modelo** é retreinado periodicamente

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passo a Passo

```bash
# 1. Clonar o repositório
git clone https://github.com/AquarelaDesign/TransformersAI.git
cd TransformersAI

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Executar o sistema
python chat_widget.py
```

### 🐳 Docker (Opcional)

```bash
# Construir imagem
docker build -t transformers-ai .

# Executar container
docker run -p 5000:5000 transformers-ai
```

## ⚙️ Configuração

### Variáveis de Ambiente

Criar arquivo `.env` na raiz do projeto:

```env
# Configurações do Flask
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=sua-chave-secreta-aqui

# Configurações da IA
MODEL_NAME=microsoft/DialoGPT-medium
MAX_LENGTH=1000
TEMPERATURE=0.7

# Configurações do banco
DATABASE_URL=sqlite:///chat_data.db

# Configurações de coleta de dados
ENABLE_WEB_SCRAPING=True
SCRAPING_INTERVAL=3600
```

### Configuração do Modelo

```python
# config.py
MODEL_CONFIG = {
    'name': 'microsoft/DialoGPT-medium',
    'max_length': 1000,
    'num_return_sequences': 1,
    'temperature': 0.7,
    'do_sample': True,
    'pad_token_id': 50256
}
```

## 🎮 Uso

### 1. Iniciar o Sistema

```bash
python chat_widget.py
```

### 2. Acessar Interfaces

- **Chat Widget**: `http://localhost:5000/chat`
- **Painel Admin**: `http://localhost:5000/admin`
- **Dashboard**: `http://localhost:5000/`

### 3. Integrar Widget no Site

```html
<!-- Adicionar ao seu site -->
<iframe src="http://localhost:5000/chat" 
        width="400" height="600"
        frameborder="0">
</iframe>
```

### 4. Configurar Atendentes

1. Acessar painel administrativo
2. Inserir nome do atendente
3. Aceitar conversas da fila
4. Usar respostas rápidas

### 5. Retreinar Modelo

1. Acessar dashboard principal
2. Verificar dados coletados
3. Configurar opções de retreinamento
4. Iniciar processo via interface

## 📡 API Documentation

### Endpoints Principais

#### Chat Endpoints

```http
POST /chat/send
Content-Type: application/json

{
  "message": "Olá, preciso de ajuda",
  "conversation_id": "uuid-here",
  "client_data": {
    "name": "João",
    "email": "joao@email.com"
  }
}
```

#### Admin Endpoints

```http
GET /admin/queue
Response: {
  "queue": [...],
  "active_conversations": [...]
}

POST /admin/accept
{
  "conversation_id": "uuid",
  "agent_id": "agent_123"
}
```

#### Training Endpoints

```http
POST /admin/retrain
{
  "use_all_data": true,
  "backup_model": true,
  "only_human": false
}

GET /chat/training_data
Response: {
  "total_conversations": 150,
  "total_interactions": 1200,
  "files": [...]
}
```

## 📁 Estrutura do Projeto

```
TransformersAI/
├── 📄 README.md                    # Este arquivo
├── 📄 requirements.txt             # Dependências Python
├── 📄 .gitignore                   # Arquivos ignorados
├── 📄 chat_widget.py               # Servidor principal Flask
├── 📄 chatbot.py                   # Lógica do chatbot IA
├── 📄 data_collector.py            # Coleta de dados web
├── 📄 index.html                   # Dashboard principal
├── 📁 chat_templates/              # Templates HTML
│   ├── 📄 chat_widget.html         # Interface do chat
│   ├── 📄 admin_panel.html         # Painel administrativo
│   └── 📄 base.html               # Template base
├── 📁 static/                      # Arquivos estáticos
│   ├── 📁 css/                    # Estilos CSS
│   ├── 📁 js/                     # Scripts JavaScript
│   └── 📁 images/                 # Imagens
├── 📁 chat_training_data/          # Dados de treinamento
├── 📁 collected_data/              # Dados coletados
├── 📁 models/                      # Modelos salvos
└── 📁 logs/                       # Logs do sistema
```

### 🔍 Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `chat_widget.py` | Servidor Flask principal com todas as rotas |
| `chatbot.py` | Classe do chatbot com IA Transformers |
| `data_collector.py` | Sistema de coleta de dados web |
| `index.html` | Dashboard com métricas e retreinamento |
| `chat_templates/chat_widget.html` | Interface do chat para clientes |
| `chat_templates/admin_panel.html` | Painel para atendentes humanos |

## 🤝 Contribuindo

Contribuições são bem-vindas! Siga estes passos:

### 1. Fork do Projeto

```bash
git clone https://github.com/SEU-USERNAME/TransformersAI.git
```

### 2. Criar Branch para Feature

```bash
git checkout -b feature/nova-funcionalidade
```

### 3. Fazer Commit das Mudanças

```bash
git commit -m "feat: adiciona nova funcionalidade incrível"
```

### 4. Push para Branch

```bash
git push origin feature/nova-funcionalidade
```

### 5. Abrir Pull Request

### 📝 Convenções

- **Commits**: Seguir [Conventional Commits](https://conventionalcommits.org/)
- **Código**: Seguir PEP 8 para Python
- **Documentação**: Atualizar README quando necessário

## 🗺 Roadmap

### 📅 Versão 2.0 (Próxima)

- [ ] **API RESTful** completa
- [ ] **Webhooks** para integrações
- [ ] **Multi-idiomas** no chat
- [ ] **Análise de sentimento** avançada
- [ ] **Integração com CRM** (Salesforce, HubSpot)

### 📅 Versão 2.1

- [ ] **App móvel** para atendentes
- [ ] **Chatbot por voz** (Speech-to-Text)
- [ ] **Analytics** avançado com gráficos
- [ ] **A/B Testing** para respostas
- [ ] **Escalabilidade** com Redis/PostgreSQL

### 📅 Versão 3.0

- [ ] **Multi-tenancy** (múltiplas empresas)
- [ ] **Marketplace** de plugins
- [ ] **AI Agents** especializados
- [ ] **Integração com GPT-4**
- [ ] **Deploy automático** com CI/CD

## 📊 Métricas do Projeto

- **Linhas de código**: ~2,500+
- **Arquivos Python**: 3 principais
- **Templates HTML**: 3 interfaces
- **Rotas API**: 15+ endpoints
- **Funcionalidades**: 20+ recursos

## 🔒 Segurança

- **Sanitização** de inputs do usuário
- **Validação** de dados recebidos
- **Rate limiting** nas APIs
- **Logs** de segurança
- **Dados sensíveis** não versionados

## 📈 Performance

- **Tempo de resposta**: < 500ms para IA
- **Concurrent users**: Suporta 100+ usuários
- **Memória**: ~200MB em execução
- **Storage**: Crescimento linear com dados

## 🐛 Problemas Conhecidos

- **Modelo grande**: Download inicial de ~500MB
- **Memória**: Requer mínimo 4GB RAM
- **GPU**: Recomendado para retreinamento

## 💡 FAQ

**Q: Como posso integrar no meu site WordPress?**
A: Use um iframe ou plugin personalizado com o endpoint `/chat`.

**Q: O modelo funciona em português?**
A: Sim, treina automaticamente com suas conversas em português.

**Q: Posso usar outros modelos de IA?**
A: Sim, modifique a configuração em `chatbot.py`.

**Q: Como fazer backup dos dados?**
A: Dados ficam em `chat_training_data/` - faça backup desta pasta.

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/AquarelaDesign/TransformersAI/issues)
- **Discussões**: [GitHub Discussions](https://github.com/AquarelaDesign/TransformersAI/discussions)
- **Email**: suporte@aquareladesign.com.br

## 📜 License

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">

**Feito com ❤️ por [AquarelaDesign](https://github.com/AquarelaDesign)**

⭐ Se este projeto te ajudou, considere dar uma estrela!

</div>
