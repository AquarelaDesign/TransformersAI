import requests
from bs4 import BeautifulSoup
import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import os
import json
from datetime import datetime
import re
import time
from pathlib import Path

class DataCollector:
    def __init__(self, config_manager = None):
        self.config_manager = config_manager
        self.collected_data = []
        self.base_dir = 'collected_data'
        self.data_dir = Path(self.base_dir)
        self.ensure_directory()
    
    def ensure_directory(self):
        """Garantir que o diretório existe"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def collect_data(self, config):
        """Coleta dados baseado na configuração fornecida"""
        try:
            # Coletar dados da web
            if config.get('web_sources'):
                for url in config['web_sources']:
                    self.scrape_website(url, config.get('keywords', []))
            
            # Coletar dados de emails
            if config.get('email_config'):
                self.collect_emails(config['email_config'])
            
            # Salvar dados coletados
            self.save_collected_data()
            
        except Exception as e:
            print(f"Erro na coleta de dados: {e}")
    
    def is_relevant_text(self, text, keywords):
        """Verifica se o texto é relevante baseado nas palavras-chave"""
        if not text or len(text.strip()) < 20:  # Mínimo de 20 caracteres
            return False
            
        # Se não há palavras-chave, aceita textos com pelo menos 20 caracteres
        if not keywords:
            return len(text.strip()) >= 20
        
        text_lower = text.lower()
        # Verifica se alguma palavra-chave está presente
        keyword_found = any(keyword.lower() in text_lower for keyword in keywords)
        
        # Debug: mostrar textos que passaram pelo filtro
        if keyword_found and len(text.strip()) >= 20:
            print(f"✅ Texto relevante encontrado (tamanho: {len(text)}): {text[:100]}...")
            
        return keyword_found and len(text.strip()) >= 20

    def scrape_website(self, url, keywords):
        """Faz scraping de um website"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove elementos desnecessários
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extrair texto relevante com prioridades
            texts = []
            
            # Prioridade 1: Conteúdo principal
            main_tags = ['main', 'article', 'section[class*="content"]', 'div[class*="content"]']
            for selector in main_tags:
                elements = soup.select(selector)
                for element in elements:
                    paragraphs = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if self.is_relevant_text(text, keywords):
                            texts.append(text)
            
            # Prioridade 2: Se não encontrou conteúdo principal, busca em todos os parágrafos
            if not texts:
                print(f"⚠️ Nenhum conteúdo principal encontrado em {url}, buscando em todos os elementos...")
                for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td']:
                    elements = soup.find_all(tag)
                    for element in elements:
                        text = element.get_text().strip()
                        if self.is_relevant_text(text, keywords):
                            texts.append(text)
            
            # Adicionar dados coletados
            valid_texts = []
            for text in texts:
                if len(text.strip()) >= 20:  # Validação adicional
                    valid_texts.append({
                        'source': url,
                        'type': 'web',
                        'content': text,
                        'timestamp': datetime.now().isoformat()
                    })
            
            self.collected_data.extend(valid_texts)
            print(f"📝 Coletados {len(valid_texts)} textos válidos de {url}")
            
            # Debug: mostrar alguns exemplos
            if valid_texts:
                print(f"🔍 Exemplo de texto coletado: {valid_texts[0]['content'][:200]}...")
            
        except Exception as e:
            print(f"❌ Erro no scraping de {url}: {e}")
    
    def collect_emails(self, email_config):
        """Coleta emails com suporte a diferentes provedores"""
        try:
            # Configurações do servidor IMAP
            imap_server = email_config['imap_server']
            imap_port = email_config.get('imap_port', 993)
            username = email_config['username']
            password = email_config['password']
            use_ssl = email_config.get('use_ssl', True)
            
            print(f"Conectando ao servidor IMAP: {imap_server}:{imap_port}")
            
            # Conectar ao servidor IMAP
            if use_ssl:
                mail = imaplib.IMAP4_SSL(imap_server, imap_port)
            else:
                mail = imaplib.IMAP4(imap_server, imap_port)
                if email_config.get('use_starttls', False):
                    mail.starttls()
            
            # Fazer login
            mail.login(username, password)
            print("Login realizado com sucesso")
            
            # Selecionar caixa de entrada
            mail.select('inbox')
            
            # Buscar emails recentes (últimos 50)
            search_criteria = 'ALL'
            if email_config.get('only_unread', False):
                search_criteria = 'UNSEEN'
            
            status, messages = mail.search(None, search_criteria)
            
            if status != 'OK':
                print("Erro ao buscar emails")
                return
            
            # Pegar IDs dos emails
            email_ids = messages[0].split()
            
            # Limitar a últimos 50 emails para evitar sobrecarga
            email_ids = email_ids[-50:] if len(email_ids) > 50 else email_ids
            
            print(f"Processando {len(email_ids)} emails...")
            
            for i, msg_id in enumerate(email_ids):
                try:
                    # Buscar email
                    status, msg_data = mail.fetch(msg_id, '(RFC822)')
                    
                    if status != 'OK':
                        continue
                    
                    # Parsear email
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extrair informações
                    subject = self.decode_header_value(msg.get('Subject', ''))
                    from_addr = self.decode_header_value(msg.get('From', ''))
                    date = msg.get('Date', '')
                    
                    # Extrair conteúdo
                    content = self.extract_email_content(msg)
                    
                    if content and len(content.strip()) > 20:
                        # Combinar assunto e conteúdo
                        full_content = f"Assunto: {subject}\n\n{content}"
                        
                        self.collected_data.append({
                            'source': f"Email de {from_addr}",
                            'type': 'email',
                            'content': full_content,
                            'subject': subject,
                            'from': from_addr,
                            'date': date,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Mostrar progresso
                    if (i + 1) % 10 == 0:
                        print(f"Processados {i + 1}/{len(email_ids)} emails")
                
                except Exception as e:
                    print(f"Erro ao processar email {msg_id}: {e}")
                    continue
            
            mail.close()
            mail.logout()
            print(f"Coleta de emails concluída: {len([d for d in self.collected_data if d['type'] == 'email'])} emails coletados")
            
        except Exception as e:
            print(f"Erro na coleta de emails: {e}")
    
    def decode_header_value(self, value):
        """Decodifica headers de email que podem estar em diferentes encodings"""
        if not value:
            return ""
        
        try:
            decoded_parts = decode_header(value)
            decoded_value = ""
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_value += part.decode(encoding)
                    else:
                        decoded_value += part.decode('utf-8', errors='ignore')
                else:
                    decoded_value += str(part)
            
            return decoded_value
        except Exception as e:
            print(f"Erro ao decodificar header: {e}")
            return str(value)
    
    def extract_email_content(self, msg):
        """Extrai conteúdo de texto do email com melhor tratamento de encoding"""
        content = ""
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    
                    # Pegar apenas texto simples, não anexos
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                # Tentar diferentes encodings
                                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                                    try:
                                        content = payload.decode(encoding)
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                
                                if content:
                                    break
                        except Exception as e:
                            print(f"Erro ao extrair payload: {e}")
                            continue
            else:
                # Email não é multipart
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                content = payload.decode(encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                except Exception as e:
                    print(f"Erro ao extrair conteúdo simples: {e}")
        
        except Exception as e:
            print(f"Erro geral na extração de conteúdo: {e}")
        
        return content.strip() if content else ""
    
    def collect_email_data(self, email_config):
        """Método para compatibilidade com app.py - coleta dados de email"""
        try:
            self.collect_emails(email_config)
            
            # Extrair apenas o conteúdo dos emails coletados
            email_texts = []
            for item in self.collected_data:
                if item.get('type') == 'email':
                    email_texts.append(item['content'])
            
            print(f"📧 Extraídos {len(email_texts)} textos de email")
            return email_texts
            
        except Exception as e:
            print(f"❌ Erro na coleta de emails: {e}")
            return []

    def collect_web_data(self, url, keywords=None):
        """Coleta dados de uma URL web (corrigida para retornar lista de textos)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts e estilos
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extrai texto das tags principais com melhor seleção
            content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'div', 'li'])
            text_content = []
            
            for tag in content_tags:
                text = tag.get_text(strip=True)
                # Filtros mais flexíveis
                if text and len(text) >= 20:  # Mínimo de 20 caracteres
                    # Se não há keywords, aceita qualquer texto válido
                    if not keywords or any(keyword.lower() in text.lower() for keyword in keywords):
                        text_content.append(text)
            
            result_string = '\n'.join(text_content)
            print(f"📊 Coletado {len(text_content)} textos, total de caracteres: {len(result_string)}")
            
            # Debug: mostrar amostra do conteúdo
            if result_string:
                print(f"🔍 Amostra do conteúdo coletado: {result_string[:300]}...")
                
            # Validar se o resultado tem conteúdo útil
            valid_paragraphs = [p for p in result_string.split('\n') if len(p.strip()) >= 30]
            print(f"📋 Parágrafos válidos encontrados: {len(valid_paragraphs)}")
            
            if valid_paragraphs:
                print(f"✅ Exemplo de parágrafo válido: {valid_paragraphs[0][:100]}...")
                
            # IMPORTANTE: Processar e armazenar os dados válidos
            processed_data = self.process_collected_text(result_string, url)
            if processed_data:
                self.collected_data.extend(processed_data)
                print(f"💾 Adicionados {len(processed_data)} itens válidos à coleção")
            
            # RETORNAR LISTA DE TEXTOS VÁLIDOS (não string)
            valid_texts = [p.strip() for p in valid_paragraphs if len(p.strip()) >= 30]
            print(f"🔄 Retornando {len(valid_texts)} textos válidos para o app.py")
            
            return valid_texts  # Retorna lista, não string
            
        except requests.RequestException as e:
            print(f"❌ Erro de requisição para {url}: {e}")
            return []
        except Exception as e:
            print(f"❌ Erro ao processar {url}: {e}")
            return []

    def save_collected_data(self, texts, sources, config):
        """Salva dados coletados em formato esperado pelo model_trainer"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Criar diretório se não existir
            os.makedirs('training_data', exist_ok=True)
            
            # Preparar dados no formato esperado pelo trainer
            training_data = []
            for i, text in enumerate(texts):
                training_data.append({
                    'id': i,
                    'content': text,
                    'source': sources[0] if sources else 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'length': len(text),
                    'word_count': len(text.split())
                })
            
            # Salvar em formato compatível com model_trainer
            filename = f'training_data/collected_data_{timestamp}.json'
            data_to_save = {
                'timestamp': datetime.now().isoformat(),
                'total_texts': len(texts),
                'sources': sources,
                'config_used': config,
                'texts': texts,  # Lista de strings para o trainer
                'detailed_data': training_data  # Dados detalhados para análise
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            print(f"📁 Dados salvos em: {filename}")
            print(f"📊 Total de textos: {len(texts)}")
            print(f"📋 Fontes: {', '.join(sources)}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return None

    def get_collected_data(self):
        """Retornar dados coletados"""
        return self.collected_data
    
    def clear_collected_data(self):
        """Limpar dados coletados da memória"""
        self.collected_data = []
        print("🗑️ Dados limpos da memória")
    
    def load_collected_data(self, filepath):
        """Carregar dados de arquivo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'collected_data' in data:
                self.collected_data.extend(data['collected_data'])
                print(f"📂 {len(data['collected_data'])} itens carregados de {filepath}")
                return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados de {filepath}: {e}")
            return False
    
    def prepare_training_data(self):
        """Preparar dados coletados para treinamento"""
        if not self.collected_data:
            return None
        
        training_data = []
        
        for item in self.collected_data:
            # Dividir texto em parágrafos
            paragraphs = item['text'].split('\n')
            
            for paragraph in paragraphs:
                if len(paragraph.strip()) > 50:  # Parágrafos com mais de 50 caracteres
                    training_data.append({
                        'input': f"Baseado no conteúdo de {item['title']}:",
                        'output': paragraph.strip(),
                        'source': item['url'],
                        'collected_at': item['collected_at']
                    })
        
        return training_data
    
    def get_statistics(self):
        """Obter estatísticas dos dados coletados"""
        if not self.collected_data:
            return {
                'total_items': 0,
                'total_characters': 0,
                'total_urls': 0,
                'average_length': 0
            }
        
        total_chars = sum(len(item['text']) for item in self.collected_data)
        unique_urls = len(set(item['url'] for item in self.collected_data))
        
        return {
            'total_items': len(self.collected_data),
            'total_characters': total_chars,
            'total_urls': unique_urls,
            'average_length': total_chars // len(self.collected_data) if self.collected_data else 0
        }
    
    def get_collected_data_info(self):
        """Retorna informações sobre os dados coletados"""
        # Informações dos arquivos salvos
        files = list(self.data_dir.glob("*.txt"))
        json_files = list(Path(self.base_dir).glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in files)
        
        # Informações dos dados em memória
        memory_data_count = len(self.collected_data)
        memory_data_size = sum(len(str(item.get('content', ''))) for item in self.collected_data)
        
        return {
            "files_count": len(files),
            "json_files_count": len(json_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "memory_data_count": memory_data_count,
            "memory_data_size_kb": round(memory_data_size / 1024, 2),
            "files": [f.name for f in files],
            "json_files": [f.name for f in json_files]
        }

    def validate_collected_text(self, text):
        """Valida se um texto coletado é útil para treinamento"""
        if not text or not isinstance(text, str):
            return False, "Texto vazio ou inválido"
            
        text = text.strip()
        
        # Verificações básicas
        if len(text) < 30:
            return False, f"Texto muito curto ({len(text)} caracteres)"
            
        # Verificar se não é apenas caracteres especiais ou números
        letters = sum(1 for c in text if c.isalpha())
        if letters < len(text) * 0.5:  # Pelo menos 50% letras
            return False, "Texto com poucos caracteres alfabéticos"
            
        # Verificar se tem palavras suficientes
        words = text.split()
        if len(words) < 5:
            return False, f"Poucas palavras ({len(words)})"
            
        return True, "Texto válido"

    def process_collected_text(self, raw_text, source_url):
        """Processa texto coletado e extrai parágrafos válidos"""
        if not raw_text:
            return []
            
        # Dividir em parágrafos
        paragraphs = raw_text.split('\n')
        valid_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            is_valid, reason = self.validate_collected_text(paragraph)
            
            if is_valid:
                valid_paragraphs.append({
                    'source': source_url,
                    'type': 'web',
                    'content': paragraph,
                    'timestamp': datetime.now().isoformat(),
                    'length': len(paragraph),
                    'word_count': len(paragraph.split())
                })
            else:
                print(f"⚠️ Parágrafo rejeitado: {reason} - '{paragraph[:50]}...'")
        
        print(f"📊 Processamento completo: {len(valid_paragraphs)} parágrafos válidos de {len(paragraphs)} total")
        return valid_paragraphs
