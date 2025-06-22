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

class DataCollector:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.collected_data = []
        self.base_dir = 'collected_data'
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
    
    def scrape_website(self, url, keywords):
        """Faz scraping de um website"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extrair texto relevante
            texts = []
            for tag in ['p', 'h1', 'h2', 'h3', 'article']:
                elements = soup.find_all(tag)
                for element in elements:
                    text = element.get_text().strip()
                    if self.is_relevant_text(text, keywords):
                        texts.append(text)
            
            self.collected_data.extend([{
                'source': url,
                'type': 'web',
                'content': text,
                'timestamp': datetime.now().isoformat()
            } for text in texts])
            
        except Exception as e:
            print(f"Erro no scraping de {url}: {e}")
    
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
    
    def is_relevant_text(self, text, keywords):
        """Verifica se o texto é relevante baseado nas palavras-chave"""
        if not keywords:
            return len(text) > 50  # Mínimo de caracteres
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def save_collected_data(self):
        """Salvar dados coletados em arquivo"""
        filename = f"collected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
            
            print(f"📁 Dados salvos em: {filepath}")
            return filepath
            
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
