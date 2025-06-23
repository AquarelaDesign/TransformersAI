from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
import json
import os
import warnings
from datetime import datetime

# Suprimir avisos específicos
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*loss_type.*")

class ModelTrainer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.training_status = {
            'status': 'idle',
            'progress': 0,
            'message': 'Aguardando início do treinamento'
        }
    
    def train_model(self, config):
        """Treina o modelo com os dados coletados"""
        try:
            self.update_status('preparing', 5, 'Carregando dados coletados...')
            
            # Carregar dados coletados
            data = self.load_training_data()
            
            if not data:
                raise Exception("Nenhum dado encontrado para treinamento")
            
            print(f"📊 Dados carregados: {len(data)} itens")
            
            self.update_status('preparing', 15, 'Configurando modelo e tokenizer...')
            
            # Configurar modelo e tokenizer primeiro
            model_name = config.get('base_model', 'microsoft/DialoGPT-small')
            print(f"🤖 Carregando modelo: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Adicionar token de padding se necessário
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.resize_token_embeddings(len(tokenizer))
            
            self.update_status('preparing', 25, 'Preparando dataset para treinamento...')
            
            # Preparar dataset com tokenizer configurado
            dataset = self.prepare_dataset(data, config, tokenizer)
            print(f"📝 Dataset preparado: {len(dataset)} exemplos")
            
            self.update_status('training', 30, 'Configurando parâmetros de treinamento...')
            
            # Determinar batch size baseado na memória disponível
            batch_size = config.get('batch_size', 2)  # Reduzido para 2 por padrão
            epochs = config.get('epochs', 3)
            
            # Configurar argumentos de treinamento otimizados
            training_args = TrainingArguments(
                output_dir=f'./models/trained_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,  # Compensar batch size menor
                save_steps=len(dataset) // 2,  # Salvar na metade do treinamento
                save_total_limit=2,
                prediction_loss_only=True,
                logging_steps=max(1, len(dataset) // 10),  # Log a cada 10% dos dados
                remove_unused_columns=False,
                dataloader_drop_last=True,
                dataloader_pin_memory=False,  # Evitar aviso de pin_memory
                report_to=None,  # Não enviar para wandb/tensorboard
                disable_tqdm=False,  # Manter barra de progresso
            )
            
            self.update_status('training', 35, 'Inicializando processo de treinamento...')
            
            # Preparar data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Inicializar trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Executar treinamento
            print(f"🚀 Iniciando treinamento com {epochs} épocas...")
            self.update_status('training', 40, f'Treinando modelo ({epochs} épocas)...')
            
            # Treinar modelo
            trainer.train()
            
            # Salvar modelo treinado
            self.update_status('saving', 90, 'Salvando modelo treinado...')
            print("💾 Salvando modelo...")
            
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            
            # Salvar informações do treinamento
            training_info = {
                'model_name': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'dataset_size': len(dataset),
                'data_sources': len(data),
                'training_date': datetime.now().isoformat(),
                'output_dir': training_args.output_dir
            }
            
            info_file = os.path.join(training_args.output_dir, 'training_info.json')
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, ensure_ascii=False, indent=2)
            
            success_message = f'Treinamento concluído! Modelo salvo em: {training_args.output_dir}'
            self.update_status('completed', 100, success_message)
            print(f"✅ {success_message}")
            
        except Exception as e:
            error_message = f'Erro no treinamento: {str(e)}'
            self.update_status('error', 0, error_message)
            print(f"❌ {error_message}")
    
    def load_training_data(self):
        """Carrega dados coletados para treinamento (corrigida)"""
        data = []
        
        # Primeiro, tentar carregar do diretório training_data (novo formato)
        training_data_dir = 'training_data'
        if os.path.exists(training_data_dir):
            for filename in os.listdir(training_data_dir):
                if filename.startswith('collected_data_') and filename.endswith('.json'):
                    file_path = os.path.join(training_data_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                        
                        # Verificar se tem o campo 'texts' (novo formato)
                        if 'texts' in file_data:
                            texts = file_data['texts']
                            sources = file_data.get('sources', [])
                            
                            # Converter para formato esperado pelo trainer
                            for i, text in enumerate(texts):
                                if isinstance(text, str) and len(text.strip()) > 20:
                                    data.append({
                                        'content': text.strip(),
                                        'source': sources[0] if sources else 'unknown',
                                        'id': f"{filename}_{i}",
                                        'timestamp': file_data.get('timestamp', datetime.now().isoformat())
                                    })
                            
                            print(f"📂 Carregados {len(texts)} textos de {filename}")
                        
                        # Verificar formato detalhado
                        elif 'detailed_data' in file_data:
                            for item in file_data['detailed_data']:
                                if 'content' in item and len(item['content'].strip()) > 20:
                                    data.append({
                                        'content': item['content'].strip(),
                                        'source': item.get('source', 'unknown'),
                                        'id': item.get('id', f"{filename}_{len(data)}"),
                                        'timestamp': item.get('timestamp', datetime.now().isoformat())
                                    })
                        
                    except Exception as e:
                        print(f"❌ Erro ao carregar {filename}: {e}")
                        continue
        
        # Fallback: tentar diretório 'data' (formato antigo)
        if not data and os.path.exists('data'):
            for filename in os.listdir('data'):
                if filename.endswith('.json'):
                    file_path = os.path.join('data', filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                        
                        # Formato antigo - lista de objetos
                        if isinstance(file_data, list):
                            for item in file_data:
                                if isinstance(item, dict) and 'content' in item:
                                    content = item['content']
                                    if isinstance(content, str) and len(content.strip()) > 20:
                                        data.append({
                                            'content': content.strip(),
                                            'source': item.get('source', 'unknown'),
                                            'id': item.get('id', f"{filename}_{len(data)}"),
                                            'timestamp': item.get('timestamp', datetime.now().isoformat())
                                        })
                        
                        print(f"📂 Carregados dados do formato antigo: {filename}")
                        
                    except Exception as e:
                        print(f"❌ Erro ao carregar {filename}: {e}")
                        continue
        
        print(f"📊 Total de dados carregados: {len(data)}")
        
        # Debug: mostrar amostra dos dados
        if data:
            print(f"🔍 Exemplo de dados carregados:")
            for i, item in enumerate(data[:3]):  # Mostrar 3 primeiros
                print(f"  {i+1}. Fonte: {item['source']}")
                print(f"     Conteúdo: {item['content'][:100]}...")
                print(f"     Tamanho: {len(item['content'])} caracteres")
        
        return data
    
    def prepare_dataset(self, data, config, tokenizer):
        """Prepara dataset para treinamento com formato correto"""
        # Filtrar textos muito curtos ou muito longos
        texts = []
        for item in data:
            content = item['content']
            if 30 <= len(content) <= 2000:  # Filtro melhorado
                texts.append(content)
        
        if not texts:
            raise Exception("Nenhum texto válido encontrado nos dados (muito curtos ou muito longos)")
        
        print(f"📋 Textos válidos após filtragem: {len(texts)}")
        
        # Preparar textos para treinamento de linguagem
        formatted_texts = []
        for text in texts:
            # Limpar e formatar texto
            clean_text = text.strip()
            if clean_text:
                formatted_texts.append(clean_text + tokenizer.eos_token)
        
        def tokenize_function(examples):
            # Tokenizar os textos
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                max_length=256,  # Reduzido para treinar mais rápido
                padding='max_length',
                return_tensors=None
            )
            
            # Para modelos causais, input_ids e labels são iguais
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Criar dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text'],
            desc="Tokenizando textos"
        )
        
        return tokenized_dataset
    
    def update_status(self, status, progress, message):
        """Atualiza status do treinamento"""
        self.training_status = {
            'status': status,
            'progress': progress,
            'message': message
        }
        print(f"Status: {message} ({progress}%)")
    
    def get_status(self):
        """Retorna status atual do treinamento"""
        return self.training_status
