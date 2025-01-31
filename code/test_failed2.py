import pandas as pd
import os
import re
import json
import yaml
from datetime import datetime
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import BartConfig, AutoTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from typing import List, Tuple
import numpy as np
from nltk.tokenize import sent_tokenize
from googletrans import Translator
import kss  # Korean Sentence Splitter

# config 설정에 tokenizer 모듈이 사용되므로 미리 tokenizer를 정의해줍니다.
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")

config_data = {
    "general": {
        "data_path": "/home/data/",
        "model_name": "digit82/kobart-summarization",
        "output_dir": "./model_output"  # 모델이 저장될 디렉토리
    },
    "tokenizer": {
        "encoder_max_len": 512,
        "decoder_max_len": 100,
        "bos_token": f"{tokenizer.bos_token}",
        "eos_token": f"{tokenizer.eos_token}",
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": 'cosine',
        "optim": 'adamw_torch',
        "gradient_accumulation_steps": 2,
        "evaluation_strategy": 'epoch',
        "save_strategy": 'epoch',  # 'epoch'마다 모델 저장
        "save_total_limit": 5,
        "fp16": True,
        "load_best_model_at_end": True,
        "seed": 42,
        "logging_dir": "./logs",
        "logging_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 100,
        "do_train": True,
        "do_eval": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    },
    "inference": {

        "ckt_path": "./model_output/",  # 실제 모델 경로로 설정
        "result_path": "./prediction/",
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "generate_max_length": 100,
        "num_beams": 4,
        "batch_size": 8,
        "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
    }
}

# 모델의 구성 정보를 YAML 파일로 저장합니다.
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)

# 저장된 config 파일을 불러옵니다.
with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

class DialoguePreprocessor:
    def __init__(self, window_size: int = 10, stride: int = 5):
        self.window_size = window_size
        self.stride = stride
        self.noise_patterns = [
            r'ㅋ+',
            r'ㅎ+',
            r'[ㄱ-ㅎㅏ-ㅣ]+',
            r'[\u2600-\u26FF\u2700-\u27BF]+',
            r'[!?]{2,}',
        ]
        self.translator = Translator()

    def sliding_window(self, dialogue: str) -> List[str]:
        try:
            sentences = kss.split_sentences(dialogue)
            windows = []
            for i in range(0, len(sentences), self.stride):
                window = sentences[i:i + self.window_size]
                windows.append(" ".join(window))
            return windows
        except:
            return [dialogue]

    def remove_noise(self, text: str) -> str:
        try:
            cleaned = text
            for pattern in self.noise_patterns:
                cleaned = re.sub(pattern, '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip()
        except:
            return text

    def normalize_repeated_utterances(self, text: str) -> str:
        try:
            sentences = kss.split_sentences(text)
            unique_sentences = []
            for sent in sentences:
                if sent not in unique_sentences:
                    unique_sentences.append(sent)
            return " ".join(unique_sentences)
        except:
            return text

    def back_translate(self, text: str, middle_lang='en') -> str:
        try:
            middle = self.translator.translate(text, dest=middle_lang).text
            result = self.translator.translate(middle, dest='ko').text
            return result
        except:
            return text

    def replace_synonyms(self, text: str) -> str:
        try:
            synonyms = {
                '말했다': ['이야기했다', '얘기했다', '설명했다'],
                '물었다': ['질문했다', '문의했다', '요청했다'],
            }
            result = text
            for word, replacements in synonyms.items():
                if word in result:
                    result = result.replace(word, np.random.choice(replacements))
            return result
        except:
            return text

    def process_dialogue(self, dialogue: str, summary: str = None) -> Tuple[str, str]:
        try:
            # 1. 노이즈 제거
            cleaned_dialogue = self.remove_noise(dialogue)
            cleaned_dialogue = self.normalize_repeated_utterances(cleaned_dialogue)
            
            # 2. 긴 대화 처리
            if len(cleaned_dialogue.split()) > 200:
                windows = self.sliding_window(cleaned_dialogue)
                cleaned_dialogue = " ".join(windows)
            
            # 3. 데이터 증강
            if summary is not None and np.random.random() < 0.3:
                aug_type = np.random.choice(['back_translate', 'synonym'])
                if aug_type == 'back_translate':
                    cleaned_dialogue = self.back_translate(cleaned_dialogue)
                    summary = self.back_translate(summary)
                else:
                    cleaned_dialogue = self.replace_synonyms(cleaned_dialogue)
                    summary = self.replace_synonyms(summary)
            
            return cleaned_dialogue, summary if summary else None
        except Exception as e:
            print(f"Error processing dialogue: {e}")
            return dialogue, summary

class DataAugmenter:
    def __init__(self):
        self.translator = Translator()
        
    def back_translate(self, text: str, middle_lang='en') -> str:
        """역번역을 통한 데이터 증강"""
        try:
            # 한국어 -> 중간 언어
            middle = self.translator.translate(text, dest=middle_lang).text
            # 중간 언어 -> 한국어
            result = self.translator.translate(middle, dest='ko').text
            return result
        except:
            return text  # 번역 실패시 원본 반환

    def replace_synonyms(self, text: str) -> str:
        """유사어 교체 (간단한 예시)"""
        # 실제 구현시에는 Word2Vec이나 형태소 분석기 기반 유사어 사전 활용 필요
        synonyms = {
            '말했다': ['이야기했다', '얘기했다', '설명했다'],
            '물었다': ['질문했다', '문의했다', '요청했다'],
            # ... 더 많은 유사어 쌍 추가 가능
        }
        
        result = text
        for word, replacements in synonyms.items():
            if word in result:
                result = result.replace(word, np.random.choice(replacements))
        return result

def process_dialogue(dialogue: str, summary: str = None) -> Tuple[str, str]:
    """전체 처리 과정"""
    preprocessor = DialoguePreprocessor()
    augmenter = DataAugmenter()
    
    # 1. 노이즈 제거
    cleaned_dialogue = preprocessor.remove_noise(dialogue)
    cleaned_dialogue = preprocessor.normalize_repeated_utterances(cleaned_dialogue)
    
    # 2. 긴 대화 처리
    if len(cleaned_dialogue.split()) > 200:  # 긴 대화 기준
        windows = preprocessor.sliding_window(cleaned_dialogue)
        # windows를 활용한 계층적 요약은 모델 추론 시점에서 처리
        cleaned_dialogue = " ".join(windows)
    
    # 3. 데이터 증강 (학습 데이터의 경우)
    if summary is not None:
        if np.random.random() < 0.3:  # 30% 확률로 증강
            aug_type = np.random.choice(['back_translate', 'synonym'])
            if aug_type == 'back_translate':
                cleaned_dialogue = augmenter.back_translate(cleaned_dialogue)
                summary = augmenter.back_translate(summary)
            else:
                cleaned_dialogue = augmenter.replace_synonyms(cleaned_dialogue)
                summary = augmenter.replace_synonyms(summary)
    
    return cleaned_dialogue, summary if summary else None

# 데이터 전처리를 위한 클래스
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.dialogue_processor = DialoguePreprocessor()  # 추가

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue'].apply(
                lambda x: self.dialogue_processor.process_dialogue(x)[0]
            )
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue'].apply(
                lambda x: self.dialogue_processor.process_dialogue(x)[0]
            )
            decoder_input = dataset['summary'].apply(
                lambda x: self.bos_token + str(x)
            )
            decoder_output = dataset['summary'].apply(
                lambda x: str(x) + self.eos_token
            )
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

# Dataset 클래스 정의
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.len

class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.len

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len

# 데이터셋 준비 함수
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                                         add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                                         add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_outputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                                          add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_outputs, len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                                             add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                                             add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_outputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                                              add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_outputs, len(encoder_input_val))

    return train_inputs_dataset, val_inputs_dataset

# 모델 성능 평가 함수
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    # predictions이 tuple인 경우 첫 번째 요소 사용
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # 로짓(logits)을 토큰 ID로 변환
    predictions = np.argmax(predictions, axis=-1)
    
    # numpy 배열로 변환
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)
    
    # -100을 pad_token_id로 변경
    labels[labels == -100] = tokenizer.pad_token_id

    # 리스트 형식으로 변환
    decoded_preds = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # 스페셜 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        decoded_preds = [sentence.replace(token, " ") for sentence in decoded_preds]
        decoded_labels = [sentence.replace(token, " ") for sentence in decoded_labels]

    # ROUGE 점수 계산
    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {
        'eval_rouge-1': results['rouge-1']['f'],
        'eval_rouge-2': results['rouge-2']['f'],
        'eval_rouge-l': results['rouge-l']['f']
    }
    
    return result

def format_checkpoint_name(metrics, step):
    """체크포인트 이름을 포맷팅하는 함수"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rouge1 = f"{metrics['eval_rouge-1']:.4f}"
    rouge2 = f"{metrics['eval_rouge-2']:.4f}"
    rougeL = f"{metrics['eval_rouge-l']:.4f}"
    loss = f"{metrics['eval_loss']:.4f}"
    
    return f"checkpoint-{step}-r1_{rouge1}-{timestamp}"

class CustomTrainer(Seq2SeqTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        """체크포인트 저장 로직 커스터마이징"""
        # 체크포인트 이름에 메트릭 추가
        if metrics:
            checkpoint_folder = (
                f"checkpoint-step_{self.state.global_step}-"
                f"r1_{metrics.get('eval_rouge-1', 0):.4f}-"
                f"r2_{metrics.get('eval_rouge-2', 0):.4f}-"
                f"rL_{metrics.get('eval_rouge-l', 0):.4f}-"
                f"loss_{metrics.get('eval_loss', 0):.4f}"
            )
        else:
            checkpoint_folder = f"checkpoint-step_{self.state.global_step}"
        
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.save_model(output_dir)
        
        if self.args.should_save:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            
            if self.do_save_full_model:
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            if self.do_save_optimizer:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if self.do_save_scheduler:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        return output_dir
    
# Trainer 클래스와 매개변수 정의
def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval']
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = CustomTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[MyCallback]
    )

    return trainer

class EnhancedTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model_name = config['general']['model_name']
        self.tokenizer = None
        self.model = None

    # 학습을 위한 tokenizer와 모델 불러오기
    def load_tokenizer_and_model(self):
        # Load base config
        bart_config = BartConfig().from_pretrained(self.model_name, num_labels=2)
        bart_config.use_cache = False
        bart_config.gradient_checkpointing = True  # Gradient Checkpointing 활성화
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name, 
            config=bart_config
        )
        
        # Add special tokens
        special_tokens = {
            'additional_special_tokens': self.config['tokenizer']['special_tokens'] + ['<summarize>', '</summarize>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
    def get_training_args(self):
        return TrainingArguments(
            output_dir=self.config['general']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            learning_rate=self.config['training']['learning_rate'],
            fp16=self.config['training']['fp16'],
            logging_strategy=self.config['training']['logging_strategy'],
            save_strategy=self.config['training']['save_strategy'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
        )
        
    def prepare_trainer(self, train_dataset, val_dataset):
        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda pred: compute_metrics(self.config, self.tokenizer, pred),
        )
        
        return trainer

def prepare_input_data(text, tokenizer):
    # Add prompt tokens to input
    prompted_text = f"<summarize> {text} </summarize>"
    return tokenizer(prompted_text, truncation=True, padding='max_length')

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config,preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset

def evaluate_checkpoints(config, device):
    """모든 체크포인트를 평가하고 가장 좋은 성능의 체크포인트를 찾습니다."""
    results = []
    
    # 모델 체크포인트 디렉토리 찾기
    model_dir = config['general']['output_dir']
    checkpoint_dirs = sorted(glob(os.path.join(model_dir, "checkpoint-*")))
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    for ckt_path in checkpoint_dirs:
        print(f"\nEvaluating checkpoint: {ckt_path}")
        trainer = load_trainer_for_train(config, None, None, None, None)  # 필요에 따라 인자 조정
        trainer.model = BartForConditionalGeneration.from_pretrained(ckt_path)
        trainer.model.to(device)
        
        eval_result = trainer.evaluate()
        eval_result['checkpoint'] = ckt_path
        results.append(eval_result)
    
    # 가장 좋은 성능의 체크포인트 찾기 (eval_rouge-1 기준)
    best_checkpoint = max(results, key=lambda x: x.get('eval_rouge-1', 0))
    print(f"Best checkpoint: {best_checkpoint['checkpoint']} with ROUGE-1: {best_checkpoint['eval_rouge-1']}")
    
    return best_checkpoint['checkpoint']

# 추론을 위한 tokenizer와 학습시킨 모델을 불러옵니다.
def load_tokenizer_and_model_for_test(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))  # 특수 토큰 반영
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model, tokenizer

# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output

# 메인 함수
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer = EnhancedTrainer(config, device)
    trainer.load_tokenizer_and_model()

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    trainer = trainer.prepare_trainer(train_inputs_dataset, val_inputs_dataset)
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    trainer.train()
    
    output = inference(config)
    output

if __name__ == "__main__":
    main(loaded_config)
    
    