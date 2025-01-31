import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import EarlyStoppingCallback
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# config 설정에 tokenizer 모듈이 사용되므로 미리 tokenizer를 정의해줍니다.
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")

config_data = {
    "general": {
        "data_path": "/home/fine_data/",
        "model_name": "digit82/kobart-summarization",
        "output_dir": "./model_output"  # 모델이 저장될 디렉토리
    },
    "tokenizer": {
        "encoder_max_len": 256,
        "decoder_max_len": 100,
        "bos_token": f"{tokenizer.bos_token}",
        "eos_token": f"{tokenizer.eos_token}",
        "special_tokens": ['<usr>', '</s>', '#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "prompt": {
        "template": "배경 정보나 맥락을 제공하면서 인물의 행동과 결과를 중심으로 포괄적인 상황을 요약해주세요: {dialogue}",
        "max_length": 256
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": 'cosine',
        "optim": 'adamw_torch',
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0, 
        "evaluation_strategy": 'steps',
        "save_strategy": 'steps',  # 'epoch'마다 모델 저장
        "save_total_limit": 3,
        "fp16": True,
        "bf16": False,
        "load_best_model_at_end": True,
        "seed": 42,
        "logging_dir": "./logs",
        "logging_strategy": "steps",
        "predict_with_generate": True,
        "generation_max_length": 100,
        "do_train": True,
        "do_eval": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    },
    "inference": {
        "ckt_path": "./model_output/checkpoint-epoch",  # 실제 모델 경로로 설정
        "result_path": "./prediction/",
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "generate_max_length": 100,
        "num_beams": 4,
        "batch_size": 32,
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

# 데이터 전처리를 위한 클래스
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
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

    # 프롬프트 템플릿 적용
    encoder_input_train = [
        config['prompt']['template'].format(dialogue=dialogue)
        for dialogue in encoder_input_train
    ]
    # 프롬프트 템플릿 적용 - 검증 데이터
    encoder_input_val = [
        config['prompt']['template'].format(dialogue=dialogue)
        for dialogue in encoder_input_val
    ]
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
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge = Rouge()
    scores = []
    
    for pred, label in zip(pred_str, label_str):
        try:
            score = rouge.get_scores(pred, label)[0]
            scores.append(score)
        except ValueError:
            scores.append({'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}})

    rouge1 = np.mean([score['rouge-1']['f'] for score in scores]) * 100
    rouge2 = np.mean([score['rouge-2']['f'] for score in scores]) * 100
    rougel = np.mean([score['rouge-l']['f'] for score in scores]) * 100
    
    # ROUGE 점수의 평균 계산
    rouge_avg = (rouge1 + rouge2 + rougel) / 3

    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougel': rougel,
        'rouge_avg': rouge_avg  # 평균 ROUGE 점수 추가
    }

# 가중치 검증을 위한 유틸리티 클래스 추가
class WeightValidator:
    @staticmethod
    def check_weights(model):
        """모델의 가중치 상태를 검사합니다."""
        issues_found = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.data is None:
                    print(f"가중치 누락: {name}")
                    issues_found = True
                elif torch.isnan(param.data).any():
                    print(f"NaN이 포함된 가중치 발견: {name}")
                    issues_found = True
                elif torch.isinf(param.data).any():
                    print(f"무한대가 포함된 가중치 발견: {name}")
                    issues_found = True
        return not issues_found

    @staticmethod
    def summarize_weights(model):
        """모델 가중치의 통계 정보를 출력합니다."""
        print("\n=== 가중치 통계 정보 ===")
        for name, param in model.named_parameters():
            if param.requires_grad:
                mean = param.data.mean().item()
                std = param.data.std().item()
                print(f"{name} - Mean: {mean:.6f}, Std: {std:.6f}")

    @staticmethod
    def initialize_weights(model):
        """필요한 경우 가중치를 초기화합니다."""
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        print("모델 가중치를 초기화했습니다.")

class SummaryPPOTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rouge = Rouge()
        self.validator = WeightValidator()
        
        # PPO 하이퍼파라미터
        self.ppo_config = {
            'eps_clip': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'ppo_epochs': 4,
            'mini_batch_size': 8,
            'gamma': 0.99,
            'lambda_': 0.95
        }
        
        # 옵티마이저 설정
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 멀티 GPU 설정
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # Rouge 점수 목표 설정
        self.target_rouge_score = 50.0
        
        # TensorBoard 설정
        self.writer = SummaryWriter(log_dir=config['training']['logging_dir'])
        
    def check_model_state(self):
        """모델의 현재 상태를 검사합니다."""
        return self.validator.check_weights(self.model)

    def log_metrics(self, epoch, metrics):
        """메트릭을 TensorBoard에 기록합니다."""
        self.writer.add_scalar('Loss/Policy', metrics['policy_loss'], epoch)
        self.writer.add_scalar('Loss/Entropy', metrics['entropy'], epoch)
        self.writer.add_scalar('Reward/Mean', metrics['mean_reward'], epoch)
        self.writer.add_scalar('Rouge/R1', metrics['rouge1'], epoch)
        self.writer.add_scalar('Rouge/R2', metrics['rouge2'], epoch)
        self.writer.add_scalar('Rouge/RL', metrics['rougel'], epoch)

    def compute_rewards(self, generated_summaries, reference_summaries):
        rewards = []
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0
        
        for gen, ref in zip(generated_summaries, reference_summaries):
            if not gen.strip():
                # 빈 요약문 처리: 예를 들어, Rouge 점수를 0으로 설정
                rewards.append(0.0)
                continue
            try:
                scores = self.rouge.get_scores(gen, ref, avg=True)
            except ValueError as e:
                # 예상치 못한 에러 처리
                print(f"Error computing ROUGE scores: {e}")
                rewards.append(0.0)
                continue

            total_rouge1 += scores['rouge-1']['f'] * 100
            total_rouge2 += scores['rouge-2']['f'] * 100
            total_rougel += scores['rouge-l']['f'] * 100
            
            # 평균 Rouge 점수를 보상으로 사용
            avg_rouge = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3 * 100
            rewards.append(avg_rouge)
        
        batch_size = len(generated_summaries)
        avg_rouge1 = total_rouge1 / batch_size
        avg_rouge2 = total_rouge2 / batch_size
        avg_rougel = total_rougel / batch_size
        
        return (torch.tensor(rewards, device=self.model.device), 
                avg_rouge1, avg_rouge2, avg_rougel)

    def check_rouge_threshold(self, rouge1, rouge2, rougel):
        avg_rouge = (rouge1 + rouge2 + rougel) / 3
        return avg_rouge >= self.target_rouge_score

    def generate_with_log_probs(self, input_ids, attention_mask):
        """생성 및 로그 확률 계산을 수행합니다."""
        # 입력 텐서를 float 타입으로 변환
        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = input_embeds.detach().requires_grad_(True)
        attention_mask = attention_mask.float().detach().requires_grad_(True)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs_embeds=input_embeds,  # input_ids 대신 임베딩 사용
                attention_mask=attention_mask,
                max_length=100,
                min_length=10,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                use_cache=False,  # gradient checkpointing과 호환되도록 False로 설정
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 모든 시퀀스를 동일한 길이로 패딩
        max_length = max(generated_ids.size(1), input_ids.size(1))
        
        # 입력 시퀀스 패딩
        if input_ids.size(1) < max_length:
            padding = torch.full(
                (input_ids.size(0), max_length - input_ids.size(1)),
                self.tokenizer.pad_token_id,
                device=input_ids.device
            )
            input_ids = torch.cat([input_ids, padding], dim=1)
            attention_padding = torch.zeros(
                (attention_mask.size(0), max_length - attention_mask.size(1)),
                device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, attention_padding], dim=1)
        
        # 생성된 시퀀스 패딩
        if generated_ids.size(1) < max_length:
            padding = torch.full(
                (generated_ids.size(0), max_length - generated_ids.size(1)),
                self.tokenizer.pad_token_id,
                device=generated_ids.device
            )
            generated_ids = torch.cat([generated_ids, padding], dim=1)
        
        # 로그 확률 계산
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=generated_ids
        )
        
        logits = outputs.logits
        
        # 패딩 마스크 생성
        padding_mask = (generated_ids != self.tokenizer.pad_token_id).float()
        
        # 로그 확률 계산
        log_probs = F.log_softmax(logits, dim=-1)
        
        batch_size = generated_ids.size(0)
        seq_length = generated_ids.size(1)
        
        # 새로운 로그 확률 텐서 초기화
        new_log_probs = torch.zeros((batch_size, seq_length), device=generated_ids.device)
        old_log_probs = torch.zeros((batch_size, seq_length), device=generated_ids.device)
        
        # 각 위치에서의 로그 확률 계산
        for i in range(batch_size):
            for j in range(seq_length):
                if padding_mask[i, j] == 1:  # 패딩이 아닌 경우만 처리
                    token_id = generated_ids[i, j]
                    if j < log_probs.size(1):
                        new_log_probs[i, j] = log_probs[i, j, token_id]
                        old_log_probs[i, j] = log_probs[i, j, token_id].clone()
        
        # NaN 및 Inf 값 처리
        new_log_probs = torch.nan_to_num(new_log_probs, 0.0)
        old_log_probs = torch.nan_to_num(old_log_probs, 0.0)
        
        # requires_grad 설정
        new_log_probs = new_log_probs.requires_grad_(True)
        old_log_probs = old_log_probs.detach()
        
        return generated_ids, new_log_probs, old_log_probs

    def ppo_train_step(self, dataloader):
        """PPO 학습 단계를 수행합니다."""
        self.model.train()
        # gradient checkpoint 활성화
        self.model.gradient_checkpointing_enable()
        
        total_policy_loss = 0
        total_entropy = 0
        total_rewards = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc="PPO Training"):
            try:
                # 입력 텐서를 device로 이동하고 그래디언트 계산 활성화
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device).float().detach().requires_grad_(True)
                reference_summaries = batch['labels'].to(self.model.device)

                # 임베딩 레이어의 파라미터에 requires_grad=True 설정
                for param in self.model.get_input_embeddings().parameters():
                    param.requires_grad_(True)

                # 텍스트 디코딩
                reference_summaries = self.tokenizer.batch_decode(reference_summaries, skip_special_tokens=True)

                # PPO 업데이트
                for _ in range(self.ppo_config['ppo_epochs']):
                    # 그래디언트 초기화
                    self.optimizer.zero_grad()
                    
                    # 이전 정책의 출력 계산
                    with torch.no_grad():
                        old_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False  # gradient checkpointing을 위해 필요
                        )
                        old_summaries = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=100,
                            min_length=10,
                            num_beams=4,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            use_cache=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        old_summaries_text = self.tokenizer.batch_decode(old_summaries, skip_special_tokens=True)
                        old_log_probs = torch.nn.functional.log_softmax(old_outputs.logits, dim=-1)
                        
                        old_rewards, rouge1, rouge2, rougel = self.compute_rewards(
                            old_summaries_text, 
                            reference_summaries
                        )

                    if self.check_rouge_threshold(rouge1, rouge2, rougel):
                        return {
                            'policy_loss': 0,
                            'entropy': 0,
                            'mean_reward': old_rewards.mean().item(),
                            'rouge1': rouge1,
                            'rouge2': rouge2,
                            'rougel': rougel,
                            'target_achieved': True
                        }

                    # 새로운 정책의 출력 계산
                    new_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False  # gradient checkpointing을 위해 필요
                    )
                    new_summaries = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=100,
                        min_length=10,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        use_cache=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_summaries_text = self.tokenizer.batch_decode(new_summaries, skip_special_tokens=True)
                    new_log_probs = torch.nn.functional.log_softmax(new_outputs.logits, dim=-1)
                    
                    new_rewards, new_rouge1, new_rouge2, new_rougel = self.compute_rewards(
                        new_summaries_text, 
                        reference_summaries
                    )

                    # 어드밴티지 계산
                    advantages = (new_rewards - old_rewards).to(self.model.device)
                    
                    # 텐서 크기 맞추기
                    advantages = advantages.view(-1, 1, 1).expand(-1, new_log_probs.size(1), -1)

                    # 정책 비율 계산
                    ratios = torch.exp(new_log_probs - old_log_probs)

                    # PPO 클립된 목적 함수
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(
                        ratios,
                        1 - self.ppo_config['eps_clip'],
                        1 + self.ppo_config['eps_clip']
                    ) * advantages

                    # 정책 손실
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 엔트로피 보너스
                    probs = torch.exp(new_log_probs)
                    entropy = -(probs * new_log_probs).mean()

                    # 총 손실
                    loss = policy_loss - self.ppo_config['entropy_coef'] * entropy

                    # 역전파
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_entropy += entropy.item()
                    total_rewards += new_rewards.mean().item()
                    total_samples += 1

            except Exception as e:
                print(f"배치 처리 중 오류 발생: {str(e)}")
                continue

        # 평균 계산
        avg_policy_loss = total_policy_loss / total_samples if total_samples > 0 else 0
        avg_entropy = total_entropy / total_samples if total_samples > 0 else 0
        avg_reward = total_rewards / total_samples if total_samples > 0 else 0

        return {
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy,
            'mean_reward': avg_reward,
            'rouge1': new_rouge1,
            'rouge2': new_rouge2,
            'rougel': new_rougel,
            'target_achieved': False
        }

    def evaluate_test_performance(self, test_dataloader):
        self.model.eval()
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                if self.n_gpu > 1:
                    input_ids = torch.nn.parallel.scatter(batch['input_ids'], range(self.n_gpu))
                    attention_mask = torch.nn.parallel.scatter(batch['attention_mask'], range(self.n_gpu))
                else:
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)

                # 요약문 생성
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config['training']['generation_max_length'],
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                
                # 생성된 요약문 디코딩
                generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                reference_summaries = batch['labels']
                
                # Rouge 점수 계산
                for gen, ref in zip(generated_summaries, reference_summaries):
                    if not gen.strip() or not ref.strip():
                        # 빈 문자열이 있을 경우 기본 점수 할당
                        scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
                    else:
                        try:
                            scores = self.rouge.get_scores(gen, ref, avg=True)
                        except ValueError as e:
                            print(f"Error computing ROUGE scores: {e}")
                            # 오류 발생 시 기본 점수 할당
                            scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
                    total_rouge1 += scores['rouge-1']['f'] * 100
                    total_rouge2 += scores['rouge-2']['f'] * 100
                    total_rougel += scores['rouge-l']['f'] * 100
                    total_samples += 1
        
        # 평균 Rouge 점수 계산
        avg_rouge1 = total_rouge1 / total_samples
        avg_rouge2 = total_rouge2 / total_samples
        avg_rougel = total_rougel / total_samples
        
        self.model.train()
        return avg_rouge1, avg_rouge2, avg_rougel

    def close(self):
        """TensorBoard writer를 정리합니다."""
        self.writer.close()

def load_trainer_for_train(config, generate_model, tokenizer, train_dataset, val_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        resume_from_checkpoint=True,  # 체크포인트에서 이어서 학습
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
        max_grad_norm=config['training']['gradient_clipping'],
        eval_strategy=config['training']['evaluation_strategy'],
        eval_steps=500,
        save_strategy=config['training']['save_strategy'],
        save_steps=500,
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model="rouge_avg", # rouge1 점수 기준으로 best 모델 선정
        greater_is_better=True,      # 높은 점수가 더 좋음
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=100,
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval']
    )

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    # 테스트 데이터 로더는 콜백 내부에서 준비
    test_data, test_dataset = prepare_test_dataset(config, preprocessor, tokenizer)

    ppo_trainer = SummaryPPOTrainer(generate_model, tokenizer, config)

    class PPOCallbackCustom(TrainerCallback):
        def __init__(self, ppo_trainer, test_dataset):
            self.ppo_trainer = ppo_trainer
            self.test_dataset = test_dataset
            self.target_achieved = False
            self.no_improvement_count = 0

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.target_achieved:
                return

            # 테스트 데이터 로더 생성
            test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=config['inference']['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # PPO 학습 단계
            metrics = self.ppo_trainer.ppo_train_step(kwargs['train_dataloader'])

            # 테스트 데이터셋에서의 성능 평가
            rouge1, rouge2, rougel = self.ppo_trainer.evaluate_test_performance(test_dataloader)

            # 로그 기록
            with open("ppo_training_log.txt", "a") as f:
                f.write(f"Epoch {state.epoch}:\n")
                f.write(f"Test Rouge-1: {rouge1:.2f}\n")
                f.write(f"Test Rouge-2: {rouge2:.2f}\n")
                f.write(f"Test Rouge-L: {rougel:.2f}\n")
                f.write(f"Mean Reward: {metrics['mean_reward']:.4f}\n")
                f.write("-" * 50 + "\n")

            # 목표 Rouge 점수 달성 확인
            if self.ppo_trainer.check_rouge_threshold(rouge1, rouge2, rougel):
                self.target_achieved = True
                control.should_training_stop = True
                print(f"목표 Rouge 점수({self.ppo_trainer.target_rouge_score})를 달성했습니다!")
                print(f"최종 Test Rouge 점수 - R1: {rouge1:.2f}, R2: {rouge2:.2f}, RL: {rougel:.2f}")

            # 일정 기간 동안 성능 향상이 없으면 조기 종료
            if self.no_improvement_count >= config['training']['early_stopping_patience']:
                control.should_training_stop = True
                print(f"성능 향상이 없어 학습을 종료합니다.")

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[PPOCallbackCustom(ppo_trainer, test_dataset)]
    )
    
    return trainer

# 학습을 위한 tokenizer와 모델 불러오기
def load_tokenizer_and_model_for_train(config, device):
    """토크나이저와 모델을 로드하고 가중치 상태를 검증합니다."""
    print("=== 모델 및 토크나이저 로드 시작 ===")
    
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    
    # Gradient Checkpointing 활성화
    generate_model.gradient_checkpointing_enable()

    # 가중치 상태 초기 검사
    validator = WeightValidator()
    print("\n=== 초기 가중치 상태 검사 ===")
    initial_check = validator.check_weights(generate_model)
    if not initial_check:
        print("WARNING: 초기 가중치에 문제가 발견되었습니다.")
    
    # 토크나이저 설정
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")

    # 임베딩 크기 조정
    generate_model.resize_token_embeddings(len(tokenizer))
    print("토크나이저와 모델의 임베딩 크기를 재조정했습니다.")

    # 가중치 통계 정보 출력
    validator.summarize_weights(generate_model)

    # GPU로 모델 이동
    generate_model.to(device)
    print(f"모델을 {device}로 이동했습니다.")

    # 최종 가중치 상태 검사
    print("\n=== 최종 가중치 상태 검사 ===")
    final_check = validator.check_weights(generate_model)
    if not final_check:
        print("ERROR: 최종 가중치에 문제가 발견되었습니다.")
        raise ValueError("모델 가중치에 문제가 있습니다. 학습을 시작할 수 없습니다.")

    return generate_model, tokenizer

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-'*10, 'Load data complete', '-'*10)

    # 프롬프트 템플릿 적용
    encoder_input_test = [
        config['prompt']['template'].format(dialogue=dialogue)
        for dialogue in encoder_input_test
    ]

    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False,
    )
    test_tokenized_decoder_inputs = tokenizer(
        decoder_input_test, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False,
    )

    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs, 
        test_id, 
        len(encoder_input_test)
    )
    print('-'*10, 'Make dataset complete', '-'*10)

    return test_data, test_encoder_inputs_dataset

# 추론을 위한 tokenizer와 학습시킨 모델을 불러옵니다.
def load_tokenizer_and_model_for_test(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10)

    return generate_model, tokenizer

def find_best_checkpoint(output_dir):
    """
    output_dir에서 'best_model_rouge' 접두사를 가진 모델들 중
    가장 높은 Rouge 점수를 가진 모델의 경로를 반환합니다.
    """
    model_paths = glob(os.path.join(output_dir, "best_model_rouge_*"))
    if not model_paths:
        raise ValueError("No checkpoint found with 'best_model_rouge' prefix")
    
    # Rouge 점수 추출 및 정렬
    scores = []
    for path in model_paths:
        score = float(path.split('_')[-1])  # 파일명에서 Rouge 점수 추출
        scores.append((score, path))
    
    # 가장 높은 점수의 모델 경로 반환
    best_score, best_path = max(scores)
    print(f"Selected best checkpoint with Rouge score: {best_score:.2f}")
    return best_path

# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config):
    # 기존 ckt_path 대신 best checkpoint 사용
    best_checkpoint = find_best_checkpoint(config['general']['output_dir'])
    config['inference']['ckt_path'] = best_checkpoint
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-'*10, f'device : {device}', '-'*10)
    print(torch.__version__)

    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=True)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary": preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output_dl.csv"), index=False)

    return output

# 메인 함수
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    except ValueError as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    
    # Dataset을 반환하도록 수정
    train_dataset, val_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_dataset, val_dataset)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
    finally:
        if hasattr(trainer, 'ppo_trainer'):
            trainer.ppo_trainer.close()

if __name__ == "__main__":
    main(loaded_config)
    output = inference(loaded_config)
    print(output)