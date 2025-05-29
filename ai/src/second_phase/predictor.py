import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from second_phase.reinforcement_learning import ActorCritic
from utils.gdrive import GDrive
import second_phase.config as configs
import utils.model_helper as helper

class Predictor:
  def __init__(self):
    self.config = configs.load_config()
    self.SFT_model_path = self.config['rl_training']['SFT_model_path']
    self.max_chunk_length = self.config['rl_training']['max_chunk_length']
    self.action_read = self.config['rl_training']['action_read']
    self.action_predict = self.config['rl_training']['action_predict']
    self.k = self.config['rl_training']['k']
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = BertForSequenceClassification.from_pretrained(self.SFT_model_path, num_labels=4)
    self.tokenizer = BertTokenizer.from_pretrained(self.SFT_model_path)
    self.actor_critic = ActorCritic(model=self.model).to(self.device)
    self.optimizer = AdamW(self.actor_critic.parameters(), lr=3e-5)
    self._load_model()

  def _load_model(self):
    if self.config['project']['use_gdrive']:
      best_checkpoint = GDrive().load_model_from_drive(file_name='best_checkpoint.tar', model_dir='actor_critic')
      self.actor_critic.load_state_dict(state_dict=best_checkpoint['model_state_dict'], strict=False)
      self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
    else:
      best_checkpoint = helper.load_checkpoint(
        model_dir='actor_critic',
        model=self.actor_critic,
        optimizer=self.optimizer,
        is_the_best=True
      )
      self.actor_critic = best_checkpoint['model']
      self.optimizer = best_checkpoint['optimizer']

  def _pad_fixed_length(self, tensors, max_length, pad_token_id):
    padded = []
    for tensor in tensors:
        tensor = tensor.view(-1)
        length = tensor.size(0)
        if length > max_length: 
            padded_tensor = tensor[:max_length]
        else:
            pad_size = max_length - length
            padding = torch.full((pad_size,), pad_token_id, dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        padded.append(padded_tensor)
    return torch.stack(padded, dim=0)

  def predict_full_text(self, sentence):
    self.model.eval()
    chunk = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')[0]
    state_input_ids = self._pad_fixed_length([chunk], self.max_chunk_length, self.tokenizer.pad_token_id).to(self.device)
    state_attention_mask = (state_input_ids != self.tokenizer.pad_token_id).long().to(self.device)
    state_inputs = {
        'input_ids': state_input_ids,
        'attention_mask': state_attention_mask
    }
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
      output = self.model(**state_inputs)
      pred_label = torch.argmax(output.logits, dim=1)
    
    torch.cuda.synchronize()
    end_time = time.time()
    prediction_time = end_time - start_time
    return pred_label.item(), prediction_time
  
  # Using batch
  def predict(self, sentence, k=None):
    self.actor_critic.eval()
    
    chunk = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')[0]
    _k = self.k if not k else k
    states = []
    for ids in range(0, len(chunk), _k):
      state = chunk[:ids+_k]
      states.append(state)

    state_input_ids = self._pad_fixed_length(states, len(chunk), self.tokenizer.pad_token_id).to(self.device)
    state_attention_mask = (state_input_ids != self.tokenizer.pad_token_id).long().to(self.device)
    state_inputs = {
        'input_ids': state_input_ids,
        'attention_mask': state_attention_mask
    }

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
      action_probs, _, pred_labels = self.actor_critic(state_inputs)
      action = torch.argmax(action_probs, dim=1)
    torch.cuda.synchronize()
    end_time = time.time()
    prediction_time = end_time - start_time

    ids = torch.nonzero(action == self.action_predict, as_tuple=False)
    if len(ids) > 0:
      idx = ids[0].item()
      readed_sentence = self.tokenizer.decode(states[idx], skip_special_tokens=True)
      pred_label = pred_labels[idx].item()
    else:
      readed_sentence = self.tokenizer.decode(states[-1], skip_special_tokens=True)
      pred_label = pred_labels[-1].item()

    return readed_sentence, pred_label, prediction_time


  # def predict(self, sentence, k=None):
  #   self.actor_critic.eval()
  #   prediction_time = 0.0
  #   chunk = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')[0]
  #   _k = self.k if not k else k
  #   for ids in range(0, len(chunk), _k):
  #     state = chunk[:ids+_k]
  #     state_input_ids = self._pad_fixed_length([state], self.max_chunk_length, self.tokenizer.pad_token_id).to(self.device)
  #     state_attention_mask = (state_input_ids != self.tokenizer.pad_token_id).long().to(self.device)
  #     state_inputs = {
  #         'input_ids': state_input_ids,
  #         'attention_mask': state_attention_mask
  #     }
  #     torch.cuda.synchronize()
  #     start_time = time.time()
  #     with torch.no_grad():
  #       action_probs, _, pred_label = self.actor_critic(state_inputs)
  #       action = torch.argmax(action_probs, dim=1)
  #     torch.cuda.synchronize()
  #     end_time = time.time()
  #     prediction_time += (end_time - start_time)
  #     if len(state) == len(chunk):
  #       return sentence, pred_label.item(), prediction_time
  #     if action.item() == self.action_predict:
  #       readed_sentence = self.tokenizer.decode(state, skip_special_tokens=True)
  #       return readed_sentence, pred_label.item(), prediction_time