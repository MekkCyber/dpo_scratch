from transformers import AutoTokenizer
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from collections import defaultdict

from dataset import get_batch_iterator
from utils import pad_to_length

@dataclass
class MyConfig: 
    max_length = 512
    max_prompt_length = 128
    sft_mode = False
    tokenizer_name_or_path = None
    cache_dir = None
    n_epochs = 1
    batch_size=1
    eval_batch_size=1
    beta = 0.1
    optimizer = "Adam"
    lr = 0.001
    warmup_steps = 2
    eval_every = 100
    model_name_or_path = None
    archive = None

def get_batch_logps(logits, labels, average_log_prob=False) : 
    ''' Compute the log probabilities of the given labels under the given logits.
        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    '''
    labels = labels[:,1:].clone() # we ignore the first token so that the labels are one token ahead of logits, because we try to predict the next word
    logits = logits[:,:-1,:]
    loss_mask = (labels != -100)

    labels[labels == -100] = 0
    # this will get the probabilities of each token in the label
    # shape : (batch_size, seq_length)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob : 
        return (per_token_logps*loss_mask).sum(-1) / loss_mask.sum(-1)
    else : 
        return (per_token_logps*loss_mask).sum(-1)

def concatenate_inputs(batch) : 
    ''' Concatenate the chosen and rejected inputs into a single tensor.
            batch form : 
            {
                # list of strings
                'prompt': [...]
                'chosen': [...]
                'rejected': [...]
                'chosen_response_only': [...]
                'rejected_response_only': [...]
                # list of lists of tokens
                'chosen_input_ids': [[...]]
                'chosen_attention_mask: [[...]]
                'chosen_labels': [[...]]
                'rejected_input_ids': [[...]]
                'rejected_attention_mask: [[...]]
                'rejected_labels': [[...]]
                'prompt_input_ids': [[...]]
                'prompt_attention_mask: [[...]]
            }
        '''
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concat_batch = {}
    for k in batch : 
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor) : 
            pad_value = -100 if 'labels' in k else 0
            concat_key = k.replace('chosen','concatenated')
            concat_batch[concat_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch : 
        if k.startswith('rejected') and isinstance(batch[k]) : # batch[k] coming from the collate_fn is normally a tensor because pad_sequence returns a Tensor
            pad_value = -100 if 'labels' in k else 0
            concat_key = k.replace('rejected','concatenated')
            concat_batch[concat_key] = torch.cat([concat_batch[concat_key], pad_to_length(batch[k], max_length, pad_value=pad_value)], dim=0)
    '''
        concat_batch has the form : 
            {
            'concatenated_input_ids': [[...]]
            'concatenated_attention_mask: [[...]]
            'concatenated_labels': [[...]]
            }
    '''
    return concat_batch

def get_preference(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta) :
    policy_ratios = policy_chosen_logps-policy_rejected_logps
    ref_ratios = ref_chosen_logps-ref_rejected_logps

    logits = policy_ratios - ref_ratios
    losses = -F.logsigmoid(beta*logits)

    chosen_rewards = beta*(policy_chosen_logps-ref_chosen_logps).detach()
    rejected_rewards = beta*(policy_rejected_logps-ref_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

class Trainer():
    def __init__(self, policy, config, run_dir, reference_model) :
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = MyConfig.tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, MyConfig.cache_dir)
        if self.tokenizer.pad_token_id is None : 
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # policy is just the causal LLM we are trying to finetune
        self.policy = policy
        # reference_model is in the start the same LLM, but it doesnt change, its not trained
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(self.tokenizer, split='train', max_length=MyConfig.max_length, max_prompt_length=MyConfig.max_prompt_length, sft_mode=MyConfig.sft_mode, n_epochs=MyConfig.n_epochs, batch_size=MyConfig.batch_size, cache_dir=MyConfig.cache_dir)
        self.eval_iterator = get_batch_iterator(self.tokenizer, split='test', max_length=MyConfig.max_length, max_prompt_length=MyConfig.max_prompt_length, sft_mode=MyConfig.sft_mode, n_epochs=MyConfig.n_epochs, batch_size=MyConfig.eval_batch_size, cache_dir=MyConfig.cache_dir)


    def get_batch_samples(self, batch) : 
        '''
            batch form : 
            {
                # list of strings
                'prompt': [...]
                'chosen': [...]
                'rejected': [...]
                'chosen_response_only': [...]
                'rejected_response_only': [...]
                # list of lists of tokens
                'chosen_input_ids': [[...]]
                'chosen_attention_mask: [[...]]
                'chosen_labels': [[...]]
                'rejected_input_ids': [[...]]
                'rejected_attention_mask: [[...]]
                'rejected_labels': [[...]]
                'prompt_input_ids': [[...]]
                'prompt_attention_mask: [[...]]
            }
        '''
        policy_out = self.policy.generate(batch['prompt_input_ids'], max_length=MyConfig.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        if not MyConfig.sft_mode : 
            reference_out = self.reference_model.generate(batch['prompt_input_ids'], max_length=MyConfig.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        policy_out = pad_to_length(policy_out, MyConfig.max_length, self.tokenizer.pad_token_id)
        policy_out_decoded = self.tokenizer.batch_decode(policy_out, skip_special_tokens=True)
        if not MyConfig.sft_mode : 
            reference_out = pad_to_length(reference_out, MyConfig.max_length, self.tokenizer.pad_token_id)
            reference_out_decoded = self.tokenizer.batch_decode(reference_out, skip_special_tokens=True)
        else : reference_out_decoded = []

        return policy_out_decoded, reference_out_decoded
    
    def concatenate_forward(self, model, batch) :
        concat_batch = concatenate_inputs(batch)
        '''
            concat_batch has the form : 
            {
                'concatenated_input_ids': [[...]]
                'concatenated_attention_mask: [[...]]
                'concatenated_labels': [[...]]
            }
        '''
        all_logits = model(concat_batch['concatenated_input_ids'],  attention_mask=concat_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = get_batch_logps(all_logits, batch['concatenated_labels'])
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps
    
    def get_batch_metrics(self, batch, train=True) : 
        metrics = {}
        train_test = 'train' if train else 'test'

        if not MyConfig.sft_mode : 
            # policy_chosen_logps, policy_rejected_logps are both of size (batch_size,)
            # policy_chosen_logps : is the log probability of the chosen label given the policy
            # policy_rejected_logps : is the log probability of the rejected label given the policy
            policy_chosen_logps, policy_rejected_logps = self.concatenate_forward(self.policy, batch)
            with torch.no_grad() :
                reference_chosen_logps, reference_rejected_logps = self.concatenate_forward(self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = get_preference(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, MyConfig.beta)
            reward_accuracies = (chosen_rewards > rejected_rewards)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        else : 
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, batch['chosen_labels'])

            losses = -policy_chosen_logits
        
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        return losses.mean(), metrics

    
    def train(self) :
        self.optimizer = getattr(torch.optim, MyConfig.optimizer)(self.policy.parameters(), lr=MyConfig.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step+1)/(MyConfig.warmup_steps+1)))
        all_eval_metrics = {}
        if not MyConfig.sft_mode : 
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0

        for batch in self.train_iterator : 
            if self.example_counter % MyConfig.eval_every == 0 and (self.example_counter > 0):
                self.policy.eval()
            
                for eval_batch in self.eval_iterator : 
                    with torch.no_grad() :
                        _, eval_metrics = self.get_batch_metrics(eval_batch, False)
                    for k, v in eval_metrics.items() : 
                        all_eval_metrics[k].extend(v)

                mean_eval_metrics = {k: sum(v)/len(v) for k, v in all_eval_metrics.items()}

            self.policy.train()
            batch_metrics = defaultdict(list)
            loss, metrics = self.get_batch_metrics(batch, train=True)
            loss.backward()

            for k, v in metrics.items() : 
                batch_metrics[k].extend(v)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.batch_counter += 1
            self.example_counter += MyConfig.batch_size

        