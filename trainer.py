from transformers import AutoTokenizer
from dataclasses import dataclass
import torch

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

class Trainer():
    def __init__(self, policy, config, run_dir, reference_model) :
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = MyConfig.tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, MyConfig.cache_dir)
        if self.tokenizer.pad_token_id is None : 
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        

        self.policy = policy
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
    
    def get_batch_metrics(self, batch, train=True) : 
        metrics = {}
        train_test = 'train' if train else 'test'

        if not MyConfig.sft_mode : 
