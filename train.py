from trainer import MyConfig
from transformers import AutoModelForCausalLM
import torch

def main() : 
    policy = AutoModelForCausalLM.from_pretrained(MyConfig.model_name_or_path, cache_dir = MyConfig.cache_dir, low_cpu_mem_usage=True, device_map='auto')
    if not MyConfig.sft_mode : 
        reference = AutoModelForCausalLM.from_pretrained(MyConfig.model_name_or_path, low_cpu_mem_usage=True,device_map='auto')
    else : 
        reference = None
    if MyConfig.archive is not None:
        state_dict = torch.load(MyConfig.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        policy.load_state_dict(state_dict['state'])
        if not MyConfig.sft_mode :
            reference.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')