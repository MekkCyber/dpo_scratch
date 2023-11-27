import datasets
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence

def extract_prompt(prompt_and_response) : 
    '''
        will extract only the prompt, which is the example without the last response from the assistant
    '''
    search_term = "\n\nAssistant:"
    search_term_idx = search_term.rfind(search_term)
    assert search_term_idx != -1, f"example doesnt contain : {search_term}"
    return prompt_and_response[:search_term_idx+len(search_term)]
    
def split_prompt_and_response(example) : 
    prompt = extract_prompt(example['chosen'])
    chosen_response = example['chosen'][len(prompt):]
    # this works because exactly the same prompt is used for both responses
    rejected_response = example['rejected'][len(prompt):]
    return prompt, chosen_response, rejected_response

def get_hh(cache_dir, silent=False) : 
    '''
        example from the dataset : 
            {'chosen': 
                "\n\nHuman: What are some cuss words in english?
                \n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, 
                    shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, 
                    blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, 
                    sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: I haven't even thought about it.", 
            'rejected': 
                "\n\nHuman: What are some cuss words in english?
                \n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, 
                    shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, 
                    blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, 
                    sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: Ass."}
    '''
    dataset = datasets.load_dataset("Anthropic/hh-rlhf", cache_dir=cache_dir)
    dataset = dataset['train']
    '''
        we initialize data as an empty dict, with values as dicts too, the format is : 
        data = {
            '<prompt1>' : {
                'pairs' : [(0,1), (2,3),...]
                'responses' : [chosen1, rejected1, chosen2, rejected2,...]
                'sft_target' : last_chosen
            }
            '<prompt2>' : {
                'pairs' : [(0,1), (2,3),...]
                'responses' : [chosen1, rejected1, chosen2, rejected2,...]
                'sft_target' : last_chosen
            }
            ...
        }
    '''
    
    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm(dataset, desc="Processing Dataset", disable=silent) :
        prompt, chosen, rejected = split_prompt_and_response(row)
        responses = [chosen, rejected]
        n_responses  = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses+1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
    
    return data

def tokenize_element(prompt, chosen, rejected, truncation_mode, tokenizer, max_length, max_prompt_length) : 
    chosen_tokens = tokenizer(chosen)
    rejected_tokens = tokenizer(rejected)
    prompt_tokens = tokenizer(prompt)
    '''
        after the tokenization chosen_tokens, rejected_tokens, and prompt_tokens will have the form
        {
            'input_ids':[...],
            'attention_mask':[...],
            'token_type_ids':[...]
        }
    '''
    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)
    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
    # Here we will truncate all the fields in prompt_tokens (attention_mask, input_ids, and token_type_ids)
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length : 
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')
    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}
    '''
        chosen_sequence_tokens will have the same form as prompt_tokens and chosen_tokens
        {
            'input_ids':[...],
            'attention_mask':[...],
            'token_type_ids':[...]
        }
        input_ids in this case are the concatenation of the prompt_tokens input_ids and chosen_tokens input_ids
    '''
    chosen_sequence_tokens = {k: prompt_tokens[k]+chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k]+rejected_tokens[k] for k in rejected_tokens}
    '''
        labels are just the inputs_ids, but we replace the input_ids that correspond to the prompt by -100
        -100 means that they will be ignored
    '''
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100]*len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100]*len(prompt_tokens['input_ids'])

    result = {}

    result['prompt'] = prompt
    result['chosen'] = prompt+chosen
    result['rejected'] = prompt+rejected
    result['chosen_response_only'] = chosen
    result['rejected_response_only'] = rejected

    for k, v in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items() :
        for type_key, tokens in v.items() :
            if type_key != "token_type_ids" : 
                continue
            result[f'{k}_{type_key}'] = tokens
    '''
        result will have the form : 
        {
            # strings
            'prompt': ...
            'chosen': ...
            'rejected': ...
            'chosen_response_only': ...
            'rejected_response_only': ...
            # tokens
            'chosen_input_ids': [...]
            'chosen_attention_mask: [...]
            'chosen_labels': [...]
            'rejected_input_ids': [...]
            'rejected_attention_mask: [...]
            'rejected_labels': [...]
            'prompt_input_ids': [...]
            'prompt_attention_mask: [...]
        }
    '''
    return result


def get_collate_fn(tokenizer) : 
    '''
        batch is a list of dictionnaries of the form : 
        {
            # strings
            'prompt': ...
            'chosen': ...
            'rejected': ...
            'chosen_response_only': ...
            'rejected_response_only': ...
            # tokens
            'chosen_input_ids': [...]
            'chosen_attention_mask: [...]
            'chosen_labels': [...]
            'rejected_input_ids': [...]
            'rejected_attention_mask: [...]
            'rejected_labels': [...]
            'prompt_input_ids': [...]
            'prompt_attention_mask: [...]
        }
    '''
    def collate_fn(batch) :
        padded = {}
        for k in batch[0].keys() : 
            # we only collate tokens
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels') :
                # prompt tokens
                if 'prompt' in k :
                    # we pad the prompt from the start and not the end
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else : 
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids') : 
                    padding_value = tokenizer.pad_token_id
                elif k.endswidth('_labels') :
                    padding_value = -100
                elif k.endswidth('_attention_mask') : 
                    padding_value = 0
                else : 
                    raise ValueError(f"unexpected key in batch {k}")
                
                padded[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded[k] = padded[k].flip(dim=[1])
            else : 
                padded[k] = [ex[k] for ex in batch]
        '''
            padded is a dict of the form : 
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
        return padded
    return collate_fn



def get_batch_iterator(tokenizer, batch_size=1, max_length=512, max_prompt_length=128,sft_mode=False, n_epochs=None, silent=False, cache_dir=None) :

    '''
        The hh data format if you don't recall is :
        data = {
            '<prompt1>' : {
                'pairs' : [(0,1), (2,3),...]
                'responses' : [chosen1, rejected1, chosen2, rejected2,...]
                'sft_target' : last_chosen
            }
            '<prompt2>' : {
                'pairs' : [(0,1), (2,3),...]
                'responses' : [chosen1, rejected1, chosen2, rejected2,...]
                'sft_target' : last_chosen
            }
            ...
        }
    '''
    list_data = []
    for prompt, data in get_hh(cache_dir, silent).items() :
        list_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], 'keep_end'))
    
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    while True : 
        if epoch_idx >= n_epochs : 
            if not silent : 
                print(f'Finished the {n_epochs} epochs')
            break
        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in list_data : 
            if sft_mode : 
                # we pass sft_target even for rejected because we are going to delete it after anyway
                element = tokenize_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                element = {k: v for k, v in element.items() if 'rejected' not in k}
                '''
                    element will have the form : 
                    {
                        # strings
                        'prompt': ...
                        'chosen': ...
                        'chosen_response_only': ...
                        # tokens
                        'chosen_input_ids': [...]
                        'chosen_attention_mask: [...]
                        'chosen_labels': [...]
                        'prompt_input_ids': [...]
                        'prompt_attention_mask: [...]
                    }
                '''
                batch.append(element)
                if len(batch) == batch_size :
                    yield collate_fn(batch)
                    batch = []

            else : 
                for p in pairs :
                    element = tokenize_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(element)
                    if len(batch) == batch_size : 
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
                        yield collate_fn(batch)
        epoch_idx += 1
                
