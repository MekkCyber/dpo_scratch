import torch


def pad_to_length(tensor, length, pad_value, dim) : 
    if tensor.size(dim) >= length : 
        return tensor
    else : 
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor,pad_value*torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],dim=dim)