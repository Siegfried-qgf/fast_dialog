import torch

def reduce_mean(number):
    results = [number for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, number)
    results = torch.Tensor(results)
    return results.mean()

def reduce_sum(number):
    results = [number for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, number)
    results = torch.Tensor(results)
    return results.sum()