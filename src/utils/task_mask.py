import torch

def apply_task_mask(task_outputs, task_mask):
    masked_outputs = {}

    for task, output in task_outputs.items():
        mask = task_mask[task].unsqueeze(-1)
        masked_outputs[task] = output * mask

    return masked_outputs
