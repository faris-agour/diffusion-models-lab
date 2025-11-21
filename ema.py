
import copy
import torch

def build_ema_model(model):

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model

@torch.no_grad()
def ema_update(ema_model, model, decay: float = 0.999):

    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        if ema_p.data.dtype.is_floating_point:
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)
