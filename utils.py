import torch
import torch.nn as nn

def make_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim=1024, outscale=1.0):
    layers = []
    for i in range(num_hidden_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.RMSNorm(hidden_dim))
        layers.append(nn.SiLU())
    last_input = input_dim if num_hidden_layers == 0 else hidden_dim
    last_layer = nn.Linear(last_input, output_dim)
    last_layer.weight.data *= outscale
    last_layer.bias.data *= outscale
    layers.append(last_layer)
    return nn.Sequential(*layers)

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

def log_sigmoid(x):
    return torch.log(torch.sigmoid(x))

def unimix_dist(logits, unimix = 0.01):
    probs = torch.softmax(logits, -1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    probs = (1 - unimix) * probs + unimix * uniform
    logits = torch.log(probs)
    dist = torch.distributions.Categorical(logits)
    dist.logits = logits
    return dist, probs

def lambda_return(last, term, rew, boot, lam=0.95, disc=1.0):
    rets = [boot[:, -1]]  # start from last time step
    live = (1 - term.float())[:, 1:] * disc     # [batch, T-1]
    cont = (1 - last.float())[:, 1:] * lam      # [batch, T-1]
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]  # [batch, T-1]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return torch.stack(list(reversed(rets))[:-1], dim=1)  # [batch, T]

def adaptive_gradient_clipping(parameters, clip_factor=0.3, eps=1e-3):
    with torch.no_grad():
        for p in parameters:
            if p.grad is None:
                continue

            grad_norm = torch.norm(p.grad)
            param_norm = torch.norm(p)

            max_norm = clip_factor * torch.clamp(param_norm, min=eps)

            if grad_norm > max_norm:
                p.grad.mul_(max_norm / (grad_norm + 1e-6))

def expand_mask(mask, target):
    # mask: [B]
    # target: [B, ...]
    return mask.view(mask.shape[0], *([1] * (target.ndim - 1)))