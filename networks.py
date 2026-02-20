import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli, OneHotCategoricalStraightThrough, Independent
from torch.distributions.utils import probs_to_logits
from utils import make_mlp, symlog, symexp

class StochNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, stoch = 32, classes = 64, unimix = 0.01):
        super().__init__()
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        self.output_dim = stoch * classes
        self.net = make_mlp(input_dim, self.output_dim, num_hidden_layers)
    
    def forward(self, input):
        logits = self.net(input)
        probs = logits.view(-1, self.stoch, self.classes).softmax(-1)
        uniform = torch.ones_like(probs) / self.classes
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        logits = probs_to_logits(probs)
        stoch = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return stoch, logits

class ContinuousActorNetwork(nn.Module):
    """
    Network used as the actor for continuous actions.
    """
    def __init__(self, input_dim, output_dim, action_low, action_high, log_std_min = -2, log_std_max = 5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = make_mlp(input_dim, output_dim * 2, 3)
        self.register_buffer("action_scale", ((torch.tensor(action_high) - torch.tensor(action_low)) / 2.0))
        self.register_buffer("action_bias", ((torch.tensor(action_high) + torch.tensor(action_low)) / 2.0))

    def forward(self, input, training=False):
        mean, log_std = self.net(input).chunk(2, dim=-1)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min)/2*(torch.tanh(log_std) + 1) # (-1, 1) to (min, max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        sample = dist.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.action_scale + self.action_bias
        if training:
            logprobs = dist.log_prob(sample)
            logprobs -= torch.log(self.action_scale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = dist.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action
        
    def pred(self, input):
        mean, _ = self.net(input).chunk(2, dim=-1)
        action = torch.tanh(mean)
        action = action*self.action_scale + self.action_bias
        return action
        
class DiscreteActorNetwork(nn.Module):
    """
    Network used as the actor for discrete actions.
    """
    def __init__(self, input_dim, output_dim, unimix = 0.01):
        super().__init__()
        self.unimix = unimix
        self.net = make_mlp(input_dim, output_dim, 3, outscale=0.01)

    def forward(self, input, training=False):
        logits = self.net(input)
        probs = torch.softmax(logits, -1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        # logits = torch.log(probs)
        # probs = probs.clamp(1e-6, 1.0)
        dist = Categorical(probs=probs)
        action = dist.sample()
        if training:
            logprobs = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprobs, entropy
        else:
            return action
        
    def pred(self, input):
        logits = self.net(input)
        action = torch.argmax(logits, dim=-1)
        return action

class RecurrentModel(nn.Module):
    def __init__(self, deter_dim, stoch_dim, action_dim):
        super().__init__()

        self.linear = nn.Linear(deter_dim + stoch_dim + action_dim, 1024)
        self.activation = nn.RMSNorm(1024)
        self.recurrent = nn.GRUCell(1024, deter_dim)

    def forward(self, deter, stoch, action):
        flat_stoch = stoch.view((*stoch.shape[:-2], -1))
        return self.recurrent(self.activation(self.linear(torch.cat((deter, flat_stoch, action), -1))), deter)

class EncoderNetwork(nn.Module):
    def __init__(self, obs_shape, output_dim=1024):
        super().__init__()

        self.is_image = len(obs_shape) > 1

        if self.is_image:
            C, H, W = obs_shape
            assert H == W, "Square images assumed"
            assert H % 16 == 0, "H must be divisible by 16 for 4 convs"
            self.end_hw = H // 16
            self.depths = [32, 64, 128, 256]

            self.net = nn.Sequential(
                nn.Conv2d(C, self.depths[0], 4, stride=2, padding=1),
                RMSNorm2d(self.depths[0]),
                nn.SiLU(),

                nn.Conv2d(self.depths[0], self.depths[1], 4, stride=2, padding=1),
                RMSNorm2d(self.depths[1]),
                nn.SiLU(),

                nn.Conv2d(self.depths[1], self.depths[2], 4, stride=2, padding=1),
                RMSNorm2d(self.depths[2]),
                nn.SiLU(),

                nn.Conv2d(self.depths[2], self.depths[3], 4, stride=2, padding=1),
                RMSNorm2d(self.depths[3]),
                nn.SiLU(),
            )

            self.fc = nn.Linear(self.depths[-1] * self.end_hw * self.end_hw, output_dim)

        else:
            self.net = make_mlp(obs_shape[0], output_dim, 0)
            self.output_dim = output_dim

    def forward(self, input):
        if self.is_image:
            *leading_dims, C, H, W = input.shape
            B = int(torch.prod(torch.tensor(leading_dims)))
            x = input.reshape(B, C, H, W).float() / 255 - 0.5
            x = self.net(x)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            x = x.view(*leading_dims, -1)
            return x
        else:
            normalized_input = symlog(input)
            return self.net(normalized_input)

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-4, scale=True):
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.weight = nn.Parameter(torch.ones(num_channels)) if scale else 1.0

    def forward(self, x):
        # x: [B, C, H, W]
        rms = (x.pow(2).mean(dim=(2,3), keepdim=True) + self.eps).sqrt()
        x = x / rms
        return x * self.weight.view(1, -1, 1, 1)

class DecoderNetwork(nn.Module):
    def __init__(self, input_dim, obs_shape, binary_dims = None):
        super().__init__()

        self.is_image = len(obs_shape) > 1
        self.binary_dims = binary_dims

        if self.is_image:
            C, H, W = obs_shape
            assert H == W, "Decoder assumes square images"
            self.start_hw = H // 16
            self.depths = [256, 128, 64, 32]
            self.fc = nn.Linear(input_dim, self.depths[0] * self.start_hw * self.start_hw)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(self.depths[0], self.depths[1], 4, 2, 1),
                RMSNorm2d(self.depths[1]),
                nn.SiLU(),
                nn.ConvTranspose2d(self.depths[1], self.depths[2], 4, 2, 1),
                RMSNorm2d(self.depths[2]),
                nn.SiLU(),
                nn.ConvTranspose2d(self.depths[2], self.depths[3], 4, 2, 1),
                RMSNorm2d(self.depths[3]),
                nn.SiLU(),
                nn.ConvTranspose2d(self.depths[3], C, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.base = make_mlp(input_dim, 1024, 2)
            if binary_dims:
                self.cont_head = make_mlp(1024, obs_shape[0] - len(binary_dims), 1)
                self.binary_head = make_mlp(1024, len(binary_dims), 1)
            else:
                self.cont_head = make_mlp(1024, obs_shape[0], 1)

    def forward(self, input):
        if self.is_image:
            orig_shape = input.shape[:-1]
            input = input.view(-1, input.shape[-1])  # flatten all dims but last
            x = self.fc(input)
            x = x.view(x.shape[0], -1, self.start_hw, self.start_hw)
            x = self.net(x)
            x = x.view(*orig_shape, *x.shape[1:]) # restore all initial dims
            return x
        else:
            x = self.base(input)
            if self.binary_dims:
                return self.cont_head(x), self.binary_head(x)
            else:
                return self.cont_head(x)

    def loss(self, input, target):
        if self.is_image:
            pred = self.forward(input)
            target = target.float() / 255.0
            loss = (pred - target) ** 2
            loss = loss.sum(dim=(-3, -2, -1))
            return loss
        elif self.binary_dims:
            cont_pred, binary_pred = self.forward(input)
            cont_indices = [i for i in range(target.shape[-1]) if i not in self.binary_dims]
            binary_indices = self.binary_dims
            cont_target = target[..., cont_indices]
            binary_target = target[..., binary_indices]
            cont_loss = (cont_pred - symlog(cont_target)) ** 2
            binary_target = binary_target.float()
            binary_loss = F.binary_cross_entropy_with_logits(binary_pred, binary_target, reduction='none')
            return cont_loss.sum(-1) + binary_loss.sum(-1)
        else:
            pred = self.forward(input)
            loss = (pred - symlog(target))**2
            return loss.sum(-1)

class BinsNetwork(nn.Module):
    """
    Network used for critic and reward head.
    """
    def __init__(self, device, input_dim, num_hidden_layers, bins = 255):
        super().__init__()
        self.device = device
        self.net = make_mlp(input_dim, bins, num_hidden_layers, outscale=0.0)
        if bins % 2 == 1:
            half = symexp(torch.linspace(-20, 0, (bins - 1) // 2 + 1))
            self.bins = torch.cat([half, -torch.flip(half[:-1], dims=[0])], dim=0).to(self.device)
        else:	
            half = symexp(torch.linspace(-20, 0, (bins - 1) // 2))
            self.bins = torch.cat([half, -torch.flip(half, dims=[0])], dim=0).to(self.device)

    def forward(self, input):
        return self.pred(input)

    def pred(self, input):
        logits = self.net(input)
        probs = torch.nn.functional.softmax(logits, -1)
        n = logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = probs[..., :m]
            p2 = probs[..., m: m + 1]
            p3 = probs[..., m + 1:]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m: m + 1]
            b3 = self.bins[..., m + 1:]
            wavg = (p2 * b2).sum(-1) + (torch.flip(p1 * b1, dims=[-1]) + (p3 * b3)).sum(-1)
        else:
            p1 = probs[..., :n // 2]
            p2 = probs[..., n // 2:]
            b1 = self.bins[..., :n // 2]
            b2 = self.bins[..., n // 2:]
            wavg = (torch.flip(p1 * b1, dims=[-1]) + (p2 * b2)).sum(-1)
        return wavg
    
    def loss(self, input, target):
        logits = self.net(input)
        below = (self.bins <= target[..., None]).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).sum(-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = (below == above)
        dist_to_below = torch.where(equal, 1, abs(self.bins[below] - target))
        dist_to_above = torch.where(equal, 1, abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target_below = torch.nn.functional.one_hot(below, num_classes=len(self.bins))
        target_above = torch.nn.functional.one_hot(above, num_classes=len(self.bins))
        target = (target_below * weight_below[..., None] + target_above * weight_above[..., None])
        log_pred = logits - torch.logsumexp(logits, -1, keepdims=True)
        loss = -(target * log_pred).sum(-1)
        return loss
    
class BinaryNetwork(nn.Module):
    """
    Network used for continuation head.
    """
    def __init__(self, input_dim, num_hidden_layers = 1):
        super().__init__()
        self.net = make_mlp(input_dim, 1, num_hidden_layers)

    def forward(self, input):
        logit = self.net(input).squeeze(-1)
        probs = torch.sigmoid(logit)
        dist = Bernoulli(probs)
        sample = (dist.sample() > 0).int()
        return sample, logit

    def pred(self, input):
        logit = self.net(input)
        return (logit > 0).int()
    
    def loss(self, input, target):
        logit = self.net(input).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logit, target, reduction="none")

class Normalize(nn.Module):
    def __init__(self, device, impl='perc', rate=0.01, limit=1e-8, perclo=5.0, perchi=95.0, debias=True):
        super().__init__()
        self.device = device
        self.impl = impl
        self.rate = rate
        self.limit = limit
        self.perclo = perclo
        self.perchi = perchi
        self.debias = debias

        if self.debias and self.impl != 'none':
            self.register_buffer('corr', torch.zeros(()))
        
        if self.impl == 'perc':
            self.register_buffer('lo', torch.zeros((), device=self.device))
            self.register_buffer('hi', torch.zeros((), device=self.device))
        elif self.impl == 'none':
            pass
        else:
            raise NotImplementedError(self.impl)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return self.stats()

    @torch.no_grad()
    def update(self, x):
        x = x.float()
        if self.impl == 'none':
            return
        elif self.impl == 'perc':
            lo_val = torch.quantile(x, self.perclo / 100.0).to(self.device)
            hi_val = torch.quantile(x, self.perchi / 100.0).to(self.device)
            self.lo.mul_(1 - self.rate).add_(self.rate * lo_val)
            self.hi.mul_(1 - self.rate).add_(self.rate * hi_val)
        else:
            raise NotImplementedError(self.impl)

        if self.debias and self.impl != 'none':
            self.corr.mul_(1 - self.rate).add_(self.rate * 1.0)

    def stats(self):
        corr = 1.0
        if self.debias and self.impl != 'none':
            corr /= max(self.rate, self.corr.item())
        
        if self.impl == 'none':
            return 0.0, 1.0
        elif self.impl == 'perc':
            lo = self.lo * corr
            hi = self.hi * corr
            return lo.item(), max(self.limit, (hi - lo).item())
        else:
            raise NotImplementedError(self.impl)
        
if __name__ == '__main__':

    # test continuous actor (receives deter + stoch, outputs action)
    actor = ContinuousActorNetwork(input_dim = 8192 + 32 * 64, output_dim = 3, action_low = (-10, -5, -2), action_high=(2, 5, 10))
    inputs = torch.rand((16, 8192 + 32 * 64))
    actions, logprobs, entropy = actor(inputs, True)
    assert actions.shape == (16, 3)
    assert logprobs.shape == (16,)
    assert entropy.shape == (16,)

    # test discrete actor (receives deter + stoch, outputs possible actions)
    actor = DiscreteActorNetwork(input_dim = 8192 + 32 * 64, output_dim=10)
    inputs = torch.rand((16, 8192 + 32 * 64))
    actions, logprobs, entropy = actor(inputs, True)
    assert actions.shape == (16,)
    assert logprobs.shape == (16,)
    assert entropy.shape == (16,)

    # test posterior (receives deter + action, outputs stoch)
    posterior = StochNetwork(input_dim = 8192 + 10, num_hidden_layers=1) 
    inputs = torch.rand((16, 8192 + 10))
    stoch, post_logits = posterior(inputs)
    assert stoch.shape == (16, 32, 64)
    assert post_logits.shape == (16, 32, 64)

    # test prior (receives deter, outputs stoch)
    prior = StochNetwork(input_dim = 8192, num_hidden_layers=1)
    inputs = torch.rand((16, 8192))
    stoch, prior_logits = prior(inputs)
    assert stoch.shape == (16, 32, 64)
    assert prior_logits.shape == (16, 32, 64)

    # test sequence (receives deter + stoch + action, outputs deter)
    sequence = SequenceNetwork(deter_dim=8192, stoch_dim=32*64, action_dim=10)
    inputs_deter = torch.rand((16, 8192))
    inputs_stoch = torch.rand((16, 32*64))
    inputs_action = torch.rand((16, 10))
    deter = sequence(inputs_deter, inputs_stoch, inputs_action)
    assert deter.shape == (16, 8192)

    # test vector encoder (receives vector observation, outputs tokens)
    encoder = EncoderNetwork(obs_shape=(20, ), output_dim=1024)
    inputs = torch.rand((16, 20))
    tokens = encoder(inputs)
    assert tokens.shape == (16, 1024)

    # test image encoder (receives image observation, outputs tokens)
    encoder = EncoderNetwork(obs_shape=(3, 64, 64), output_dim=1024)
    inputs = torch.rand((16, 64, 3, 64, 64))
    tokens = encoder(inputs)
    assert tokens.shape == (16, 64, 1024)

    # test vector decoder (receives deter + stoch, outputs vector observation)
    decoder = DecoderNetwork(input_dim=8192 + 32 * 64, obs_shape=(20, ))
    inputs = torch.rand((16, 8192 + 32 * 64))
    targets = torch.rand((16, 20)) * 20
    predictions = decoder(inputs)
    loss = decoder.loss(inputs, targets)
    assert predictions.shape == (16, 20)
    assert loss.shape == (16,)

    # test image decoder (receives deter + stoch, outputs image observation)
    decoder = DecoderNetwork(input_dim=8192 + 32 * 64, obs_shape=(3, 64, 64))
    inputs = torch.rand((16, 8192 + 32 * 64))
    targets = torch.rand((16, 3, 64, 64)) * 255
    predictions = decoder(inputs)
    loss = decoder.loss(inputs, targets)
    assert predictions.shape == (16, 3, 64, 64)
    assert loss.shape == (16,)

    # test bins network (receives deter + stoch, outputs scalar)
    bins = BinsNetwork(torch.device('cpu'), input_dim=8192 + 32 * 64, num_hidden_layers=3)
    inputs = torch.rand((16, 8192 + 32 * 64))
    scalars = bins(inputs)
    targets = (torch.rand((16)) - 0.5) * 20
    loss = bins.loss(inputs, targets)
    assert scalars.shape == (16,)
    assert loss.shape == (16,)

    # test binary network (receives deter + stoch, outputs bool)
    binary = BinaryNetwork(input_dim=8192 + 32 * 64, num_hidden_layers=3)
    inputs = torch.rand((16, 8192 + 32 * 64))
    predictions, logits = binary(inputs)
    targets = ((torch.rand((16)) - 0.5) > 0).float()
    loss = binary.loss(inputs, targets)
    assert logits.shape == (16,)
    assert loss.shape == (16,)
