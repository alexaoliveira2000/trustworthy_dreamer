import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough
from tensorboardX import SummaryWriter
import gymnasium as gym
from gymnasium import spaces

from networks import ContinuousActorNetwork, DiscreteActorNetwork, BinsNetwork, BinaryNetwork, Normalize, EncoderNetwork, DecoderNetwork, StochNetwork, RecurrentModel
from memory import Memory
from utils import log_sigmoid, lambda_return, adaptive_gradient_clipping
from optimizer import LaProp

class Dreamer:

    def __init__(self,
                 env: gym.Env,
                 seed: int = None,
                 binary_dims = None,
                 emb_dim = 1024,
                 deter_dim = 1024,
                 stoch = 32,
                 classes = 64,
                 batch_size = 16, 
                 batch_length = 64,
                 horizon = 333,
                 imagination_length = 15,
                 max_episodes=10,
                 tau = 0.02,
                ):
        
        if seed:
            torch.manual_seed(seed)

        self._print_env_specs(env)

        # environment attributes
        self.is_cont_policy = isinstance(env.action_space, spaces.Box)
        self.act_dim = len(env.action_space.low) if self.is_cont_policy else env.action_space.n
        self.total_episodes = 0
        self.gradient_steps = 0
        self.best_score = -1e7

        # algorithm attributes
        stoch_dim = stoch * classes
        input_dim = deter_dim + stoch_dim
        self.emb_dim = emb_dim
        self.stoch = stoch
        self.classes = classes
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.horizon = horizon
        self.imagination_length = imagination_length
        self.tau = tau

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.memory = Memory(sequence_length=self.batch_length, max_episodes=max_episodes)
        if self.is_cont_policy:
            self.pol = ContinuousActorNetwork(input_dim, self.act_dim, env.action_space.low, env.action_space.high).to(self.device)
        else:
            self.pol = DiscreteActorNetwork(input_dim, self.act_dim).to(self.device)
        self.val = BinsNetwork(self.device, input_dim, 3).to(self.device)
        self.slowval = BinsNetwork(self.device, input_dim, 3).to(self.device)
        self.slowval.load_state_dict(self.val.state_dict())
        self.rew = BinsNetwork(self.device, input_dim, 1).to(self.device)
        self.con = BinaryNetwork(input_dim).to(self.device)
        self.sequence = RecurrentModel(deter_dim, stoch_dim, self.act_dim).to(self.device)
        self.enc = EncoderNetwork(env.observation_space.shape, emb_dim).to(self.device)
        self.dec = DecoderNetwork(input_dim, env.observation_space.shape, binary_dims).to(self.device)
        self.posterior = StochNetwork(deter_dim + emb_dim, 1, stoch, classes).to(self.device)
        self.prior = StochNetwork(deter_dim, 2, stoch, classes).to(self.device)
        self.retnorm = Normalize(self.device, impl = 'perc', rate = 0.01, limit = 1.0, perclo = 5.0, perchi = 95.0, debias = False)
        self.valnorm = Normalize(self.device, impl = 'none', rate = 0.01, limit = 1e-8)
        self.advnorm = Normalize(self.device, impl = 'none', rate = 0.01, limit = 1e-8)

        components = [self.rew, self.con, self.sequence, self.enc, self.posterior, self.prior, self.dec, self.pol, self.val]
        self.optimizer = LaProp(params=[p for v in components for p in v.parameters()])

    def _print_env_specs(self, env):
        print("========== ENVIRONMENT SPEC ==========")
        # Observation space
        print("Observation space:")
        if len(env.observation_space.shape) > 1:
            print(f"  Type: Image {env.observation_space.shape}")
        else:
            print(f"  Type: Vector ({env.observation_space.shape[0]})")
            print(f"  Low:  {env.observation_space.low}")
            print(f"  High: {env.observation_space.high}")
        # Action space
        print("Action space:")
        if isinstance(env.action_space, spaces.Box):
            print(f"  Type: Continuous ({env.action_space.shape[0]})")
            print(f"  Low:  {env.action_space.low}")
            print(f"  High: {env.action_space.high}")
        else:
            print(f"  Type: Discrete ({env.action_space.n})")
        print("======================================")

    @torch.no_grad()
    def play(self, env, seed=None, file_name = 'dreamer_play.gif', deterministic = False):
        obs, _ = env.reset(seed=seed)

        deter = torch.zeros(1, self.deter_dim, device=self.device)
        stoch = torch.zeros(1, self.stoch, self.classes, device=self.device)
        action = torch.zeros(1, self.act_dim, device=self.device) if self.is_cont_policy else torch.zeros(1, device=self.device)

        frames = []

        score = 0

        while True:
            # render environment
            frame = env.render()
            frames.append(frame)

            obs_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)

            # world model update (posterior)
            deter = self.sequence(deter, stoch, action if self.is_cont_policy else torch.nn.functional.one_hot(action.long(), self.act_dim))

            tokens = self.enc(obs_tensor)
            stoch, _ = self.posterior(torch.cat([deter, tokens], dim=-1))
            stoch_flat = stoch.view(stoch.shape[0], -1)

            # policy
            if deterministic:
                action = self.pol.pred(torch.cat([deter, stoch_flat], dim=-1))
            else:
                action = self.pol(torch.cat([deter, stoch_flat], dim=-1))

            # step env
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            score += float(reward)

            if terminated or truncated:
                break

        env.close()

        print(score)

        import imageio
        imageio.mimsave(file_name, frames, fps=30)

    def run(self,
            env, 
            seed = None, 
            warmup_episodes = 5,
            gradient_steps = 60_000,
            replay_ratio = 32,
            save_every_updates = 2000,
            run_id = 'dreamer',
        ):
        """
        Train the agent on a given environment.
        The environment should have the following functions (with inputs/outputs in numpy):
        - observation, info = env.reset(seed)
        - observation, reward, terminated, truncated, info = env.step(action)
        """
        log_dir = f'runs/{run_id}'
        save_dir = f'models/{run_id}'
        writer = SummaryWriter(log_dir=log_dir)
        updates_to_save = save_every_updates
        # run some episodes before start training
        self._env_interaction(env, seed, warmup_episodes, writer)
        while self.gradient_steps < gradient_steps:
            # train
            for _ in range(replay_ratio):
                losses, metrics = self.train()
                self.gradient_steps += 1
                updates_to_save -= 1
                # log losses and metrics
                if writer is not None:
                    losses['loss'] = sum([loss for loss in losses.values()])
                    writer.add_scalars("losses", {component: loss for component, loss in losses.items()}, self.gradient_steps)
                    writer.add_scalars("metrics", {component: metric for component, metric in metrics.items()}, self.gradient_steps)
                if updates_to_save <= 0:
                    updates_to_save = save_every_updates
                    self.save(save_dir)
            # run more episodes
            score = self._env_interaction(env, seed, 1, writer)
            if score > self.best_score:
                self.best_score = score
                self.save(f'{save_dir}_best')
                print(f'Best model saved with an average score of {score}')
        if writer is not None:
            writer.close()

    @torch.no_grad()
    def _env_interaction(self, env, seed = None, episodes = 1, writer = None, eval = False):
        total_score = 0
        for _ in range(episodes):
            deter = torch.zeros(1, self.deter_dim, device=self.device)
            stoch = torch.zeros(1, self.stoch, self.classes, device=self.device)
            action   = torch.zeros(1, self.act_dim, device=self.device) if self.is_cont_policy else torch.zeros(1, device=self.device)
            observation, _ = env.reset(seed=seed)
            score = 0
            is_first = True
            experiences = {
                "observations": [],
                "actions":      [],
                "rewards":      [],
                "is_firsts":    [],
                "is_terminals": [],
                "is_lasts":     [],
                "deters":       [],
                "stochs":       [],
            }
            # start episode
            while True:
                deter = self.sequence(deter, stoch, action if self.is_cont_policy else F.one_hot(action.long(), self.act_dim))
                observation = torch.tensor(observation).unsqueeze(0).to(self.device)
                tokens = self.enc(observation)
                stoch, _ = self.posterior(torch.cat([deter, tokens], dim=-1))
                stoch_flat = stoch.view(stoch.shape[0], -1)
                action = self.pol(torch.cat([deter, stoch_flat], dim=-1))
                next_observation, reward, is_terminal, is_truncated, _ = env.step(action.squeeze(0).cpu().numpy())
                is_last = is_terminal or is_truncated
                score += float(reward)
                # save experience
                if not eval:
                    experiences['observations'].append(observation.squeeze(0))
                    experiences['actions'].append(action.squeeze(0))
                    experiences['rewards'].append(torch.tensor(reward))
                    experiences['is_firsts'].append(torch.tensor(is_first))
                    experiences['is_terminals'].append(torch.tensor(is_terminal))
                    experiences['is_lasts'].append(torch.tensor(is_last))
                    experiences['deters'].append(deter.squeeze(0))
                    experiences['stochs'].append(stoch.squeeze(0))
                # end episode
                if is_last:
                    break
                is_first = False
                observation = next_observation
            if not eval:
                # episode stats
                print(f'Episode {self.total_episodes}: {score}')
                if writer is not None:
                    writer.add_scalars("score", {'score': score}, self.total_episodes)
                # print(len(experiences['observations']))
                # add experience to memory (only if there are enough experiences to create trajectories for training)
                if len(experiences['observations']) >= self.batch_length:
                    experiences = {k: torch.stack(v) for k, v in experiences.items()}
                    self.memory.add_episode(experiences)
                self.total_episodes += 1
            total_score += score
        return total_score /episodes

    def train(self):
        """
        Train the agent based on batches of trajectories stored in memory.
        """
        # sample B trajectories of length T
        batches, indices = self.memory.sample(num_sequences=self.batch_size, device=self.device)

        # compute the loss for every component
        losses, metrics = self._losses(batches, indices)
        loss = sum([loss for loss in losses.values()])
        self.optimizer.zero_grad()
        loss.backward()
        adaptive_gradient_clipping(self.optimizer.param_groups[0]['params'])
        self.optimizer.step()

        losses = {component: loss.item() for component, loss in losses.items()}
        metrics = {component: metric.item() for component, metric in metrics.items()}

        # update target value (slowval)
        with torch.no_grad():
            for p, p_targ in zip(self.val.parameters(), self.slowval.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        return losses, metrics

    def _losses(self, batches, indices):

        B = self.batch_size
        T = self.batch_length - 1
        D = self.deter_dim
        S = self.stoch
        C = self.classes

        # B, T, D = batches['deters'].shape
        # B, T, S, C = batches['stochs'].shape

        # encode observations (images or vectors) into embeddings
        tokens = self.enc(batches['observations'])  # [B, emb_dim]
        # get first state of each trajectory
        prev_deter = batches['deters'][:, 0]        # [B, deter_dim]
        prev_stoch = batches['stochs'][:, 0]        # [B, stoch, classes]

        # compute trajectory starting from first state
        deters = torch.empty(B, T, D, device=self.device)               # [B, T, deter_dim]
        stochs = torch.empty(B, T, S, C, device=self.device)            # [B, T, stoch, classes]
        post_logits = torch.empty(B, T, S, C, device=self.device)       # [B, T, stoch, classes]
        prior_logits = torch.empty(B, T, S, C, device=self.device)      # [B, T, stoch, classes]
        for t in range(T):
            actions = batches['actions'][:, t]
            deter = self.sequence(prev_deter, prev_stoch, actions if self.is_cont_policy else F.one_hot(actions.long(), self.act_dim))
            stoch, post_logit = self.posterior(torch.cat([deter, tokens[:, t + 1]], -1))
            _, prior_logit = self.prior(prev_deter)
            deters[:, t] = deter
            stochs[:, t] = stoch
            post_logits[:, t] = post_logit
            prior_logits[:, t] = prior_logit
            prev_deter = deter
            prev_stoch = stoch

        # update state of each used experience (except the first experience of each batch)
        self.memory.update_latents(indices, deters.detach(), stochs.detach())

        # dynamics and representation loss
        prior_dist = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
        prior_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1)
        post_dist = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
        post_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=post_logits.detach()), 1)
        dyn_kl = kl_divergence(post_dist_sg, prior_dist)
        rep_kl = kl_divergence(post_dist, prior_dist_sg)
        dyn_loss = torch.maximum(dyn_kl, torch.ones_like(dyn_kl))
        rep_loss = torch.maximum(rep_kl, torch.ones_like(rep_kl))
        dyn_ent = prior_dist.entropy().mean()
        rep_ent = post_dist.entropy().mean()

        inputs = torch.cat([deters, stochs.view(B, T, S * C)], dim=-1)  # [B, T, deter_dim + stoch_dim]

        # decoder loss
        targets = batches['observations'][:, 1:]                # [B, T, *obs_shape]
        dec_loss = self.dec.loss(inputs, targets)               # [B, T]

        # reward loss
        targets = batches['rewards'][:, 1:]                     # [B, T]
        rew_loss = self.rew.loss(inputs, targets)               # [B, T]

        # continuation loss
        targets = (1 - batches['is_terminals'][:, 1:].float())  # [B, T]
        targets *= 1 - 1 / self.horizon                         # [B, T]
        con_loss = self.con.loss(inputs, targets)               # [B, T]

        assert all(loss.shape == (B, T) for loss in [dyn_loss, rep_loss, rew_loss, con_loss, dec_loss])

        H = self.imagination_length
        rep_deter = deters.detach()
        rep_stoch = stochs.detach()

        # flatten all states (all of them representing initial states)
        initial_deters = rep_deter.view(B * T, D)
        initial_stochs = rep_stoch.view(B * T, S, C)

        # imagine trajectory from initial states (H steps into the future)
        deters = torch.empty(B * T, H, D, device=self.device)
        stochs = torch.empty(B * T, H, S, C, device=self.device)
        logprobs = torch.empty(B * T, H, device=self.device)
        entropies = torch.empty(B * T, H, device=self.device)
        prev_deter = initial_deters
        prev_stoch = initial_stochs
        for t in range(H):
            full_state = torch.cat([prev_deter, prev_stoch.view(B * T, S * C)], dim=-1)
            action, logprob, entropy = self.pol(full_state.detach(), True)
            with torch.no_grad():
                prev_deter = self.sequence(prev_deter, prev_stoch, action if self.is_cont_policy else F.one_hot(action.long(), self.act_dim))
                prev_stoch, _ = self.prior(prev_deter)
            deters[:, t] = prev_deter
            stochs[:, t] = prev_stoch
            logprobs[:, t] = logprob
            entropies[:, t] = entropy
        deter_first = initial_deters.unsqueeze(dim=1)
        stoch_first = initial_stochs.unsqueeze(dim=1)
        deters = torch.cat([deter_first, deters], dim=1)                # [B * T, H + 1, deter_dim]
        stochs = torch.cat([stoch_first, stochs], dim=1)                # [B * T, H + 1, stoch, classes]

        assert all(tensor.shape[:2] == (B * T, H + 1) for tensor in [deters, stochs])

        # compute return
        inputs = torch.cat([deters, stochs.view((*stochs.shape[:-2], -1))], -1)                         # [B * T, H + 1, deter_dim + stoch_dim]
        rew_pred = self.rew.pred(inputs)                                                                # [B * T, H + 1]
        con_prob = torch.exp(log_sigmoid(self.con.net(inputs).squeeze(-1)))                             # [B * T, H + 1]
        voffset, vscale = self.valnorm.stats()
        val = self.val.pred(inputs) * vscale + voffset                                                  # [B * T, H + 1]
        weight = torch.cumprod(con_prob, 1)                                                             # [B * T, H + 1]
        last = torch.zeros_like(con_prob)                                                               # [B * T, H + 1]
        term = 1 - con_prob                                                                             # [B * T, H + 1]
        ret = lambda_return(last, term, rew_pred, val, lam = 0.95)                                      # [B * T, H]

        # policy loss
        _, rscale = self.retnorm(ret, update = True)
        adv = (ret - val[:, :-1]) / rscale                                                                  # [B * T, H]
        aoffset, ascale = self.advnorm(adv, update = True)
        adv_normed = ((adv - aoffset) / ascale)                                                             # [B * T, H]
        pol_loss = weight[:, :-1].detach() * - (adv_normed.detach() * logprobs + 3e-4 * entropies)          # [B * T, H]
        pol_ent = entropies.mean()

        # critic loss
        voffset, vscale = self.valnorm(ret, update = True)
        tar_normed = (ret - voffset) / vscale                                                                                                           # [B * T, H]
        tar_padded = torch.cat([tar_normed, 0 * tar_normed[:, -1:]], dim=1).detach()                                                                    # [B * T, H + 1]
        val_loss = weight[:, :-1].detach() * (self.val.loss(inputs, tar_padded) + self.val.loss(inputs, self.slowval.pred(inputs).detach()))[:, :-1]    # [B * T, H]
        val_loss = val_loss.mean(1).reshape((B, T))                                                                                                     # [B, T]

        # replay loss
        last, term, rew = batches['is_lasts'], batches['is_terminals'], batches['rewards']  # [B, T]
        boot = ret[:, 0].reshape(B, T)                                                      # [B, T]
        rep_deters, rep_stochs, last, term, rew, boot = rep_deter[:, -T:], rep_stoch[:, -T:], last[:, -T:], term[:, -T:], rew[:, -T:], boot[:, -T:]
        inputs = torch.cat([rep_deters, rep_stochs.view(B, T, S * C)], dim=-1)  # [B, T, dim_deter, dim_stoch]
        voffset, vscale = self.valnorm.stats()
        val = self.val.pred(inputs) * vscale + voffset                          # [B, T]
        disc = 1 - 1 / self.horizon
        weight = (1 - last.float())                                             # [B, T]
        ret = lambda_return(last, term, rew, val, lam = 0.95, disc=disc)        # [B, T - 1]

        voffset, vscale = self.valnorm(ret, update = True)
        ret_normed = (ret - voffset) / vscale                                                                                                   # [B, T - 1]
        ret_padded = torch.cat([ret_normed, 0 * ret_normed[:, -1:]], dim=1).detach()                                                            # [B, T]
        repval_loss = weight[:, :-1] * (self.val.loss(inputs, ret_padded) + self.val.loss(inputs, self.slowval.pred(inputs).detach()))[:, :-1]  # [B, T - 1]

        losses = {'dyn': dyn_loss, 'rep': rep_loss, 'dec': dec_loss, 'rew': rew_loss, 'con': con_loss, 'pol': pol_loss, 'val': val_loss, 'repval': repval_loss}
        scales = {'dyn': 1.0, 'rep': 0.1, 'dec': 1.0, 'rew': 1.0, 'con': 1.0, 'pol': 1.0, 'val': 1.0, 'repval': 0.3}
        losses = {component: losses[component].mean() * scales[component] for component in losses.keys()}
        metrics = {'dyn_ent': dyn_ent.detach(), 'rep_ent': rep_ent.detach(), 'pol_ent': pol_ent.detach()}

        return losses, metrics

    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        checkpoint = {
            'rew': self.rew.state_dict(),
            'con': self.con.state_dict(),
            'sequence': self.sequence.state_dict(),
            'enc': self.enc.state_dict(),
            'posterior': self.posterior.state_dict(),
            'prior': self.prior.state_dict(),
            'dec': self.dec.state_dict(),
            'pol': self.pol.state_dict(),
            'val': self.val.state_dict(),
            'slowval': self.slowval.state_dict(),
            'retnorm': self.retnorm.state_dict(),
            'valnorm': self.valnorm.state_dict(),
            'advnorm': self.advnorm.state_dict(),
            'total_episodes': self.total_episodes,
            'gradient_steps': self.gradient_steps,
            'best_score': self.best_score,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        checkpoint = torch.load(path, map_location=self.device)
        self.rew.load_state_dict(checkpoint['rew'])
        self.con.load_state_dict(checkpoint['con'])
        self.sequence.load_state_dict(checkpoint['sequence'])
        self.enc.load_state_dict(checkpoint['enc'])
        self.posterior.load_state_dict(checkpoint['posterior'])
        self.prior.load_state_dict(checkpoint['prior'])
        self.dec.load_state_dict(checkpoint['dec'])
        self.pol.load_state_dict(checkpoint['pol'])
        self.val.load_state_dict(checkpoint['val'])
        self.slowval.load_state_dict(checkpoint['slowval'])
        self.retnorm.load_state_dict(checkpoint['retnorm'])
        self.valnorm.load_state_dict(checkpoint['valnorm'])
        self.advnorm.load_state_dict(checkpoint['advnorm'])
        self.total_episodes = checkpoint['total_episodes']
        self.gradient_steps = checkpoint['gradient_steps']
        self.best_score = checkpoint['best_score']