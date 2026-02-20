import torch

class Memory:

    def __init__(self, device=None, sequence_length=64, max_episodes=500):
        self.device = device if device is not None else torch.device("cpu")
        self.sequence_length = sequence_length
        self.max_episodes = max_episodes

        self.episodes = []
        self.traj_index = []  # (episode_id, start)

    def add_episode(self, episode):
        L = episode["observations"].shape[0]

        eid = len(self.episodes)
        self.episodes.append(episode)

        for t in range(L - self.sequence_length + 1):
            self.traj_index.append((eid, t))

        # Evict old episodes if needed
        if len(self.episodes) > self.max_episodes:
            self._evict_oldest()

    def _evict_oldest(self):
        # remove episode 0
        removed = self.episodes.pop(0)

        L = removed["observations"].shape[0]
        num_trajs = L - self.sequence_length + 1

        # remove its trajectories
        self.traj_index = self.traj_index[num_trajs:]

        # shift episode ids
        self.traj_index = [
            (eid - 1, t) for (eid, t) in self.traj_index
        ]

    def sample(self, num_sequences=16, device=None):
        if device is None:
            device = self.device

        idxs = torch.randint(0, len(self.traj_index), (num_sequences,), device="cpu")

        sequences = {k: [] for k in [
            "observations",
            "actions", 
            "rewards",
            "deters", 
            "stochs",
            "is_firsts", 
            "is_terminals", 
            "is_lasts",
        ]}

        batch_indices = []
        T = self.sequence_length

        for idx in idxs.tolist():
            episode_id, start = self.traj_index[idx]
            ep = self.episodes[episode_id]
            end = start + T

            batch_indices.append((episode_id, start))

            for k in sequences:
                sequences[k].append(ep[k][start:end])

        batch = {
            k: torch.stack(v, dim=0).to(device)
            for k, v in sequences.items()
        }

        return batch, batch_indices
    
    def update_latents(self, indices, deters, stochs):
        # deters: [B, L=63, D]
        # stochs: [B, L=63, S]
        L = deters.shape[1]

        for b, (eid, start) in enumerate(indices):
            self.episodes[eid]["deters"][start+1 : start+1+L] = deters[b].cpu()
            self.episodes[eid]["stochs"][start+1 : start+1+L] = stochs[b].cpu()
