import torch
import numpy as np

from dataclasses import dataclass, field
from collections import deque
import random
from .config import Config


@dataclass
class Transition:
    """One timestep within an episode."""
    canvas:    torch.Tensor   # (3, H, W)  canvas state before the stroke
    concept:   torch.Tensor   # (3, H, W)  concept image, constant per episode
    action:    torch.Tensor   # (stroke_dim,)
    log_prob:  torch.Tensor   # scalar
    value:     torch.Tensor   # scalar — critic estimate V(s_t)
    reward:    float          # r_t (0 except terminal, plus optional dense bonus)
    done:      bool


class RolloutBuffer:
    def __init__(self, config: Config):
        self.config = config
        self.transitions: list[Transition] = []
        self._episode_start_idx = 0

    def push(self, t: Transition):
        self.transitions.append(t)

    def endEpisode(self, value: torch.Tensor):
        episode = self.transitions[self._episode_start_idx:]
        self._compute_gae(episode, value)
        self._episode_start_idx = len(self.transitions)

    def _compute_gae(self, episode: list[Transition], value: torch.Tensor):
        gae = 0

        nextValue = value.item()

        for t in reversed(episode):
            valueT = 0 if t.done else nextValue
            delta = t.reward + self.config.gamma * valueT - t.value.item()
            gae = delta + self.config.gamma * self.config.gaeLambda * (0 if t.done else gae)
            t.advantage = gae
            t.ret = gae + t.value.item()
            nextValue = t.value.item()

    def finalise(self):
        assert hasattr(self.transitions[0], 'advantage'), \
            "endEpisode() must be called before finalise()"

        canvases   = torch.stack([t.canvas  for t in self.transitions])
        concepts   = torch.stack([t.concept for t in self.transitions])
        actions    = torch.stack([t.action  for t in self.transitions])
        logProbs   = torch.stack([t.logProb for t in self.transitions])
        values     = torch.stack([t.value   for t in self.transitions])
        advantages = torch.tensor([t.advantage for t in self.transitions], dtype=torch.float32)
        returns    = torch.tensor([t.ret       for t in self.transitions], dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return canvases, concepts, actions, logProbs.detach(), values.detach(), advantages, returns

    def reset(self):
        self.transitions.clear()
        self._episodeStart = 0


class ConceptPool:
    def __init__(self, config: Config):
        self.pools = [deque(maxlen=config.poolSize) for _ in range(config.numConcepts)]

        self._initialize()

    def push(self, conceptIDs, images):
        for (image, conceptID) in zip(images.cpu(), conceptIDs.cpu()):
            self.pools[conceptID].append(image)

    def sample(self, numSamples):
        conceptIDs = random.choices(list(range(len(self.pools))), numSamples)
        images = []
        for conceptID in conceptIDs:
            images.append(random.choice(self.pools[conceptID]))

        return torch.tensor(images), torch.tensor(conceptIDs)

    def _initialize(self):
        canvases = torch.ones([self.config.numConcepts * self.config.samplesPerConcept, 3, self.config.imageSize, self.config.imageSize])

        # TODO: Draw random canvases

        for conceptID in range(self.config.numConcepts):
            for sample in range(self.config.samplesPerConcept):
                self.pools.append(canvases[conceptID * self.config.samplesPerConcept + sample].cpu())


def drawStrokes(canvases, strokes):
    pass