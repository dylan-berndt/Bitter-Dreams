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
    logProb:   torch.Tensor   # scalar
    value:     torch.Tensor   # scalar — critic estimate V(s_t)
    reward:    float          # r_t (0 except terminal, plus optional dense bonus)
    done:      bool


class RolloutBuffer:
    def __init__(self, config: Config):
        self.config = config
        self.transitions: list[Transition] = []
        self._episodeStart = 0

    def push(self, t: Transition):
        self.transitions.append(t)

    def endEpisode(self, value: torch.Tensor):
        episode = self.transitions[self._episodeStart:]
        self._compute_gae(episode, value)
        self._episodeStart = len(self.transitions)

    def _compute_gae(self, episode: list[Transition], value: torch.Tensor):
        gae = 0

        nextValue = value

        for t in reversed(episode):
            valueT = 0 if t.done else nextValue
            delta = t.reward + self.config.gamma * valueT - t.value
            gae = delta + self.config.gamma * self.config.gaeLambda * (0 if t.done else gae)
            t.advantage = gae
            t.ret = gae + t.value
            nextValue = t.value

    def finalise(self):
        assert hasattr(self.transitions[0], 'advantage'), \
            "endEpisode() must be called before finalise()"

        canvases   = torch.stack([t.canvas  for t in self.transitions])  # (T, B, 3, H, W)
        concepts   = torch.stack([t.concept for t in self.transitions])
        actions    = torch.stack([t.action  for t in self.transitions])  # (T, B, strokeDim)
        logProbs   = torch.stack([t.logProb for t in self.transitions])  # (T, B)
        values     = torch.stack([t.value   for t in self.transitions])  # (T, B, 1)
        advantages = torch.stack([t.advantage for t in self.transitions])
        returns    = torch.stack([t.ret       for t in self.transitions])

        # Flatten time and batch into a single sample dimension
        def flat(x): return x.flatten(0, 1)

        canvases  = flat(canvases)    # (T*B, 3, H, W)
        concepts  = flat(concepts)
        actions   = flat(actions)     # (T*B, strokeDim)
        logProbs  = flat(logProbs)    # (T*B,)
        values    = flat(values)      # (T*B, 1)
        advantages = flat(advantages)
        returns    = flat(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return canvases, concepts, actions, logProbs.detach(), values.detach(), advantages, returns

    def reset(self):
        self.transitions.clear()
        self._episodeStart = 0


# TODO: Refactor as Dataset?
class ConceptPool:
    def __init__(self, config: Config):
        self.pools = [deque(maxlen=config.poolSize) for _ in range(config.numConcepts)]
        self.config = config

        self._initialize()

    def push(self, conceptIDs, images, agentIdx):
        for (image, conceptID) in zip(images.cpu(), conceptIDs.cpu()):
            self.pools[conceptID].append((image, agentIdx))

    def sample(self, numSamples):
        conceptIDs = random.choices(list(range(len(self.pools))), k=numSamples)
        images = []
        for conceptID in conceptIDs:
            choice = random.choice(self.pools[conceptID])
            images.append(choice[0])

        return torch.stack(images), torch.tensor(conceptIDs, dtype=torch.long)

    def _initialize(self):
        canvases = torch.ones([self.config.numConcepts * self.config.samplesPerConcept, 3, self.config.imageSize, self.config.imageSize])
        strokes = torch.rand(self.config.numConcepts * self.config.samplesPerConcept, self.config.maxStrokes, self.config.strokeActionSize)

        canvases = drawStrokes(canvases, strokes)

        for conceptID in range(self.config.numConcepts):
            for sample in range(self.config.samplesPerConcept):
                self.pools[conceptID].append((canvases[conceptID * self.config.samplesPerConcept + sample].cpu(), -1))


def drawStroke(canvases: torch.Tensor, stroke: torch.Tensor) -> torch.Tensor:
    """
    canvas: (B, 3, H, W)
    stroke: (B, 8) — (x1, y1, x2, y2, r, g, b, radius) all in [0, 1]
    returns: (B, 3, H, W)
    """
    B, _, H, W = canvases.shape
    device = canvases.device

    x1     = stroke[:, 0].view(B, 1, 1)
    y1     = stroke[:, 1].view(B, 1, 1)
    x2     = stroke[:, 2].view(B, 1, 1)
    y2     = stroke[:, 3].view(B, 1, 1)
    r      = stroke[:, 4].view(B, 1, 1)
    g      = stroke[:, 5].view(B, 1, 1)
    b      = stroke[:, 6].view(B, 1, 1)
    radius = stroke[:, 7].view(B, 1, 1).clamp(0.01, 0.1)

    coords = torch.linspace(0, 1, H, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')  # (H, W)

    # Vector from p1 to p2, and from p1 to each pixel
    dx = x2 - x1   # (B, 1, 1)
    dy = y2 - y1
    px = xx - x1   # (B, H, W)
    py = yy - y1

    # Project pixel onto the line segment, clamped to [0, 1]
    # t=0 means closest point is p1, t=1 means p2
    lenSq = (dx * dx + dy * dy).clamp(min=1e-8)
    t = ((px * dx + py * dy) / lenSq).clamp(0, 1)  # (B, H, W)

    # Closest point on segment to each pixel
    closestX = x1 + t * dx
    closestY = y1 + t * dy

    # Distance from each pixel to the closest point on the segment
    dist = ((xx - closestX) ** 2 + (yy - closestY) ** 2).sqrt()  # (B, H, W)

    # Hard mask — 1 inside the line, 0 outside
    # Use a small sigmoid softening at the boundary for differentiability
    sharpness = 200.0
    mask = torch.sigmoid(sharpness * (radius - dist)).unsqueeze(1)  # (B, 1, H, W)

    color = torch.stack([r, g, b], dim=1)  # (B, 3, 1, 1)
    return (canvases * (1 - mask) + color * mask).clamp(0, 1)


def drawStrokes(canvases: torch.Tensor, strokes: torch.Tensor):
    for s in range(strokes.shape[1]):
        canvases = drawStroke(canvases, strokes[:, s])

    return canvases

