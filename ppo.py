from utils import *
import numpy as np


def terminalReward(finishedCanvas, conceptImage, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        peerLogProbs = []
        baselineLogProbs = []

        for peer in peers:
            canvasLogits = peer.discriminate(finishedCanvas)
            baselineLogits = peer.discriminate(conceptImage)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[0, conceptID]
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[0, conceptID]

            peerLogProbs.append(canvasLP)
            baselineLogProbs.append(baselineLP)

        diversityBonus = 0.0
        # L2 pixel distance, replace with lpips.LPIPS for perceptual distance
        diff = (finishedCanvas - conceptImage).pow(2).mean(dim=[1, 2, 3])
        diversityBonus = diff.mean().item()

        recognitionReward = (torch.mean(torch.stack(peerLogProbs)) - torch.mean(torch.stack(baselineLogProbs))).item()

        return recognitionReward + config.dissimilarityWeight * diversityBonus
    

def denseReward(canvas, prevCanvas, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        deltas = []

        for peer in peers:
            canvasLogits = peer.discriminate(canvas)
            baselineLogits = peer.discriminate(prevCanvas)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[0, conceptID]
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[0, conceptID]

            deltas.append(canvasLP - baselineLP)

        return config.stepRewardWeight * torch.mean(deltas).item()
    

def runEpisode(agent: Painter, concepts: torch.Tensor, conceptIDs: torch.Tensor, peers: list[Painter], pool: ConceptPool, config: Config, device: str):
    canvases = torch.ones(concepts.shape[0], 3, config.imageSize, config.imageSize, device=device)
    transitions = []

    for step in range(config.maxStrokes):
        isTerminal = (step == config.maxStrokes - 1)

        with torch.no_grad():
            dist, value = agent.act(canvases, concepts)
            action = dist.sample()
            logProb = dist.log_prob(action).sum(-1)

        prevCanvases = canvases
        stroke = (action + 1) / 2

        canvases = drawStroke(canvases, stroke)

        if isTerminal:
            reward = terminalReward(canvases, concepts, peers, conceptIDs, config)
        elif config.includeStepReward:
            reward = denseReward(canvases, prevCanvases, peers, conceptIDs, config)
        else:
            reward = 0

        transitions.append(Transition(
            canvas  = canvases.cpu(),
            concept = concepts.cpu(),
            action  = action.cpu(),
            logProb = logProb.cpu(),
            value   = value.squeeze(-1).cpu(),
            reward  = reward,
            done    = isTerminal,
        ))

    return transitions, torch.zeros([canvases.shape[0]])


def ppoUpdate(
    agent:     Painter,
    buffer:    RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    config:    Config,
    device:    str,
) -> dict:
    canvases, concepts, actions, oldLogProbs, \
        oldValues, advantages, returns = buffer.finalise()

    canvases = canvases.to(device)
    concepts = concepts.to(device)
    actions = actions.to(device)
    oldLogProbs = oldLogProbs.to(device)
    oldValues = oldValues.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    stats = {"policyLoss": [], "valueLoss": [], "entropy": []}

    for _ in range(config.nEpochs):
        dist, values = agent.act(canvases, concepts)

        newLogProbs = dist.log_prob(actions).sum(-1)
        entropy     = dist.entropy().mean()

        ratio  = (newLogProbs - oldLogProbs).exp()
        surr1  = ratio * advantages
        surr2  = ratio.clamp(1 - config.clip, 1 + config.clip) * advantages
        policyLoss = -torch.min(surr1, surr2).mean()

        values         = values.squeeze(-1)
        valuesClipped  = oldValues.squeeze(-1) + \
                         (values - oldValues.squeeze(-1)).clamp(-config.clip, config.clip)
        valueLoss = torch.max(
            nn.functional.mse_loss(values, returns),
            nn.functional.mse_loss(valuesClipped, returns),
        )

        loss = policyLoss + config.valueCoefficient * valueLoss - config.entropyCoefficient * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), config.gradNorm)
        optimizer.step()

        stats["policyLoss"].append(policyLoss.item())
        stats["valueLoss"].append(valueLoss.item())
        stats["entropy"].append(entropy.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


def discriminatorUpdate(
    agents:     list[Painter],
    pool:       ConceptPool,
    optimizers: list[torch.optim.Optimizer],
    config:     Config,
    device:     str
):
    # TODO: Better random sampling?
    images, labels = pool.sample(config.burnInBatchSize)

    images = images.to(device)
    labels = labels.to(device)

    for agent, optimizer in zip(agents, optimizers):
        logits = agent.discriminate(images)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()