from utils import *
import numpy as np


def softmaxReward(finishedCanvas, conceptImage, peers, conceptIDs, config):
    with torch.no_grad():
        B = finishedCanvas.shape[0]
        idx = torch.arange(B)
        peerProbs = []

        for peer in peers:
            probs = nn.functional.softmax(peer.discriminate(finishedCanvas), dim=-1)
            peerProbs.append(probs[idx, conceptIDs])

        recognitionReward = torch.stack(peerProbs).mean(dim=0)

        diff = (finishedCanvas - conceptImage).pow(2).mean(dim=[1, 2, 3])

        return recognitionReward + config.dissimilarityWeight * diff


def terminalReward(finishedCanvas, conceptImage, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        B = finishedCanvas.shape[0]
        idx = torch.arange(B)
        peerLogProbs = []
        baselineLogProbs = []

        for peer in peers:
            canvasLogits = peer.discriminate(finishedCanvas)
            baselineLogits = peer.discriminate(conceptImage)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[idx, conceptID]
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[idx, conceptID]

            peerLogProbs.append(canvasLP)
            baselineLogProbs.append(baselineLP)

        # L2 pixel distance, replace with lpips.LPIPS for perceptual distance
        diff = (finishedCanvas - conceptImage).pow(2).mean(dim=[1, 2, 3])

        recognitionReward = torch.stack(peerLogProbs).mean(dim=0) - torch.stack(baselineLogProbs).mean(dim=0)

        return recognitionReward + config.dissimilarityWeight * diff
    

def denseReward(canvas, prevCanvas, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        B = canvas.shape[0]
        idx = torch.arange(B)
        deltas = []

        for peer in peers:
            canvasLogits = peer.discriminate(canvas)
            baselineLogits = peer.discriminate(prevCanvas)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[idx, conceptID]
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[idx, conceptID]

            deltas.append(canvasLP - baselineLP)

        return config.stepRewardWeight * torch.stack(deltas).mean(dim=0)
    

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
            reward = softmaxReward(canvases, concepts, peers, conceptIDs, config)
        elif config.includeStepReward:
            reward = denseReward(canvases, prevCanvases, peers, conceptIDs, config)
        else:
            reward = torch.zeros(canvases.shape[0])

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

    N = canvases.shape[0]

    stats = {"policyLoss": [], "valueLoss": [], "entropy": []}

    for _ in range(config.nEpochs):
        indices = torch.randperm(N)
        for start in range(0, N, config.miniBatchSize):
            idx = indices[start:start + config.miniBatchSize]
            dist, values = agent.act(canvases[idx], concepts[idx])

            newLogProbs = dist.log_prob(actions[idx]).sum(-1)
            entropy     = dist.entropy().mean()

            ratio  = (newLogProbs - oldLogProbs[idx]).exp()
            surr1  = ratio * advantages[idx]
            surr2  = ratio.clamp(1 - config.clip, 1 + config.clip) * advantages[idx]
            policyLoss = -torch.min(surr1, surr2).mean()

            values         = values.squeeze(-1)
            valuesClipped  = oldValues[idx].squeeze(-1) + \
                            (values - oldValues[idx].squeeze(-1)).clamp(-config.clip, config.clip)
            valueLoss = torch.max(
                nn.functional.mse_loss(values, returns[idx]),
                nn.functional.mse_loss(valuesClipped, returns[idx]),
            )

            minEntropy = 0.5
            entropyLoss = nn.functional.relu(minEntropy - entropy)

            loss = policyLoss + config.valueCoefficient * valueLoss - config.entropyCoefficient * entropyLoss

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
    for agentIdx, (agent, optimizer) in enumerate(zip(agents, optimizers)):
        images, labels = pool.sample(config.burnInBatchSize, excludeAgent=agentIdx)
        
        if images is None:
            continue

        images = images.to(device)
        labels = labels.to(device)

        logits = agent.discriminate(images)
        loss   = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()