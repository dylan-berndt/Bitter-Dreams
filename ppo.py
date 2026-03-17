from utils import *
import numpy as np


def softmaxReward(finishedCanvas, peers, conceptIDs, config):
    with torch.no_grad():
        B = finishedCanvas.shape[0]
        idx = torch.arange(B)
        peerProbs = []

        finishedCanvas = (finishedCanvas - 0.5) / 0.5

        for peer in peers:
            probs = nn.functional.softmax(peer.discriminate(finishedCanvas), dim=-1)
            peerProbs.append(probs[idx, conceptIDs])

        recognitionReward = torch.stack(peerProbs)

        # diff = (finishedCanvas - conceptImage).pow(2).mean(dim=[1, 2, 3])

        return (recognitionReward).mean(dim=0)
    

def contrastiveReward(canvas, peers, conceptIDs, config):
    with torch.no_grad():
        B = canvas.shape[0]
        idx = torch.arange(B, device=canvas.device)
        scores = []
        for peer in peers:
            probs = torch.softmax(peer.discriminate(canvas), dim=-1)
            correct = probs[idx, conceptIDs]
            # subtract mean probability of wrong concepts
            wrong = (probs.sum(dim=-1) - correct) / (config.numConcepts - 1)
            scores.append(correct - wrong)
        return torch.stack(scores).mean(dim=0)


def terminalReward(finishedCanvas, conceptImage, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        B = finishedCanvas.shape[0]
        idx = torch.arange(B)
        peerLogProbs = []
        baselineLogProbs = []

        finishedCanvas = (finishedCanvas - 0.5) / 0.5

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

        return (config.stepRewardWeight * torch.stack(deltas)).mean(dim=0)
    

def runEpisode(agent: Painter, concepts: torch.Tensor, conceptIDs: torch.Tensor, peers: list[Painter], pool: ConceptPool, config: Config, device: str):
    canvases = torch.ones(concepts.shape[0], 3, config.imageSize, config.imageSize, device=device)
    transitions = []

    for step in range(config.maxStrokes):
        isTerminal = (step == config.maxStrokes - 1)

        with torch.no_grad():
            dist, value = agent.act((canvases - 0.5) / 0.5, concepts)
            pre = dist.sample()
            logProb = dist.log_prob(pre).sum(-1)
            stroke = ((pre / 4) + 0.5).clamp(0, 1)

        prevCanvases = canvases
        # stroke = (action + 1) / 2

        canvases = drawStroke(canvases, stroke)

        if isTerminal:
            reward = softmaxReward((canvases - 0.5) / 0.5, peers, conceptIDs, config).cpu()
        elif config.includeStepReward:
            reward = denseReward((canvases - 0.5) / 0.5, (prevCanvases - 0.5) / 0.5, peers, conceptIDs, config).cpu()
        else:
            reward = torch.zeros(canvases.shape[0])

        transitions.append(Transition(
            canvas  = prevCanvases.cpu(),
            concept = concepts.cpu(),
            action  = pre.cpu(),
            logProb = logProb.cpu(),
            value   = value.squeeze(-1).cpu(),
            reward  = reward,
            done    = isTerminal,
        ))

    return transitions, torch.zeros([canvases.shape[0]])


def ppoUpdate(
    agent:     Painter,
    buffer:    RolloutBuffer,
    actorOptimizer: torch.optim.Optimizer,
    criticOptimizer: torch.optim.Optimizer,
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

    rewards = torch.stack([t.reward for t in buffer.transitions]).flatten()
    print()
    print(f"reward mean={rewards.mean():.4f} std={rewards.std():.4f} "
          f"min={rewards.min():.4f} max={rewards.max():.4f}")

    print(f"advantages pre-norm: mean={advantages.mean():.4f} std={advantages.std():.4f} "
      f"min={advantages.min():.4f} max={advantages.max():.4f}")
    print()

    stats = {"policyLoss": [], "valueLoss": [], "entropy": []}

    for _ in range(config.nEpochs):
        indices = torch.randperm(N)
        for start in range(0, N, config.miniBatchSize):
            idx = indices[start:start + config.miniBatchSize]
            dist, values = agent.act((canvases[idx] - 0.5) / 0.5, concepts[idx])

            newLogProbs = dist.log_prob(actions[idx]).sum(-1)
            entropy     = dist.entropy().mean()

            highAdv = advantages[idx] > 0.5
            lowAdv  = advantages[idx] < -0.5

            ratio  = (newLogProbs - oldLogProbs[idx]).exp()
            # print(f"ratio for high-advantage samples: {ratio[highAdv].mean():.4f}" if highAdv.any() else "no high-adv samples")
            # print(f"ratio for low-advantage samples:  {ratio[lowAdv].mean():.4f}" if lowAdv.any() else "no low-adv samples")
            # print(f"ratio mean={ratio.mean():.4f} std={ratio.std():.4f}")
            surr1  = ratio * advantages[idx]
            # print(f"surr1 mean for high-adv: {surr1[highAdv].mean():.4f}" if highAdv.any() else "")
            # print(f"surr1 mean for low-adv:  {surr1[lowAdv].mean():.4f}" if lowAdv.any() else "")
            surr2  = ratio.clamp(1 - config.clip, 1 + config.clip) * advantages[idx]
            policyLoss = -torch.min(surr1, surr2).mean()

            # print(f"returns mean={returns[idx].mean():.4f} std={returns[idx].std():.4f}")
            # print(f"correlation returns/advantages: {torch.corrcoef(torch.stack([returns[idx], advantages[idx]]))[0,1]:.4f}")

            values         = values.squeeze(-1)
            valuesClipped  = oldValues[idx].squeeze(-1) + \
                            (values - oldValues[idx].squeeze(-1)).clamp(-config.clip, config.clip)
            valueLoss = torch.max(
                nn.functional.mse_loss(values, returns[idx]),
                nn.functional.mse_loss(valuesClipped, returns[idx]),
            )

            policyLoss = policyLoss - config.entropyCoefficient * entropy

            # loss = policyLoss + config.valueCoefficient * valueLoss - config.entropyCoefficient * entropy

            criticOptimizer.zero_grad()
            valueLoss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), config.gradNorm)
            criticOptimizer.step()

            actorOptimizer.zero_grad()
            policyLoss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), config.gradNorm)
            actorOptimizer.step()

            stats["policyLoss"].append(policyLoss.item())
            stats["valueLoss"].append(valueLoss.item())
            stats["entropy"].append(entropy.item())

            # dist2, _ = agent.act((canvases[idx] - 0.5) / 0.5, concepts[idx])
            # newLogProbs2 = dist2.log_prob(actions[idx]).sum(-1)
            # ratio2 = (newLogProbs2 - oldLogProbs[idx]).exp()
            
            # print(f"ratio after update - high-adv: {ratio2[highAdv].mean():.4f}" if highAdv.any() else "")
            # print(f"ratio after update - low-adv:  {ratio2[lowAdv].mean():.4f}" if lowAdv.any() else "")

            if (ratio - 1.0).abs().mean() > 0.2:
                return {k: float(np.mean(v)) for k, v in stats.items()}

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


def generatorBurnIn(agents, pool, genOpts, config, device, steps=100):
    for i, (agent, opt) in enumerate(zip(agents, genOpts)):
        for step in range(steps):
            concepts, conceptIDs = pool.sample(config.batchSize)
            concepts = concepts.to(device)

            canvas = torch.ones(config.batchSize, 3, config.imageSize, config.imageSize, device=device)
            
            for s in range(config.maxStrokes):
                dist, _ = agent.act((canvas - 0.5) / 0.5, concepts)
                action = dist.mean  # use mean directly, no sampling
                stroke = (action / 4) + 0.5
                canvas = drawStroke(canvas, stroke)

            # Supervise directly against the concept image
            loss = nn.functional.mse_loss(canvas, concepts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.gradNorm)
            opt.step()

            print(f"\rGenerator burn-in {step}/{steps} | loss={loss.item():.4f}", end="")

        pool.push(conceptIDs.cpu(), canvas.detach().cpu(), agentIdx=i, step=0)  # Add burn-in samples to pool
    print()