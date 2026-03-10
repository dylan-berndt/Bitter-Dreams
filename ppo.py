from utils import *
import numpy as np


def terminalReward(finishedCanvas, conceptImage, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        peerLogProbs = []
        baselineLogProbs = []

        for peer in peers:
            canvasLogits = peer.discriminate(finishedCanvas)
            baselineLogits = peer.discriminate(conceptImage)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[0, conceptID].item()
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[0, conceptID].item()

            peerLogProbs.append(canvasLP)
            baselineLogProbs.append(baselineLP)

        diversityBonus = 0.0
        # L2 pixel distance, replace with lpips.LPIPS for perceptual distance
        diff = (finishedCanvas - conceptImage).pow(2).mean(dim=[1, 2, 3])
        diversityBonus = diff.mean().item()

        recognitionReward = float(np.mean(peerLogProbs) - np.mean(baselineLogProbs))

        return recognitionReward + config.dissimilarityWeight * diversityBonus
    

def denseReward(canvas, prevCanvas, peers: list[Painter], conceptID, config):
    with torch.no_grad():
        deltas = []

        for peer in peers:
            canvasLogits = peer.discriminate(canvas)
            baselineLogits = peer.discriminate(prevCanvas)

            canvasLP = nn.functional.softmax(canvasLogits, dim=-1)[0, conceptID].item()
            baselineLP = nn.functional.softmax(baselineLogits, dim=-1)[0, conceptID].item()

            deltas.append(canvasLP - baselineLP)

        return config.stepRewardWeight * float(np.mean(deltas))
    

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

        # canvases = 

        if isTerminal:
            reward = terminalReward(canvases, concepts, peers, conceptIDs, config)
        elif config.includeStepReward:
            reward = denseReward(canvases, prevCanvases, peers, conceptIDs, config)
        else:
            reward = 0

        transitions.append(Transition(
            canvas  = canvases.squeeze(0).cpu(),
            concept = concepts.squeeze(0).cpu(),
            action  = action.squeeze(0).cpu(),
            logProb = logProb.squeeze(0).cpu(),
            value   = value.squeeze(0).cpu(),
            reward  = reward,
            done    = isTerminal,
        ))

    return transitions


# TODO: PPO update, Discriminator update for pretraining discriminators
