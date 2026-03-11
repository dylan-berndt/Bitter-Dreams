from utils import *
from ppo import *
import os


def train(config: Config, device: str = "cuda:0"):
    agents = [Painter(config).to(device) for _ in range(config.numAgents)]
    buffers = [RolloutBuffer(config) for _ in range(config.numAgents)]

    genOpts  = [torch.optim.Adam(
                    list(a.patch.parameters()) +
                    list(a.encoder.parameters()) +
                    list(a.decoder.parameters()) +
                    list(a.norm2.parameters()) +
                    list(a.genHead.parameters()) +
                    list(a.criticHead.parameters()) +
                    [a.encoderPos, a.decoderPos, a.actToken, a.criticToken],
                    lr=config.lr)
                for a in agents]

    discOpts = [torch.optim.Adam(
                    list(a.patch.parameters()) +
                    list(a.encoder.parameters()) +
                    list(a.norm1.parameters()) +
                    list(a.discHead.parameters()) +
                    [a.encoderPos, a.discToken],
                    lr=config.lr)
                for a in agents]
    
    print("Generating random canvases")
    pool = ConceptPool(config)
    pool.saveSnapshot(step=0)

    print("Pretraining discriminators")
    for i in range(config.burnInSteps):
        discriminatorUpdate(agents, pool, discOpts, config, device)
        print(f"\r{i + 1}/{config.burnInSteps}", end="")

    print()

    print("Starting main training loop")
    for update in range(config.totalUpdates):
        # ----------------------------------------------------------------
        # ROLLOUT — generators act, collect experience
        # ----------------------------------------------------------------
        for agentIdx, (agent, buffer) in enumerate(zip(agents, buffers)):
            buffer.reset()
            agent.eval()
            peers = [a for i, a in enumerate(agents) if i != agentIdx]

            for _ in range(config.rolloutEpisodes):
                concepts, conceptIDs = pool.sample(config.batchSize)
                concepts, conceptIDs = concepts.to(device), conceptIDs.to(device)

                transitions, nextValue = runEpisode(
                    agent, concepts, conceptIDs, peers, pool, config, device
                )

                for t in transitions:
                    buffer.push(t)
                buffer.endEpisode(nextValue)

                # Push finished canvas to shared pool
                finishedCanvas = transitions[-1].canvas
                pool.push(conceptIDs.cpu(), finishedCanvas, agentIdx)

            print(f"\rAgent {agentIdx} generation complete", end="")

        pool.saveSnapshot(step=update + 1)

        # ----------------------------------------------------------------
        # PPO UPDATE — one agent at a time
        # ----------------------------------------------------------------
        for agentIdx, (agent, buffer, opt) in enumerate(zip(agents, buffers, genOpts)):
            agent.train()
            stats = ppoUpdate(agent, buffer, opt, config, device)

            print(f"\rAgent {agentIdx} PPO update complete", end="")

        # ----------------------------------------------------------------
        # DISCRIMINATOR UPDATE
        # ----------------------------------------------------------------
        discriminatorUpdate(agents, pool, discOpts, config, device)

        if update % 10 == 0:
            print()
            avgReward = float(np.mean([
                np.mean([t.reward for t in b.transitions]) for b in buffers
            ]))
            print(
                f"Update {update:4d} | "
                f"reward={avgReward:.4f} | "
                f"policy={stats['policyLoss']:.4f} | "
                f"value={stats['valueLoss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | "
            )

        print(f"\rStep {update}/{config.totalUpdates} complete", end="")


# TODO: Visualizer, checkpointing, testing, etc.
if torch.cuda.is_available():
    train(Config().load(os.path.join("configs", "config.json")))