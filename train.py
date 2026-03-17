from utils import *
from ppo import *
import os


def train(config: Config, device: str = "cuda:0"):
    agents = [Painter(config).to(device) for _ in range(config.numAgents)]
    buffers = [RolloutBuffer(config) for _ in range(config.numAgents)]

    genOpts  = [torch.optim.Adam(a.actor.parameters(), lr=config.actor.lr) for a in agents]
    discOpts = [torch.optim.Adam(a.discriminator.parameters(), lr=config.discriminator.lr) for a in agents]
    criticOpts = [torch.optim.Adam(a.critic.parameters(), lr=config.critic.lr) for a in agents]

    print("Generating random canvases")
    pool = ConceptPool(config)
    pool.saveSnapshot(step=0)

    print("Pretraining discriminators")
    for i in range(config.burnInSteps):
        discriminatorUpdate(agents, pool, discOpts, config, device)

        if i % 5 == 0:
            images, labels = pool.sample(config.burnInBatchSize)
            with torch.no_grad():
                logits = agents[0].discriminate(images.to(device))
                probs = nn.functional.softmax(logits, dim=-1)
                maxConf = probs.max(dim=-1).values.mean().item()

        print(f"\r{i + 1}/{config.burnInSteps} Max Confidence: {maxConf:.4f}", end="")

    print()

    # print("Pretraining generators")
    # generatorBurnIn(agents, pool, genOpts, config, device, steps=config.generatorBurnInSteps)

    # print()

    # for i in range(config.burnInSteps // 2):
    #     discriminatorUpdate(agents, pool, discOpts, config, device)

    #     if i % 5 == 0:
    #         images, labels = pool.sample(config.burnInBatchSize)
    #         with torch.no_grad():
    #             logits = agents[0].discriminate(images.to(device))
    #             probs = nn.functional.softmax(logits, dim=-1)
    #             maxConf = probs.max(dim=-1).values.mean().item()

    #     print(f"\r{i + 1}/{config.burnInSteps} Max Confidence: {maxConf:.4f}", end="")

    # print()

    print("Starting main training loop")
    for update in range(config.totalUpdates):
        # ----------------------------------------------------------------
        # ROLLOUT — generators act, collect experience
        # ----------------------------------------------------------------
        for agentIdx, (agent, buffer) in enumerate(zip(agents, buffers)):
            buffer.reset()
            agent.eval()
            peers = [a for i, a in enumerate(agents) if i != agentIdx]
            # peers = agents

            for _ in range(config.rolloutEpisodes):
                concepts, conceptIDs = pool.sample(config.batchSize, agentIdx)
                concepts, conceptIDs = concepts.to(device), conceptIDs.to(device)

                transitions, nextValue = runEpisode(
                    agent, concepts, conceptIDs, peers, pool, config, device
                )

                for t in transitions:
                    buffer.push(t)
                buffer.endEpisode(nextValue)

                # Push finished canvas to shared pool
                finishedCanvas = transitions[-1].canvas
                if update > 100:
                    pool.push(conceptIDs.cpu(), finishedCanvas, agentIdx, update + 1)

            print(f"\rAgent {agentIdx} generation complete", end="")

        pool.saveSnapshot(step=update + 1)

        # ----------------------------------------------------------------
        # PPO UPDATE — one agent at a time
        # ----------------------------------------------------------------
        for agentIdx, (agent, buffer, actorOpt, criticOpt) in enumerate(zip(agents, buffers, genOpts, criticOpts)):
            agent.train()
            stats = ppoUpdate(agent, buffer, actorOpt, criticOpt, config, device)

            print(f"\rAgent {agentIdx} PPO update complete", end="")

        print("\nEvaluating similarity")
        with torch.no_grad():
            concepts, conceptIDs = pool.sample(8)
            concepts = concepts.to(device)
            # Get concept encoder representations
            tokens = agents[0].discriminator.patch(concepts)
            tokens = tokens + agents[0].discriminator.pos[:, :-1]
            memory = agents[0].discriminator.encoder(tokens)
            cls = memory.mean(dim=1)  # (8, D)
            # Pairwise cosine similarity
            cls = nn.functional.normalize(cls, dim=-1)
            sim = cls @ cls.T
            print(sim.round(decimals=2))

        print()

        # ----------------------------------------------------------------
        # DISCRIMINATOR UPDATE
        # ----------------------------------------------------------------
        # discriminatorUpdate(agents, pool, discOpts, config, device)

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