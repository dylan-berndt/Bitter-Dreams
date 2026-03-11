import streamlit as st
import torch
import os
import glob
import numpy as np


@st.cache_data(ttl=10) 
def listSnapshots(snapshotDir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(snapshotDir, "pool*.pt")))
    return paths


@st.cache_data
def loadSnapshot(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def tensorToNumpy(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor -> (H, W, 3) uint8 numpy array."""
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def renderConceptGrid(images: torch.Tensor, maxCols: int = 8) -> list[np.ndarray]:
    """Returns list of numpy images for a concept's pool."""
    n = min(len(images), maxCols)
    return [tensorToNumpy(images[i]) for i in range(n)]


def main():
    st.set_page_config(
        page_title="Concept Pool Visualizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    snapshotDir = "snapshots"

    st.title("Concept Pool Visualizer")

    # Sidebar — snapshot selection
    with st.sidebar:
        st.header("Snapshot")

        snapshots = listSnapshots(snapshotDir)
        if not snapshots:
            st.error(f"No snapshots found in '{snapshotDir}'.\nMake sure training is running and savePoolSnapshot() is being called.")
            st.stop()

        snapshotLabels = {os.path.basename(p).removesuffix(".pt").replace("pool_", "step "): p for p in snapshots}
        selectedLabel = st.select_slider(
            "Training step",
            options=list(snapshotLabels.keys()),
            value=list(snapshotLabels.keys())[-1],  # default to latest
        )
        selectedPath = snapshotLabels[selectedLabel]

        autoRefresh = st.checkbox("Auto-refresh (10s)", value=False)
        if autoRefresh:
            st.rerun()

        st.divider()
        st.header("Display")
        maxCols = st.slider("Max images per concept", 1, 32, 8)
        imageScale = st.slider("Image scale", 1, 8, 4)
        showEmpty = st.checkbox("Show empty concepts", value=False)

    # Load selected snapshot
    snapshot = loadSnapshot(selectedPath)
    step = snapshot["step"]
    concepts = snapshot["concepts"]   # dict: conceptID -> (N, 3, H, W)
    agentIDs = snapshot["agentIDs"]
    numConcepts = max(concepts.keys()) + 1 if concepts else 0

    # Top-level stats
    colA, colB, colC = st.columns(3)
    colA.metric("Training step", step)
    colB.metric("Concepts with images", len(concepts))
    colC.metric("Total pool images", sum(v.shape[0] for v in concepts.values()))

    st.divider()

    with st.sidebar:
        st.header("Filter")
        visibleAgents = st.multiselect("Agent IDs", list(agentIDs.keys()), default=[])

        filterMode = st.radio("Show", ["All concepts", "Select range", "Select specific"])

        if filterMode == "Select range":
            lo, hi = st.slider("Concept ID range", 0, max(numConcepts - 1, 1), (0, min(15, numConcepts - 1)))
            visibleIDs = list(range(lo, hi + 1))
        elif filterMode == "Select specific":
            visibleIDs = st.multiselect("Concept IDs", list(range(numConcepts)), default=list(range(min(8, numConcepts))))
        else:
            visibleIDs = list(range(numConcepts))

    # Render each concept
    imageSize = None
    for conceptID in visibleIDs:
        agentMask = agentIDs[conceptID].isin(visibleAgents)
        agents = agentIDs[agentMask]

        images = concepts[conceptID][agentMask]    # (N, 3, H, W)
        H, W = images.shape[-2], images.shape[-1]
        imageSize = (H, W)
        N = images.shape[0]

        with st.expander(f"Concept {conceptID}  —  {N} images", expanded=True):
            npImages = renderConceptGrid(images, maxCols=maxCols)
            cols = st.columns(len(npImages))

            for i, (col, img) in enumerate(zip(cols, npImages)):
                # Scale up small images so they're visible
                scaledH = H * imageScale
                scaledW = W * imageScale
                col.header(f"Agent {agents[i] + 1}")
                col.image(img, width=scaledW)

    if imageSize:
        st.sidebar.divider()
        st.sidebar.caption(f"Image size: {imageSize[0]}×{imageSize[1]}")


if __name__ == "__main__":
    main()