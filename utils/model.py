import torch
import torch.nn as nn

from torch.distributions import Normal

from .config import Config


class PatchEmbed(nn.Module):
    def __init__(self, imageSize, patchSize=8, inChannels=3, embedDim=256):
        super().__init__()
        self.numPatches = (imageSize // patchSize) ** 2
        self.proj = nn.Conv2d(inChannels, embedDim, kernel_size=patchSize, stride=patchSize)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# TODO: Probably try some other backbone, this is very expensive
class Painter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.patch = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.encoderPos = nn.Parameter(torch.zeros(1, self.patch.numPatches + 1, config.embedDim))
        self.decoderPos = nn.Parameter(torch.zeros(1, self.patch.numPatches + 1, config.embedDim))

        encoderLayer = nn.TransformerEncoderLayer(
            d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoderLayer, num_layers=config.encoderLayers
        )

        decoderLayer = nn.TransformerDecoderLayer(
            d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoderLayer, num_layers=config.decoderLayers
        )

        self.discToken = nn.Parameter(torch.zeros([1, 1, config.embedDim]))
        self.actToken = nn.Parameter(torch.zeros([1, 1, config.embedDim]))
        
        self.norm1 = nn.LayerNorm(config.embedDim)
        self.norm2 = nn.LayerNorm(config.embedDim)

        self.discHead = nn.Linear(config.embedDim, config.numConcepts)

        self.genHead = nn.Sequential(
            nn.Linear(config.embedDim, config.embedDim),
            nn.ReLU(),
            nn.Linear(config.embedDim, config.strokeActionSize * 2)
        )

        self.criticHead = nn.Sequential(
            nn.Linear(config.embedDim, config.embedDim // 2),
            nn.ReLU(),
            nn.Linear(config.embedDim // 2, 1)
        )

    # TODO: Consider contrastive loss instead of fixed number of concepts
    def discriminate(self, image):
        x = self.patch(image)
        x = torch.cat([x, self.discToken.expand(x.shape[0], 1, -1)], dim=1)
        x = x + self.encoderPos
        x = self.encoder(x)
        x = self.norm1(x[:, -1])
        x = self.discHead(x)
        return x

    def act(self, canvas, conceptImage):
        x = self.patch(conceptImage)
        x = x + self.encoderPos[:, :-1]
        x = self.encoder(x)

        y = self.patch(canvas)
        y = torch.cat([y, self.actToken.expand(x.shape[0], 1, -1)], dim=1)
        y = y + self.decoderPos
        y = self.norm2(self.decoder(tgt=y, memory=x)[:, -1])

        raw = self.genHead(y)
        mean, logDev = raw.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        logDev = logDev.clamp(-4, 2)
        dist = Normal(mean, logDev.exp())

        value = self.criticHead(y)

        return dist, value