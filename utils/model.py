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


class ViTEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.pos = nn.Parameter(torch.zeros(1, self.patch.numPatches + 1, config.embedDim))

        encoderLayer = nn.TransformerEncoderLayer(
            d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoderLayer, num_layers=config.encoderLayers
        )

        self.clsToken = nn.Parameter(torch.zeros([1, 1, config.embedDim]))
        self.norm = nn.LayerNorm(config.embedDim)

        self.head = nn.Sequential(
            nn.Linear(config.embedDim, config.embedDim),
            nn.ReLU(),
            nn.Linear(config.embedDim, config.outputShape)
        )

    def forward(self, image):
        x = self.patch(image)
        x = torch.cat([x, self.clsToken.expand(x.shape[0], 1, -1)], dim=1)
        x = x + self.pos
        x = self.norm(self.encoder(x))
        x = x[:, -1]
        return self.head(x)
    

class ViT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch1 = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.patch2 = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.pos1 = nn.Parameter(torch.zeros(1, self.patch1.numPatches, config.embedDim))
        self.pos2 = nn.Parameter(torch.zeros(1, self.patch2.numPatches + 1, config.embedDim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
                batch_first=True
            ),
            num_layers=config.encoderLayers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
                batch_first=True
            ),
            num_layers=config.decoderLayers
        )

        self.clsToken = nn.Parameter(torch.zeros([1, 1, config.embedDim]))
        self.norm = nn.LayerNorm(config.embedDim)
        self.head = nn.Sequential(
            nn.Linear(config.embedDim, config.embedDim),
            nn.ReLU(),
            nn.Linear(config.embedDim, config.outputShape)
        )

    def forward(self, canvas, conceptImage):
        x = self.patch1(conceptImage)
        x = x + self.pos1
        x = self.encoder(x)

        y = self.patch2(canvas)
        y = torch.cat([y, self.clsToken.expand(x.shape[0], 1, -1)], dim=1)
        y = y + self.pos2
        y = self.norm(self.decoder(tgt=y, memory=x))

        embedding = self.head(y[:, -1])

        return embedding
    

# TODO: Probably try some other backbone, this is very expensive
class Painter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.discriminator = ViTEncoder(config.discriminator)
        self.actor = ViT(config.actor)
        self.critic = ViT(config.critic)

        # Makes strokes start out in center
        nn.init.zeros_(self.actor.head[-1].weight)
        nn.init.zeros_(self.actor.head[-1].bias)

    # TODO: Consider contrastive loss instead of fixed number of concepts
    def discriminate(self, image):
        return self.discriminator(image)

    def act(self, canvas, conceptImage):
        with torch.no_grad():
            conceptTokens = self.discriminator.patch(conceptImage)
            conceptTokens = torch.cat([
                conceptTokens, 
                self.discriminator.clsToken.expand(conceptImage.shape[0], 1, -1)
            ], dim=1)
            conceptTokens = conceptTokens + self.discriminator.pos
            memory = self.discriminator.encoder(conceptTokens)

        y = self.actor.patch2(canvas)
        y = torch.cat([y, self.actor.clsToken.expand(canvas.shape[0], 1, -1)], dim=1)
        y = y + self.actor.pos2
        y = self.actor.norm(self.actor.decoder(tgt=y, memory=memory))
        
        embedding = self.actor.head(y[:, -1])

        # actEmb = self.actor(canvas, conceptImage)
        value = self.critic(canvas, conceptImage)

        mean, logDev = embedding.chunk(2, dim=-1)
        # mean = torch.tanh(mean)
        logDev = logDev.clamp(-5, 2)
        dist = Normal(mean, logDev.exp())

        return dist, value