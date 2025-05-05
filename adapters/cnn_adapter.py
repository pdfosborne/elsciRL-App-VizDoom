import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class DefaultAdapter:
    def __init__(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )

    def adapter(self, state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        img = state.screen_buffer  # shape CxHxW
        tensor_img = self.transform(np.transpose(img, (1, 2, 0)))
        tensor_img = tensor_img.unsqueeze(0)
        with torch.no_grad():
            if encode:
                features = self.cnn(tensor_img)
                return features.squeeze(0)
        return tensor_img.squeeze(0)
