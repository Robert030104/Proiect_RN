import torch
import torch.nn as nn

class DefectPredictor(nn.Module):
    """
    Retea neuronala MLP pentru predictia defectiunilor auto.
    Model definit pentru etapa de arhitectura (neantrenat).
    """

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # clasificare binara: normal / defect
        )

    def forward(self, x):
        return self.net(x)
