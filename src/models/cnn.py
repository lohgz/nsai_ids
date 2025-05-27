import torch.nn as nn

class CNN_Classifier(nn.Module):
    def __init__(self, num_features):
        super(CNN_Classifier, self).__init__()
        # 1 input channel, num_features length
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1)
        )
        self.flatten_size = (num_features // 2) * 64

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    # forward pass
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv(x)    
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x