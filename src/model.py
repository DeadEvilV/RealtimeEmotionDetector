from torch import nn

class EmotionDetecterCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(0.2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(512)

        self.adaptiveavg = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512 * 1 * 1, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        residual = x

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x += residual

        residual = x

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.adaptiveavg(x)
        x = self.dropout

        x += residual

        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x

