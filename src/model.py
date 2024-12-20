from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.mismatch = False
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
            self.mismatch = True
        
    def forward(self, x):
        if self.mismatch:
            residual = self.relu(self.bn_shortcut(self.shortcut(x)))
        else:
            residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual
        return x
            
class EmotionDetecterCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout2d(0.15)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout_fc = nn.Dropout(0.25)
        
        self.rb1 = ResidualBlock(1, 32)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
                
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.rb2 = ResidualBlock(128, 256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.rb3 = ResidualBlock(256, 256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.adaptiveavg = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.rb1(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.rb2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout2(x)
        
        x = self.rb3(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.adaptiveavg(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x