import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class cnn_autoencoder(nn.Module): 
    def __init__(self): 
        super(cnn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 確保 Encoder 的最終輸出尺寸 夠小但不會變成 0（避免網路無法學習）
            # stride=2: 圖片尺寸縮小一半, 用stride=1+MaxPool2d(2,2)有一樣的效果
            # conv2d 卷積計算公式: output size= (input_size - kernel_size + 2*padding)/stride + 1
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 28x28 -> 14x14
            # 對於 Autoencoder，有時不一定需要BatchNorm2d，特別是如果希望潛在空間保留更多變異性時
            # nn.BatchNorm2d(64), 
            nn.ReLU(), 

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(), 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7×7 → 14×14
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14×14 → 28×28
            nn.Tanh()  # 限制輸出範圍 [-1, 1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x