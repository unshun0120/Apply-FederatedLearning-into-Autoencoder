import torch
import torch.nn as nn

"""
VAE 的結構分為 Encoder、Latent Space(隱變數空間) 和 Decoder 三個部分：
    Encoder: 將輸入壓縮成潛在空間(latent space)的參數(均值 μ 和標準差 σ)。
    Latent Space: 使用 reparameterization trick 從常態分布中取樣 z。
    Decoder: 將 z 解碼回原本的數據空間。
"""
class vae(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(vae, self).__init__()

        # Encoder: 784 -> 400 -> (μ, logσ²)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一層隱藏層
        #　計算latent space的均值 μ，代表「最可能的潛在變數值」
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值 μ
        # 計算對數變異數 log(σ²)，用來決定 σ（標準差），即變數的不確定性。
        # 為什麼要輸出 log(σ²) 而不是 σ? -> 直接學習標準差 σ 可能會導致數值不穩定，取對數可以讓學習過程更穩定
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log(σ²)

        # Decoder: z -> 400 -> 784
        self.fc2 = nn.Linear(latent_dim, hidden_dim)  # Latent space -> 隱藏層
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # 隱藏層 -> 輸出層 (重建)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    """
    為什麼要 reparameterize ?為什麼不能直接 z ~ N(μ, σ²)?
    -> 因為直接從 N(μ, σ²) 取樣是不可微分的，會影響反向傳播
    reparameterize 將 z 轉換成：𝑧 = 𝜇 + σ * 𝜖
    這樣 μ 和 log(σ²) 仍然能參與梯度計算，使得 VAE 可以用梯度下降學習
    """
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 計算標準差 σ
        eps = torch.randn_like(std)  # 取標準常態分布的隨機數
        return mu + eps * std  # 𝑧 = 𝜇 + 𝜎 * 𝜖

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h))  # 使用 Sigmoid 限制輸出範圍在 (0,1)

    def forward(self, x):
        mu, logvar = self.encode(x)  # Flatten 輸入
        z = self.reparameterize(mu, logvar)  # 取樣 latent vector
        return self.decode(z), mu, logvar
  
    
class cnn_vae(nn.Module):
    def __init__(self, latent_dim=20):
        super(cnn_vae, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 7x7 -> 4x4
            nn.ReLU()
        )

        # Flatten 128x4x4 -> latent_dim
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # 將 z 還原回 128x4×4
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4x4 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14x14 -> 28x28
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)    
        return mu + eps * std         

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # 因為輸入到fc_mu, fc_logvar：大小必須是 [batch_size, 128x4x4] (MNIST)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.fc_decode(z)  # fc_decode 的輸出 shape = (batch, 2048)
        h = h.view(z.size(0), 128, 4, 4) 
        h = self.decoder(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)        
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)  
        return recon_x, mu, logvar