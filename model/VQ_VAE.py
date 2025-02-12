import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings    # Codebook 中有多少個 codeword
        self.embedding_dim = embedding_dim  # 每個 codeword 的維度
        self.beta = beta  # 控制 VQ loss 的影響程度

        # 初始化 Codebook (K x D)
        # num_embeddings (K)：代表 Codebook 的大小，即我們有多少個離散的 Codeword
        # embedding_dim (D)：代表 每個 Codeword 的維度，即 Codeword 的表徵向量有多少維度
        # nn.Embedding: 建立一個學習中的 Codebook，本質上是一個Lookup Table
        # 類似於 Codebook = torch.randn(num_embeddings, embedding_dim)
        # 每當輸入一個索引（編碼向量的最近 Codeword），就會回傳對應的 Codeword 向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.uniform_: 初始化 Codebook 的權重，避免初始值過大或過小導致收斂困難
        # self.embedding.weight：代表 Codebook 的參數矩陣 K × D，初始時是隨機數值
        #　.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)：　uniform_() 讓 Codeword 值 均勻地分布在 (-1.0 / K, 1.0 / K) 之間
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """
        z: (batch, embedding_dim)
        z 裡面有 B 個樣本，每個樣本是一個 D 維的向量。
        當 VQ-VAE 訓練時，每次 Encoder 產生的潛在變數 z, 都會與 Codebook 進行比對，找到最近的 Codeword 並用它來代表 z
        """
        # 計算 z 與 Codebook 之間的 L2 距離
        # 將 z 量化成 Codebook 裡最相近的 Codeword, 可以透過 L2 距離 (歐幾里得距離, Euclidean Distance) 來衡量
        # L2公式可看issues_log.md 
        distances = (
            # z ** 2:  代表對 z 中的每個元素做平方運算
            # dim=1 表示沿著 D 維度(每個向量的所有元素)做加總, 每個樣本的 D 個數值會被加總，變成一個單一數值 -> (B, 1)
            # keepdim=True 表示保持維度不變，即使進行加總後，仍然維持原來的軸數量
            # True, 輸出形狀為 (B, 1)，dim=1 仍然保留，只是該維度的大小變成 1。
            # False, 輸出形狀為 (B,)，即 dim=1 被壓縮掉了，變成 1D 向量。
            torch.sum(z ** 2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight ** 2, dim=1)
            # 這一項計算 z (B, D) 和 Codebook (K, D)轉置後(D, K)的內積 -> (B, K)
            # (B, K): 這代表batch中的每個 z 與 K 個 Codeword 的相似度
            - 2 * torch.matmul(z, self.embedding.weight.t())    # .t(): transpose
        )  # (B, K)
        """
        distance = (B, 1) + (K,) - (B, K) = (B, K)
        (B, 1), (K, )會自動變成 (B, K)
        (B, K): 每一行代表一個 z 到所有 Codeword 的 L2 距離，可以直接用 torch.argmin(distances, dim=1)找出最接近的Codeword
        """

        # 取距離最近的 codeword index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B, 1)
        
        # 取得最近的 codeword
        z_q = self.embedding(encoding_indices).view(z.shape)  # (B, embedding_dim)

        """
        VQ-VAE Loss = commitment_loss + embedding_loss, 分別對應 encoder 和 codebook 的學習方式
        commitment loss: 讓 encoder 產生的 z 更接近 codebook 中的對應向量 z_q
        detach(): z_q.detach() 讓 z_q 不回傳梯度，這樣 只有 encoder 受到這個 loss 的影響，而 codebook 不會更新
        * β (self.beta)：控制這個損失的權重，避免 encoder 太過依賴 codebook
        """
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.beta
        # embedding loss: 讓 codebook 學習更接近 z，減少誤差
        # 沒有 β：這個 loss 直接影響 codebook，使 codebook 的向量能更有效代表 encoder 產生的 z
        embedding_loss = F.mse_loss(z_q, z.detach())
        vq_loss = commitment_loss + embedding_loss

        """
        VQ-VAE 的 z_q 是透過 Nearest Neighbor Quantization取得的, 
        而這個過程是不可微分的, 導致梯度無法傳遞回 encoder。
        所以使用 Straight-Through Estimator (STE) 來解決這個問題
        """
        # 這行的目的就是讓梯度能夠從 z_q 傳回 z，但在 forward pass 時，z_q 仍然是量化後的值
        z_q = z + (z_q - z).detach()
        """
        (z_q - z).detach(): 
            z_q - z: 就是量化誤差 (quantization error)，表示 z 與對應的 z_q 之間的距離
            如果這個值很大，表示 z 跟 codebook 的 z_q 差距大，需要讓 encoder 產生更接近 codebook 的 z
            detach() 讓這個誤差不會影響 codebook 的更新，只影響 encoder。
        如果我們直接 z_q = z_q.detach()，這樣 z_q 在 backward pass 中就是個常數，這樣梯度無法回傳給 encoder, 導致 encoder 無法學習

        Forward Pass: 
            z_q = z + (z_q - z).detach()
            這實際上就是 z_q, 因為 (z_q - z).detach() 只是 z_q - z, 但梯度被截斷
            換句話說，這行在 forward pass 中並沒有改變輸出，依然是 z_q(量化後的值)
            
        Backward Pass: 
            z_q - z 這部分被 detach()，所以這段不會影響梯度計算
            由於 z_q 被重新定義成 z + (z_q - z).detach()，梯度在 z 時會直接傳回去，而不會影響 z_q
            這樣就像是 z_q ≈ z, 但不影響 forward pass 的值，讓 encoder 仍然能夠學習
        """

        return z_q, vq_loss

class vqvae(nn.Module):
    def __init__(self, input_dim=784, num_embeddings=512, embedding_dim=64):
        super(vqvae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, num_embeddings),
            nn.ReLU(),
            nn.Linear(num_embeddings, embedding_dim)
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, num_embeddings),
            nn.ReLU(),
            nn.Linear(num_embeddings, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 28*28)
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_recon = self.decoder(z_q)
        x_recon = x_recon.view(x.size(0), 1, 28, 28)  # 轉回影像格式
        return x_recon, vq_loss