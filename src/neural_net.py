"""神经网络模块（PyTorch）：用可训练的词嵌入做新闻分类。

思路和 TF-IDF + sklearn 那条线不同：
    不用稀疏的 TF-IDF 向量，而是给每个词学一个稠密向量（embedding），
    把一条新闻里所有词向量求平均，再经过一个小型 MLP 分类。

相比当年的版本，这版的关键升级：
    1. 在【测试集】上评估     —— 当年只报训练准确率(96.76%)，是自我欺骗
    2. 真正的隐藏层 + 激活    —— 当年号称 MLP，其实只有一个线性层
    3. 词表只从训练集构建     —— 和 TF-IDF 一样，杜绝数据泄露
    4. 和三个 sklearn 模型共用【同一套切分】—— 公平对比
"""

import os

# Windows + conda 常见的 OpenMP 运行时冲突，标准 workaround，必须在 import torch 之前设置。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD, UNK = 0, 1   # 两个特殊 token：补齐位、未登录词


# ---------- 词表 & 编码 ----------

def build_vocab(texts, max_size=20000, min_freq=2):
    """从【训练文本】构建词表：词 -> 整数 id。

    只用训练集构建（和 TF-IDF fit-on-train 同理，防泄露）。
    出现次数少于 min_freq 的稀有词不收，避免词表被一次性的拼写错误塞满。
    """
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for word, freq in counter.most_common(max_size - 2):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    """把一段文本变成一串 token id（词表里没有的词记作 <UNK>）。"""
    return [vocab.get(w, UNK) for w in text.split()]


def encode_labels(y_train, y_test):
    """把类别名（'Sports'...）映射成整数 0..3。返回 (训练标签, 测试标签, 类别顺序)。"""
    classes = sorted(set(y_train))
    to_idx = {c: i for i, c in enumerate(classes)}
    return [to_idx[c] for c in y_train], [to_idx[c] for c in y_test], classes


# ---------- 数据加载 ----------

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.samples = [encode(t, vocab) or [UNK] for t in texts]  # 空文本兜底成 [UNK]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """一个 batch 里各条新闻长度不一，补齐(pad)到等长才能堆成张量。"""
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=PAD)
    return xs, torch.tensor(ys)


def make_loaders(text_train, y_train, text_test, y_test, vocab, batch_size=64):
    """构建训练/测试 DataLoader。"""
    train_ds = NewsDataset(text_train, y_train, vocab)
    test_ds = NewsDataset(text_test, y_test, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, test_loader


# ---------- 模型 ----------

class TextMLP(nn.Module):
    """词嵌入 -> 平均池化 -> 隐藏层(ReLU) -> dropout -> 输出。

    比当年多了一个真正的隐藏层和非线性，这才配叫 MLP。
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                      # [batch, seq_len, embed_dim]

        # 掩码平均池化：只对真实词求平均，不让补齐位 PAD 稀释结果。
        mask = (x != PAD).unsqueeze(-1)              # [batch, seq_len, 1]
        summed = (emb * mask).sum(dim=1)             # [batch, embed_dim]
        counts = mask.sum(dim=1).clamp(min=1)        # 每条新闻的真实词数（至少 1）
        pooled = summed / counts                     # [batch, embed_dim]

        h = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(h)                           # [batch, num_classes]


# ---------- 训练 & 评估 ----------

@torch.no_grad()
def evaluate_nn(model, loader, device="cpu"):
    """在给定数据集上算准确率。"""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_nn(model, train_loader, test_loader, epochs=5, lr=1e-3, device="cpu"):
    """训练模型；每个 epoch 后记录训练 + 【测试】准确率，过拟合一眼可见。

    返回：
        model   —— 训练好的模型
        history —— {"train_acc": [...], "test_acc": [...]}，每个 epoch 一条，用来画曲线
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_acc": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()          # 1. 梯度清零
            loss = criterion(model(x), y)  # 2. 前向 + 算损失
            loss.backward()                # 3. 反向传播（算每个参数该往哪调）
            optimizer.step()               # 4. 沿梯度走一小步，更新参数

        train_acc = evaluate_nn(model, train_loader, device)
        test_acc = evaluate_nn(model, test_loader, device)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch}: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return model, history
