import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import os

# 创建模型保存目录
os.makedirs('model', exist_ok=True)

# 1. 加载数据
print("Loading data...")
df = pd.read_csv('data/imdb_balanced_10k.csv')

# 检查列名并自动适配
print(f"Columns found: {df.columns.tolist()}")

# 找到文本列和标签列
review_col = None
sentiment_col = None

for col in df.columns:
    if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
        review_col = col
    if 'sentiment' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
        sentiment_col = col

# 如果没找到，假设第一列是文本，第二列是标签
if review_col is None:
    review_col = df.columns[0]
if sentiment_col is None:
    sentiment_col = df.columns[1]

print(f"Using review column: {review_col}")
print(f"Using sentiment column: {sentiment_col}")

X = df[review_col].astype(str)
y = df[sentiment_col]

# 将 sentiment 转换为数值
if y.dtype == 'object':
    # 尝试常见映射
    if 'positive' in y.values or 'pos' in y.values:
        y = y.map({'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0, 'good': 1, 'bad': 0})
    else:
        # 如果不是标准格式，尝试转为数值
        y = y.astype('category').cat.codes

print(f"Classes: {y.unique()}")
print(f"Data loaded: {len(X)} samples")

# 2. TF-IDF 向量化
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X).toarray()

# 保存向量化器
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Vectorizer saved to model/vectorizer.pkl")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32).reshape(-1, 1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. 定义神经网络
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# 5. 训练模型
input_dim = X_train.shape[1]
model = SentimentNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Input dimension: {input_dim}")
print("Training model...")

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 6. 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    pred_classes = (predictions.numpy() >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, pred_classes)
    f1 = f1_score(y_test, pred_classes)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# 7. 保存模型和配置
torch.save(model.state_dict(), 'model/model.pt')
print("Model saved to model/model.pt")

config = {
    'input_dim': input_dim,
    'model_type': 'TF-IDF + Feedforward NN',
    'max_features': 5000,
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': epochs,
    'batch_size': 64,
    'accuracy': float(accuracy),
    'f1_score': float(f1)
}

with open('model/config.json', 'w') as f:
    json.dump(config, f, indent=2)

metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    'test_samples': len(y_test),
    'train_samples': len(y_train)
}

with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Training complete!")
print("Files saved in 'model/' directory:")
print("  - model.pt")
print("  - vectorizer.pkl")
print("  - config.json")
print("  - metrics.json")