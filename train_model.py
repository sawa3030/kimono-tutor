import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# データセットのロード
train_dataset = ImageFolder(root=os.path.expanduser('~/Desktop/image-recognition-demo/dataset/train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# デバッグのためにクラスとサンプル数を表示
print(f'Training classes: {train_dataset.classes}, number of samples: {len(train_dataset)}')

# 簡単なモデルの定義（ResNet18の利用）
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # クラス数に合わせる

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニング関数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# トレーニングの実行
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')

# モデルの保存
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/simple_model.pth')

print("モデルのトレーニングが完了し、保存されました")