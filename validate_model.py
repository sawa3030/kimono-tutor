import torch
import torch.nn as nn
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
val_dataset = ImageFolder(root=os.path.expanduser('~/Desktop/image-recognition-demo/dataset/val'), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# デバッグのためにクラスとサンプル数を表示
print(f'Validation classes: {val_dataset.classes}, number of samples: {len(val_dataset)}')

# 簡単なモデルの定義（ResNet18の利用）
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))  # クラス数に合わせる

# モデルのロード
model.load_state_dict(torch.load('models/pet_bottle_model.pth'))

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 検証関数
def val_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            print(f'Validation Labels: {labels.cpu().numpy()}, Predictions: {preds.cpu().numpy()}')

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return epoch_loss, accuracy

# 検証の実行
val_loss, val_accuracy = val_model(model, val_loader, criterion, device)

print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

