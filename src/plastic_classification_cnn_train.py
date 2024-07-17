import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json

# 시드 설정
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 로깅 설정
logging.basicConfig(filename='cnn_training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

# 데이터 준비
train_data_dir = "C:/Users/user/data/train_objects"
img_width, img_height = 224, 224
batch_size = 32

# 데이터 증강 설정
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(img_width, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(train_data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 생성
num_classes = len(dataset.classes)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)

# 학습 및 검증 함수
def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    return epoch_loss, epoch_acc

def validate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    return epoch_loss, epoch_acc

# 학습 곡선 그리기 함수
def plot_learning_curves(history, epochs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_acc'], label='Training Accuracy')
    plt.plot(range(1, epochs+1), history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Training Loss')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f'learning_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

# 학습 함수
def train_model(num_epochs=100, early_stopping_patience=8):
    best_val_acc = 0.0
    early_stopping_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_acc = validate(model, criterion, val_loader, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())

        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        # 실시간 그래프 업데이트
        plot_learning_curves(history, epoch+1)

        # 체크포인트 저장
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)

        # 성능이 낮은 체크포인트 삭제
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            best_model_path = checkpoint_path
        else:
            early_stopping_counter += 1
            os.remove(checkpoint_path)

        # Early Stopping
        if early_stopping_counter >= early_stopping_patience:
            logging.info("Early stopping")
            print("Early stopping")
            break

    # 최종 모델 저장
    torch.save(model.state_dict(), 'final_model.pth')

    # 학습 결과 출력 및 저장
    final_train_acc = float(history['train_acc'][-1])
    final_val_acc = float(history['val_acc'][-1])
    logging.info(f"최종 훈련 정확도: {final_train_acc}")
    logging.info(f"최종 검증 정확도: {final_val_acc}")

    with open('cnn_training_history.json', 'w') as f:
        json.dump(history, f)

    plot_learning_curves(history, len(history['train_loss']))

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # 프로그램 종료 시 리소스 해제
        torch.cuda.empty_cache()
