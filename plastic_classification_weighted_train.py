import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 로깅 설정
logging.basicConfig(filename='efficientnet_training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# PyTorch 버전 출력
print(f"PyTorch 버전: {torch.__version__}")
logging.info(f"PyTorch 버전: {torch.__version__}")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")
logging.info(f"사용 중인 디바이스: {device}")

# 데이터 준비
train_data_dir = "C:/Users/user/data/train_objects"
img_width, img_height = 300, 300
batch_size = 32

# 데이터 증강 및 전처리
data_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 생성
def create_dataloaders():
    print("데이터 로딩 시작")
    logging.info("데이터 로딩 시작")
    try:
        dataset = ImageFolder(train_data_dir, transform=data_transforms)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"학습 데이터 크기: {len(train_dataset)}")
        print(f"검증 데이터 크기: {len(val_dataset)}")
        logging.info(f"학습 데이터 크기: {len(train_dataset)}")
        logging.info(f"검증 데이터 크기: {len(val_dataset)}")
        print("데이터 로딩 완료")
        logging.info("데이터 로딩 완료")
        return train_loader, val_loader, dataset.classes
    except Exception as e:
        print(f"데이터로더 생성 중 오류 발생: {str(e)}")
        logging.error(f"데이터로더 생성 중 오류 발생: {str(e)}")
        raise

# 모델 정의
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b3(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# 학습 곡선 그리기 함수
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f'efficientnet_learning_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

# 클래스 가중치 계산 함수
def calculate_class_weights(train_loader):
    print("클래스 가중치 계산 시작")
    logging.info("클래스 가중치 계산 시작")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    print(f"계산된 클래스 가중치: {class_weights}")
    logging.info(f"계산된 클래스 가중치: {class_weights}")
    print("클래스 가중치 계산 완료")
    logging.info("클래스 가중치 계산 완료")
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# 훈련 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    print("학습 시작")
    logging.info("학습 시작")
    writer = SummaryWriter(f"runs/efficientnet_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    best_val_acc = 0.0
    patience = 8
    counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        scheduler.step(val_epoch_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, "
              f"Val loss: {val_epoch_loss:.4f}, Val acc: {val_epoch_acc:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                     f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, "
                     f"Val loss: {val_epoch_loss:.4f}, Val acc: {val_epoch_acc:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            logging.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    writer.close()
    print("학습 완료")
    logging.info("학습 완료")
    return train_losses, val_losses, train_accs, val_accs

def main():
    print("프로그램 시작")
    logging.info("프로그램 시작")
    try:
        print("데이터로더 생성 중")
        logging.info("데이터로더 생성 중")
        train_loader, val_loader, classes = create_dataloaders()
        num_classes = len(classes)
        print(f"클래스 수: {num_classes}")
        logging.info(f"클래스 수: {num_classes}")

        print("모델 생성 중")
        logging.info("모델 생성 중")
        model = EfficientNetModel(num_classes).to(device)
        print("클래스 가중치 계산 중")
        logging.info("클래스 가중치 계산 중")
        class_weights = calculate_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        print("학습 시작")
        logging.info("학습 시작")
        train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer)

        # 최종 결과 저장
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        with open('training_results.json', 'w') as f:
            json.dump(results, f)

        plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

        print("EfficientNet 학습 과정 완료")
        logging.info("EfficientNet 학습 과정 완료")
    except Exception as e:
        print(f"전체 프로세스 중 오류 발생: {str(e)}")
        logging.error(f"전체 프로세스 중 오류 발생: {str(e)}")
    finally:
        print("프로그램 종료")
        logging.info("프로그램 종료")

if __name__ == "__main__":
    main()