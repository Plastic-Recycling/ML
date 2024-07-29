import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import logging
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
print(torch.cuda.is_available())
print(torch.__version__)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# YOLOv5 경로 추가
yolov5_path = Path("C:/Plastics_ML/yolov5")
sys.path.append(str(yolov5_path))
os.chdir(yolov5_path)

from yolov5 import train
from yolov5.models.yolo import Model
from yolov5.utils.dataloaders import LoadImages, create_dataloader
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords,
                           LOGGER, colorstr, check_dataset)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.metrics import box_iou
from yolov5.val import run as validate

# plots.py 파일 수정
plots_file = os.path.join(yolov5_path, 'utils', 'plots.py')
with open(plots_file, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "w, h = self.font.getsize(label)",
    "try:\n                w, h = self.font.getsize(label)\n            except AttributeError:\n                w, h = self.font.getbbox(label)[2:]"
)

with open(plots_file, 'w', encoding='utf-8') as f:
    f.write(content)

# 로깅 설정
log_file = Path('C:/Plastics_ML/finetune_log.txt')
logging.basicConfig(filename=str(log_file), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def 모델_로드(weights_path, device):
    """사전 학습된 모델을 로드합니다."""
    model = torch.load(weights_path, map_location=device)['model'].float().fuse().eval()
    return model

def 데이터셋_준비(data_path):
    """파인튜닝을 위한 전체 데이터셋을 준비합니다."""
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    
    train_path = data['train']
    val_path = data['val']
    nc = data['nc']
    names = data['names']
    
    return train_path, val_path, nc, names

def 학습곡선_저장(train_losses, val_losses, mAP50s, mAP50_95s, save_path):
    """학습 곡선을 그리고 저장합니다."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(2, 1, 2)
    plt.plot(mAP50s, label='mAP@0.5')
    plt.plot(mAP50_95s, label='mAP@0.5:0.95')
    plt.legend()
    plt.title('mAP Curves')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_loss(pred, targets, model):
    """손실을 계산합니다."""
    loss = model.loss(pred, targets)
    return loss, loss.item()

def 파인튜닝(model, train_path, val_path, nc, names, epochs=150, batch_size=8, img_size=512, patience=30):
    """전체 데이터셋에 대해 모델을 파인튜닝합니다."""
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    
    # 학습을 위한 모델 설정
    model.train()
    model.to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 데이터로더
    train_loader, _ = create_dataloader(train_path, img_size, batch_size, stride=32, 
                                        hyp=None, augment=True, cache=False, pad=0.5, 
                                        rect=False, workers=4)
    
    # Early stopping 설정
    best_mAP50 = 0
    best_mAP50_95 = 0
    patience_counter = 0
    
    # 학습 곡선 데이터
    train_losses = []
    val_losses = []
    mAP50s = []
    mAP50_95s = []
    
    # 결과 저장 디렉토리
    output_dir = Path(f'runs/finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 루프
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        
        for batch_i, (imgs, targets, _, _) in enumerate(tqdm(train_loader, desc=colorstr('green', 'train'))):
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # 순전파
            pred = model(imgs)
            
            # 손실 계산
            loss, loss_item = compute_loss(pred, targets, model)
            
            # 역전파
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss_item
            
            # 로깅
            if batch_i % 100 == 0:
                logging.info(f"Batch {batch_i}: loss = {loss_item}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        logging.info(f"Epoch {epoch+1} average loss: {avg_loss}")
        
        # 검증
        model.eval()
        results = validate(model=model, dataloader=val_path, device=device)
        mAP50, mAP50_95 = results[2], results[3]  # mAP@0.5와 mAP@0.5:0.95
        val_losses.append(results[0])  # validation loss
        mAP50s.append(mAP50)
        mAP50_95s.append(mAP50_95)
        
        logging.info(f"Epoch {epoch+1}: mAP@0.5 = {mAP50:.4f}, mAP@0.5:0.95 = {mAP50_95:.4f}")
        
        # 최고 성능 모델 저장
        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            torch.save(model.state_dict(), output_dir / f'best_model_mAP50_{mAP50:.4f}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if mAP50_95 > best_mAP50_95:
            best_mAP50_95 = mAP50_95
            torch.save(model.state_dict(), output_dir / f'best_model_mAP50_95_{mAP50_95:.4f}.pt')
        
        # 학습 곡선 저장
        학습곡선_저장(train_losses, val_losses, mAP50s, mAP50_95s, output_dir / f'learning_curves_epoch_{epoch+1}.png')
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 최종 모델 저장
    torch.save(model.state_dict(), output_dir / 'last_model.pt')
    
    return model, best_mAP50, best_mAP50_95

if __name__ == "__main__":
    try:
        # 사전 학습된 모델 로드
        weights_path = "path/to/pretrained_model.pt"
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        model = 모델_로드(weights_path, device)
        
        # 데이터셋 준비
        data_path = "path/to/data.yaml"
        train_path, val_path, nc, names = 데이터셋_준비(data_path)
        
        # 파인튜닝 파라미터 설정
        epochs = 150
        batch_size = 8
        img_size = 512
        patience = 30
        
        # 파인튜닝
        model, best_mAP50, best_mAP50_95 = 파인튜닝(model, train_path, val_path, nc, names,
                                                    epochs=epochs, batch_size=batch_size,
                                                    img_size=img_size, patience=patience)
        
        logging.info(f"파인튜닝 완료. 최고 성능: mAP@0.5 = {best_mAP50:.4f}, mAP@0.5:0.95 = {best_mAP50_95:.4f}")
        
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}", exc_info=True)