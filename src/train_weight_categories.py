import os
import sys
import logging
import yaml
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, KFold

# Albumentations 라이브러리 추가
import albumentations as A
from albumentations.pytorch import ToTensorV2

# YOLOv5 경로 설정
yolov5_path = Path("C:/Plastics_ML/yolov5")
sys.path.append(str(yolov5_path))

# YOLOv5 모듈 임포트
from yolov5 import train
from yolov5.utils.general import (check_img_size, non_max_suppression, xyxy2xywh,
                           check_dataset, check_yaml, check_file, colorstr)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_results
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import box_iou

def convert_json_to_yolo(json_file, plastic_type, weight, class_mapping):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    yolo_annotations = []
    for shape in data['shapes']:
        label = shape['label']
        if label != plastic_type:
            continue  # JSON의 라벨이 플라스틱 타입과 일치하지 않으면 건너뜁니다.
        
        class_name = f"{plastic_type}_{weight}g"
        if class_name not in class_mapping:
            continue
        
        class_id = class_mapping[class_name]
        points = shape['points']
        
        # 폴리곤을 바운딩 박스로 변환
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # YOLO 형식으로 변환 (중심 x, 중심 y, 너비, 높이)
        x_center = (x_min + x_max) / (2 * img_width)
        y_center = (y_min + y_max) / (2 * img_height)
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


# 로깅 설정
log_file = Path('C:/Plastics_ML/detailed_training_log.txt')
logging.basicConfig(filename=str(log_file), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# PyTorch 버전 및 디바이스 정보 출력
device = select_device('0' if torch.cuda.is_available() else 'cpu')
logging.info(f"PyTorch 버전: {torch.__version__}")
logging.info(f"사용 중인 디바이스: {device}")

def convert_to_yolo_format(input_path, output_path):
    logging.info("YOLO 포맷으로 데이터 변환 시작")
    classes = ['PP_8g', 'PP_10g', 'PP_12g', 'PP_14g', 
               'PE_10g', 'PE_20g', 'PE_30g', 'PE_40g', 
               'PS_11g', 'PS_13g', 'PS_15g', 'PS_17g']
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'labels').mkdir(exist_ok=True)
    
    for plastic_type in ['PP', 'PE', 'PS']:
        plastic_path = Path(input_path) / plastic_type
        for weight_folder in plastic_path.iterdir():
            class_name = f"{plastic_type}_{weight_folder.name}g"
            class_id = classes.index(class_name)
            
            for json_file in weight_folder.glob('*.json'):
                logging.debug(f"처리 중인 JSON 파일: {json_file}")
                with open(json_file, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"빈 JSON 파일: {json_file}")
                        continue
                
                if not data.get('shapes'):
                    logging.warning(f"JSON 파일에 'shapes' 데이터 없음: {json_file}")
                    continue
                
                image_path = json_file.with_suffix('.jpg')
                if not image_path.exists():
                    logging.warning(f"이미지 파일 없음: {image_path}")
                    continue
                
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # 이미지 복사
                dest_image_path = output_path / 'images' / image_path.name
                shutil.copy(image_path, dest_image_path)
                logging.debug(f"이미지 복사됨: {dest_image_path}")
                
                # 라벨 파일 생성
                label_path = output_path / 'labels' / json_file.with_suffix('.txt').name
                with open(label_path, 'w') as f:
                    for shape in data['shapes']:
                        points = shape['points']
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x_center = sum(x_coords) / len(x_coords) / img_width
                        y_center = sum(y_coords) / len(y_coords) / img_height
                        width = (max(x_coords) - min(x_coords)) / img_width
                        height = (max(y_coords) - min(y_coords)) / img_height
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                logging.debug(f"라벨 파일 생성됨: {label_path}")
    
    logging.info("YOLO 포맷으로 데이터 변환 완료")

def create_yolo_config(data_path, num_classes):
    logging.info("YOLO 설정 파일 생성")
    config = {
        'path': str(data_path),
        'train': str(data_path / 'images' / 'train'),
        'val': str(data_path / 'images' / 'val'),
        'test': str(data_path / 'images' / 'test'),
        'nc': num_classes,
        'names': ['PP_8g', 'PP_10g', 'PP_12g', 'PP_14g', 
                  'PE_10g', 'PE_20g', 'PE_30g', 'PE_40g', 
                  'PS_11g', 'PS_13g', 'PS_15g', 'PS_17g']
    }
    
    config_path = data_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logging.info(f"YOLO 설정 파일 생성 완료: {config_path}")
    return str(config_path)

def find_latest_results_file(base_dir):
    base_dir = Path(base_dir)
    weight_categories_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('weight_categories')], 
                                    key=lambda x: int(x.name.split('weight_categories')[-1]) if x.name.split('weight_categories')[-1].isdigit() else 0,
                                    reverse=True)
    
    for dir in weight_categories_dirs:
        results_file = dir / 'results.csv'
        if results_file.exists():
            return results_file
    
    return None

def plot_learning_curves(results_file, output_dir):
    if not results_file.exists():
        logging.error(f"결과 파일을 찾을 수 없습니다: {results_file}")
        return

    try:
        results = np.loadtxt(results_file, skiprows=1, delimiter=',')
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax = ax.ravel()
        s = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
             'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
        for i, j in enumerate(s):
            ax[i].plot(results[:, 0], results[:, i + 1], label=j)
            ax[i].set_title(j)
            ax[i].set_xlabel('epoch')
            ax[i].legend()
        fig.tight_layout()
        plot_path = output_dir / 'learning_curves.png'
        fig.savefig(plot_path)
        plt.close(fig)
        logging.info(f"학습 곡선 저장됨: {plot_path}")
    except Exception as e:
        logging.error(f"학습 곡선 그리기 중 오류 발생: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
# 데이터 증강 함수 추가
def get_train_transforms():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def train_model(data_yaml, weights, epochs, batch_size, img_size, project_name, patience=10):
    logging.info("모델 학습 시작")
    
    output_dir = Path(f'C:/Plastics_ML/runs/train/{project_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 증강 설정
    train_transforms = get_train_transforms()
    
    # K-fold 교차 검증 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 데이터셋 로드
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    train_path = data['train']
    dataset = create_dataloader(train_path, img_size, batch_size, stride=32, hyp=None, augment=True, cache=False, pad=0.5, rect=False)[0].dataset
    
    best_mAP50 = 0
    best_model_path = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logging.info(f"Fold {fold+1} 시작")
        
        # YOLOv5 학습 설정
        args = dict(
            data=data_yaml,
            weights=weights,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,
            project=str(output_dir.parent),
            name=f"{output_dir.name}_fold{fold+1}",
            device='',
            workers=4,
            save_period=-1,  # 모든 epoch 저장 비활성화
            patience=patience
        )

        # 옵티마이저와 학습률 스케줄러 설정 (하이퍼파라미터 조정)
        args['optimizer'] = 'AdamW'
        args['lr0'] = 0.001
        args['lrf'] = 0.01
        args['weight_decay'] = 0.0005
        args['warmup_epochs'] = 3
        args['warmup_momentum'] = 0.8
        args['warmup_bias_lr'] = 0.1

        try:
            # YOLOv5 학습 실행
            trainer = train.run(**args)
            
            # 최고 성능 모델 저장
            if isinstance(trainer, dict) and 'best_fitness' in trainer:
                current_mAP50 = trainer['best_fitness']
            elif hasattr(trainer, 'best_fitness'):
                current_mAP50 = trainer.best_fitness
            else:
                logging.warning("최고 성능을 확인할 수 없습니다.")
                continue

            if current_mAP50 > best_mAP50:
                # 이전 최고 모델 삭제
                if best_model_path and best_model_path.exists():
                    best_model_path.unlink()
                
                # 새로운 최고 모델 저장
                best_mAP50 = current_mAP50
                best_model_path = output_dir / f'best_model_mAP50_{best_mAP50:.4f}.pt'
                
                if isinstance(trainer, dict) and 'model' in trainer:
                    # 임시 파일로 먼저 저장
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        torch.save(trainer['model'].state_dict(), tmp_file.name)
                        # 임시 파일을 원하는 위치로 이동
                        shutil.move(tmp_file.name, str(best_model_path))
                elif hasattr(trainer, 'best'):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        torch.save(trainer.best, tmp_file.name)
                        shutil.move(tmp_file.name, str(best_model_path))
                
                logging.info(f"새로운 최고 성능 모델 저장됨: {best_model_path}")
            else:
                logging.info(f"Fold {fold+1}에서 더 좋은 모델이 생성되지 않았습니다.")

        except Exception as e:
            logging.error(f"모델 학습 또는 저장 중 오류 발생: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
        
        logging.info(f"Fold {fold+1} 완료")
    
    logging.info("모델 학습 완료")
    
    # 결과 파일 찾기
    results_file = find_latest_results_file(output_dir.parent)
    if results_file:
        logging.info(f"결과 파일 찾음: {results_file}")
        logging.info(f"결과 파일 크기: {results_file.stat().st_size} bytes")
        plot_learning_curves(results_file, output_dir)
    else:
        logging.warning("결과 파일을 찾을 수 없습니다.")
    
    return trainer, results_file

def main():
    logging.info("무게 카테고리 학습 프로그램 시작")
    try:
        weight_data_path = Path('D:/data/train_weight')
        yolo_data_path = Path('D:/data/yolo_weight_data')
        
        class_mapping = {
            'PP_8g': 0, 'PP_10g': 1, 'PP_12g': 2, 'PP_14g': 3,
            'PE_10g': 4, 'PE_20g': 5, 'PE_30g': 6, 'PE_40g': 7,
            'PS_11g': 8, 'PS_13g': 9, 'PS_15g': 10, 'PS_17g': 11
        }
        
        # YOLO 형식으로 데이터 변환 및 이동
        for plastic_type in ['PE', 'PP', 'PS']:
            plastic_path = weight_data_path / plastic_type
            for weight_folder in plastic_path.iterdir():
                weight = weight_folder.name
                for img_file in weight_folder.glob('*.jpg'):
                    # 랜덤하게 train, val, test 세트로 나누기
                    split = np.random.choice(['train', 'val', 'test'], p=[0.7, 0.2, 0.1])
                    
                    # 이미지 파일 복사
                    dest_img_dir = yolo_data_path / 'images' / split
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(img_file, dest_img_dir / img_file.name)
                    
                    # JSON 파일 처리
                    json_file = img_file.with_suffix('.json')
                    if json_file.exists():
                        # JSON을 YOLO 형식으로 변환
                        yolo_annotations = convert_json_to_yolo(json_file, plastic_type, weight, class_mapping)
                        
                        # YOLO 형식의 라벨 파일 생성
                        if yolo_annotations:
                            dest_label_dir = yolo_data_path / 'labels' / split
                            dest_label_dir.mkdir(parents=True, exist_ok=True)
                            with open(dest_label_dir / img_file.with_suffix('.txt').name, 'w') as f:
                                f.write('\n'.join(yolo_annotations))
                        else:
                            logging.warning(f"유효한 주석 없음: {json_file}")
                    else:
                        logging.warning(f"라벨 파일 없음: {json_file}")
        
        data_yaml = create_yolo_config(yolo_data_path, num_classes=12)
        
        project_name = 'weight_categories'
        results, results_file = train_model(data_yaml, 
                                            weights='yolov5m.pt', 
                                            epochs=150, 
                                            batch_size=8,
                                            img_size=608,
                                            project_name=project_name,
                                            patience=10)
        
        if results_file and results_file.exists():
            logging.info(f"학습 결과 파일: {results_file}")
            logging.info(f"결과 파일 크기: {results_file.stat().st_size} bytes")
        else:
            logging.warning(f"학습 결과 파일을 찾을 수 없습니다.")
        
        logging.info("무게 카테고리 학습 프로그램 종료")
    except Exception as e:
        logging.error(f"학습 중 오류 발생: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()