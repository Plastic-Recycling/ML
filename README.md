# 처리 과정

## 1. 전처리 과정
이미지에서 객체를 분리하고 배경을 제거하여 객체 인식 모델 학습을 위한 데이터셋을 준비하는 과정.

- **객체 추출**: JSON파일의 polygon 좌표 정보를 이용하여 이미지에서 객체 영역 추출
- **배경 제거**: GrabCut 알고리즘 사용
- **분류 및 저장**: 객체를 폴더로 레이블링(PP, PE, PS, Unknown)
- **장점 및 의의**: 객체 분리를 통한 데이터 품질 향상, 배경 제거로 모델 학습 효율성 증대

## 2. 이미지 처리
- **RGB 변환**
- **크기 조정**: 224x224
- **정규화**: 
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]

## 3. 객체 탐지
- **그레이스케일 변환**
- **가우시안 블러 적용**
- **Canny 엣지 검출**
- **윤곽선 추출**
- **면적 기준 객체 선별**: 전체 이미지의 10-90%

# 하이퍼파라미터 설정
- **이미지 크기**:
  - ResNet: 224x224
  - EfficientNet: 300x300
- **학습률 (Learning Rate)**:
  - ResNet: 0.001
  - EfficientNet: 0.0001
- **배치 크기 (Batch Size)**: 32
- **에폭 수 (Number of Epochs)**: 최대 100 (조기 종료 사용)
- **옵티마이저**: Adam 옵티마이저
- **드롭아웃 비율**:
  - EfficientNet: 0.5 (과적합 방지)
- **학습률 스케줄러 파라미터**:
  - mode: 'min'
  - factor: 0.2 또는 0.5
  - patience: 3 또는 5
  - min_lr: 1e-6
- **데이터 증강 파라미터**:
  - RandomRotation: 20-40도
  - RandomResizedCrop
  - ColorJitter: 밝기, 대비, 채도, 색조 각각 20% 변화

### ML 모델별 세부 사항

## ResNet18
* 세부 모델 기용 - 각 모델을 학습시킨 후, 테스트 정확도를 측정하고 비교할 수 있음. 이를 통해 각 접근 방식의 성능을 평가 가능

데이터 로딩 과정에서 예상치 못한 오류로 중단되는 오류로 인해 배치크기를 64에서 32로 줄여 메모리 사용량 절감
problematic_files 집합을 추가하여 문제가 있는 파일들을 추적
문제가 발생했던 파일들의 목록에 대한 로그 구축
문제가 있는 파일을 건너뛰고 계속 진행하도록 수정, 문제가 있는 파일은 problematic_files 집합에 추가되고 로그에 기록.
자동으로 복구하고 파인튜닝을 계속할 수 있도록 세 코드를 모두 수정.
checkpoint model 저장 기능 추가.

가중치까지 포함한 모델 구축
matplotlib을 사용하여 정확도와 손실에 대한 학습 곡선 그래프 생성 및 표시
데이터 증강 기법과 정규화, 하이퍼파라미터 상시 조정


## EfficientNet-B3
* EfficientNet-B0 사용
* 세부 모델 기용 - 각 모델을 학습시킨 후, 테스트 정확도를 측정하고 비교할 수 있음. 이를 통해 각 접근 방식의 성능을 평가 가능
* GPU 인식 문제로 tensorflow --> pytorch 변경, EfficientNet-B3모델로 변경

데이터 로딩 과정에서 예상치 못한 오류로 중단되는 오류로 인해 배치크기를 64에서 32로 줄여 메모리 사용량 절감
problematic_files 집합을 추가하여 문제가 있는 파일들을 추적
문제가 발생했던 파일들의 목록에 대한 로그 구축
문제가 있는 파일을 건너뛰고 계속 진행하도록 수정, 문제가 있는 파일은 problematic_files 집합에 추가되고 로그에 기록.
자동으로 복구하고 파인튜닝을 계속할 수 있도록 세 코드를 모두 수정.
checkpoint model 저장 기능 추가.

가중치까지 포함한 모델 구축
matplotlib을 사용하여 정확도와 손실에 대한 학습 곡선 그래프 생성 및 표시
데이터 증강 기법과 정규화, 하이퍼파라미터 상시 조정


## EfficientNet-B3(가중치 추가)
* 가중치 적용 : 클래스 불균형 문제를 해결하기 위해 클래스 가중치를 적용.
    * 손실 함수 계산 시 데이터 개수가 적은 클래스에 큰 가중치를 줘서 오류가 더 큰 영향을 미치게 만듦.
* 소수 클래스에 대한 예측 성능을 향상 시킬 수 있지만, 전체적인 정확도는 약간 낮아질 수 있음
class_weight로 클래스 불균형 해소 모델 구축
* criterion = nn.CrossEntropyLoss(weight=class_weights)부분에서 가중치 정의, (손실 함수 정의 부분에서 이 가중치 적용)
* calculate_class_weights 함수에서 각 클래스의 가중치를 계산.
  학습 데이터에서 각 클래스의 빈도수를 바탕으로 클래스 가중치를 계산함.
* compute_class_weight 함수가 각 클래스의 가중치를 계산.
  balanced' 모드를 사용하여 클래스의 빈도수를 기반으로 가중치를 계산.
* 클래스 불균형 문제를 해결하기 위해 가중치 계산 > 각 클래스의 빈도에 반비례하는 가중치를 부여하여
  드물게 나타나는 클래스에 더 큰 가중치를, 자주 나타나는 클래스에 더 작은 가중치를 줌.
  손실 함수를 계산할 때 드물게 나타나는 클래스의 오류가 더 큰 영향을 미치게 되어 모델이 모든 클래스를 균형 있게 학습할 수 있음.
* 하지만 PP PE PS 클래스가 각각 PP-6,759장 / PE-6,913장 / PS-6,029장으로 가중치 계산 및 적용이 반드시 필요한 것인 지 확인요망.


* 세부 모델 기용 - 각 모델을 학습시킨 후, 테스트 정확도를 측정하고 비교할 수 있음. 이를 통해 각 접근 방식의 성능을 평가 가능
* EfficientNet-B0 사용
* GPU 인식 문제로 tensorflow --> pytorch 변경, EfficientNet-B3모델로 변경

데이터 로딩 과정에서 예상치 못한 오류로 중단되는 오류로 인해 배치크기를 64에서 32로 줄여 메모리 사용량 절감
problematic_files 집합을 추가하여 문제가 있는 파일들을 추적
문제가 발생했던 파일들의 목록에 대한 로그 구축
문제가 있는 파일을 건너뛰고 계속 진행하도록 수정, 문제가 있는 파일은 problematic_files 집합에 추가되고 로그에 기록.
자동으로 복구하고 파인튜닝을 계속할 수 있도록 세 코드를 모두 수정.
checkpoint model 저장 기능 추가.

가중치까지 포함한 모델 구축
matplotlib을 사용하여 정확도와 손실에 대한 학습 곡선 그래프 생성 및 표시
데이터 증강 기법과 정규화, 하이퍼파라미터 상시 조정


### ML 모델별 학습 결과 로그

# ResNet

* 초기 데이터 셋 (1000장)
    * 훈련 정확도 / 검증 정확도 모두 약 95% 이상
* 1차 시기 (21000장)
    * 최종 훈련 정확도 : 0.7876437902450562
    * 최종 검증 정확도 : 0.6938516497612
* 2차 시기 (21000장) - 완료
    * 최종 훈련 정확도 : 0.9788732394366196
    * 최종 검증 정확도 : 0.9497589444303477
(Train Loss: 0.0569 Train Acc: 0.9789 Val Loss: 0.1742 Val Acc: 0.9498)

# EfficientNet-B3

* 초기 데이터 셋 (1000장)
    * 훈련 정확도 / 검증 정확도 모두 약 99%
* 1차 시기 (21000장)
    * epoch 약 20/100 진행 중 GPU 인식 문제로 인해 tensorflow-->pytorch 변경
* 2차 시기 (21000장)
    * 최종 훈련 정확도 : 0.8713
    * 최종 검증 정확도 : 0.8785
(Train loss: 0.3169, Train acc: 0.8713, Val loss: 0.2945, Val acc: 0.8785)

* 3차 시기 (21000장) - 완료
    * 최종 훈련 정확도 : 0.9498
    * 최종 검증 정확도 : 0.9477
(Train loss: 0.1302, Train acc: 0.9498, Val loss: 0.1602, Val acc: 0.9477)

# # EfficientNet-B3(가중치 추가)

* 1차 시기 (21000장)
    * epoch 약 23/100 진행 중 GPU 인식 문제로 인해 tensorflow-->pytorch 변경

* 2차 시기 (21000장) - 완료
    * 최종 훈련 정확도 : 0.9799
    * 최종 검증 정확도 : 0.9538
(Train loss: 0.0560, Train acc: 0.9799, Val loss: 0.1580, Val acc: 0.9538)


# flask 서빙

이미지 내의 플라스틱 폐기물을 감지하고 종류를 분류하며, 예상 무게를 계산하는 Flask 서버 구현.

## 주요 기능

- 딥러닝 모델을 통한 플라스틱 종류 분류 (PP, PE, PS, Unknown)
- 객체 탐지 및 경계 상자 생성
- 픽셀 영역 기반 무게 추정
- 결과 이미지 및 JSON 생성
- 단일/다중 이미지 처리 지원

## 분류 및 무게 추정
- **EfficientNet B3 모델을 통한 분류**
- **픽셀당 무게 환산**:
  - PP: 0.00005g/pixel
  - PE: 0.00023g/pixel
  - PS: 0.00002g/pixel

## 결과 시각화
- 원본 이미지에 바운딩 박스 추가
- 분류 결과 및 예상 무게 텍스트 오버레이
- 결과 이미지 저장

# 출력 파일
- **이미지 파일**: `filename_result.jpg`
  - 바운딩 박스와 결과 텍스트가 포함된 이미지
- **JSON 파일**: `filename_result.json`
  - 라벨링 정보, 좌표, 예상 무게 등 메타데이터

## 기술 스택

- **Framework**: Flask
- **Deep Learning**: PyTorch, Resnet18, EfficientNet B3
- **Image Processing**: OpenCV, PIL
- **Data Format**: JSON, Base64


## API 엔드포인트 (예시)

### POST /predict
이미지를 받아 플라스틱 종류를 분류하고 무게를 추정

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - 단일 파일: `file`
  - 다중 파일: `files[]`

**Response**:
```json
{
    "version": "4.2.7",
    "flags": {},
    "shapes": [
        {
            "label": "PE",
            "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "group_id": null,
            "shape_type": "polygon",
            "inner": null,
            "flags": {}
        }
    ],
    "imagePath": "image_name.jpg",
    "imageHeight": 1080,
    "imageWidth": 1920,
    "estimatedWeight": 0.05,
    "processedImage": "base64_encoded_image_data"
}
