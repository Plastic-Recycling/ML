from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import cv2
from io import BytesIO
import os
import base64

app = Flask(__name__)

# 결과 저장 폴더 생성
RESULT_FOLDER = r'C:\Users\user\plastic_ML\ML\Transfer_learning_weight\result'
os.makedirs(RESULT_FOLDER, exist_ok=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetModel(num_classes=4)  # PP, PE, PS, unknown 4개 클래스

model_path = r'C:\Users\user\plastic_ML\ML\Transfer_learning_weight\weight.pth'

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Model loaded successfully")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

WEIGHT_PER_PIXEL = {
    'PP': 0.00005,
    'PE': 0.00003,
    'PS': 0.000016
}

def detect_object(image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 이미지 전처리
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 엣지 검출
    edges = cv2.Canny(blurred, 1, 10)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 면적이 큰 순서대로 윤곽선 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 이미지 전체 면적
    total_area = img_np.shape[0] * img_np.shape[1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 객체 면적이 전체 이미지의 10% 이상 90% 이하인 경우에만 객체로 간주
        if 0.1 * total_area < area < 0.9 * total_area:
            return x, y, w, h

    # 적절한 객체를 찾지 못한 경우 전체 이미지 반환
    return 0, 0, img_np.shape[1], img_np.shape[0]

def process_single_image(file):
    file_contents = file.read()
    image = Image.open(BytesIO(file_contents))
    original_width, original_height = image.size

    # 객체 탐지
    x, y, w, h = detect_object(image)

    # 객체 영역 추출
    object_image = image.crop((x, y, x+w, y+h))

    # 모델 예측
    input_tensor = preprocess_image(object_image)
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    class_names = ['PP', 'PE', 'PS']
    predicted_class = class_names[class_index]

    # 픽셀 영역 계산
    pixel_area = w * h

    estimated_weight = pixel_area * WEIGHT_PER_PIXEL[predicted_class]

    # 결과 이미지 생성
    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # 폰트 설정
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # 텍스트 추가
    text = f"{predicted_class}: {estimated_weight:.2f}g"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 텍스트 위치 조정 (이미지 상단에 배치)
    text_position = (10, 10)

    # 텍스트 배경 추가
    draw.rectangle([text_position[0], text_position[1],
                    text_position[0] + text_width, text_position[1] + text_height],
                   fill="white")

    # 텍스트 그리기
    draw.text(text_position, text, font=font, fill="red")

    # 결과 이미지 저장
    result_image_path = os.path.join(RESULT_FOLDER, f'{file.filename.split(".")[0]}_result.jpg')
    image.save(result_image_path)

    # 이미지를 바이트로 변환
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    result = {
        "version": "4.2.7",
        "flags": {},
        "shapes": [
            {
                "label": predicted_class,
                "points": [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ],
                "group_id": None,
                "shape_type": "polygon",
                "inner": None,
                "flags": {}
            }
        ],
        "imagePath": file.filename,
        "imageHeight": original_height,
        "imageWidth": original_width,
        "estimatedWeight": estimated_weight,
        "processedImage": img_str
    }

    # JSON 파일 저장
    json_path = os.path.join(RESULT_FOLDER, f'{file.filename.split(".")[0]}_result.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        return jsonify(process_single_image(file))

    elif 'files[]' in request.files:
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected files'}), 400

        results = []
        for file in files:
            result = process_single_image(file)
            results.append(result)

        return jsonify(results)

    else:
        return jsonify({'error': 'No file part'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)