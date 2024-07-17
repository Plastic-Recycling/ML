import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file
import numpy as np
from torchvision.models import efficientnet_b3
import torch.nn as nn

app = Flask(__name__)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet 모델 클래스 정의
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = efficientnet_b3(pretrained=False)
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

# EfficientNet 모델 로드
num_classes = 4  # PP, PE, PS, Others
efficientnet_model = EfficientNetModel(num_classes).to(device)
efficientnet_model.load_state_dict(torch.load('weight.pth', map_location=device))
efficientnet_model.eval()

# 클래스 레이블
class_labels = ['PP', 'PE', 'PS', 'Other']

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(file):
    # 이미지 저장
    filename = file.filename
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # EfficientNet을 이용한 이미지 분류
    image = Image.open(file_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = efficientnet_model(input_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]

    label = class_labels[predicted.item()]
    confidence = probability[predicted.item()].item()

    # 폴리곤 포인트 생성 (예: 이미지의 1/4 크기의 사각형)
    width, height = image.size
    points = [
        width//4, height//4,
        3*width//4, height//4,
        3*width//4, 3*height//4,
        width//4, 3*height//4
    ]

    # 분류 결과를 이미지에 표시
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # 폴리곤 그리기
    draw.polygon(points, outline=(255, 0, 0))

    # 레이블 표시
    text = f"{label}: {confidence:.2f}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_position = (points[0], points[1] - text_bbox[3] - 5)
    draw.rectangle([text_position[0], text_position[1], text_position[0] + text_bbox[2], text_position[1] + text_bbox[3]], fill=(255, 255, 255))
    draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 결과 이미지 저장
    result_path = os.path.join('results', f'result_{filename}')
    image.save(result_path)

    # JSON 파일 생성
    json_result = {
        "version": "4.2.7",
        "flags": {},
        "shapes": [{
            "label": label,
            "points": [[points[i], points[i+1]] for i in range(0, len(points), 2)],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }],
        "imagePath": filename,
        "imageHeight": height,
        "imageWidth": width
    }

    json_path = os.path.join('results', f'{os.path.splitext(filename)[0]}.json')
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=2)

    return {
        'filename': filename,
        'class': label,
        'confidence': confidence,
        'json_file': json_path,
        'result_image': result_path
    }

@app.route('/classify', methods=['POST'])
def classify_images():
    if 'file' not in request.files and 'files[]' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    results = []

    # 단일 파일 처리
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            results.append(process_image(file))

    # 다중 파일 처리
    if 'files[]' in request.files:
        files = request.files.getlist('files[]')
        for file in files:
            if file.filename != '':
                results.append(process_image(file))

    if not results:
        return jsonify({'error': 'No valid files uploaded'}), 400

    return jsonify(results)

@app.route('/results/<filename>')
def get_result(filename):
    return send_file(os.path.join('results', filename))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True)