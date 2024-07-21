from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import cv2
from io import BytesIO
import os

app = Flask(__name__)

# 결과 저장 폴더 생성
RESULT_FOLDER = r'C:\Users\user\plastic_ML\ML\Transfer_learning\result'
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
model = EfficientNetModel(num_classes=3)  # PP, PE, PS 3개 클래스

model_path = r'C:\Users\user\plastic_ML\ML\Transfer_learning\train_final.pth'

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Model loaded successfully")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

WEIGHT_PER_PIXEL = {
    'PP': 0.00002,
    'PE': 0.00003,
    'PS': 0.000016
}

def process_single_image(file):
    file_contents = file.read()
    image = Image.open(BytesIO(file_contents))
    original_width, original_height = image.size

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    class_names = ['PP', 'PE', 'PS']
    predicted_class = class_names[class_index]

    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        pixel_area = cv2.countNonZero(mask)

        x, y, w, h = cv2.boundingRect(largest_contour)

        estimated_weight = pixel_area * WEIGHT_PER_PIXEL[predicted_class]

        # 결과 이미지 생성
        draw = ImageDraw.Draw(image)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # 폰트 설정 (크기를 더 크게 설정)
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
            "estimatedWeight": estimated_weight
        }

        # JSON 파일 저장
        json_path = os.path.join(RESULT_FOLDER, f'{file.filename.split(".")[0]}_result.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result
    else:
        return {'error': 'No object detected'}

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
    app.run(debug=True, port=5000)