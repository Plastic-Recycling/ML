import os
import json
import cv2
import numpy as np
from tqdm import tqdm
#
# # YOLO 모델 로드
# weights_path = "C:\\Users\\user\\Plastics\\yolov3.weights"
# cfg_path = "C:\\Users\\user\\Plastics\\yolov3.cfg"
# net = cv2.dnn.readNet(weights_path, cfg_path)

# 클래스 정의
classes = ["PP", "PE", "PS", "Unknown"]

# 이미지와 JSON 파일이 있는 디렉토리 경로
data_dir = "C:/Users/user/data/train"

# 분리된 객체를 저장할 디렉토리 경로
output_dir = "C:/Users/user/data/train_objects"
os.makedirs(output_dir, exist_ok=True)

def remove_background(image):
    # GrabCut을 위한 마스크 초기화
    mask = np.zeros(image.shape[:2], np.uint8)

    # GrabCut에 사용할 임시 배열들
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # 관심 영역 설정 (전체 이미지를 관심 영역으로 설정)
    rect = (1, 1, image.shape[1]-2, image.shape[0]-2)

    # GrabCut 알고리즘 실행
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 마스크 생성
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    # 원본 이미지에 마스크 적용
    result = image * mask2[:,:,np.newaxis]

    # 그레이스케일 변환 및 이진화
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 최대 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [max_contour], 0, (255), -1)

        # 원본 이미지에 최종 마스크 적용
        result = cv2.bitwise_and(image, image, mask=mask)

    return result

# 이미지 처리 함수
def process_image(filename):
    image_path = os.path.join(data_dir, filename)
    json_path = os.path.splitext(image_path)[0] + ".json"

    # JSON 파일 확인
    if not os.path.exists(json_path):
        print(f"경고: {json_path} 파일이 존재하지 않습니다. 해당 이미지는 처리하지 않습니다.")
        return

    # 이미지가 이미 처리되었는지 확인
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")
    if os.path.exists(output_path):
        print(f"{filename} 이미 처리된 이미지입니다. 건너뜁니다.")
        return

    # JSON 파일 읽기
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"JSON 파일을 읽는 중 오류 발생: {str(e)}")
        return

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    height, width = image.shape[:2]

    # JSON 파일의 레이블 정보 사용
    for shape in json_data["shapes"]:
        label = shape["label"]
        if label not in classes[:3]:  # PP, PE, PS만 체크
            label = "Unknown"
        points = np.array(shape["points"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)

        # 객체 이미지 추출
        obj_img = image[y:y+h, x:x+w]

        # 배경 제거
        try:
            obj_img = remove_background(obj_img)
        except Exception as e:
            print(f"배경 제거 중 오류 발생: {str(e)}")
            continue

        # 결과 저장
        output_class_dir = os.path.join(output_dir, label)
        os.makedirs(output_class_dir, exist_ok=True)
        output_path = os.path.join(output_class_dir, f"{os.path.splitext(filename)[0]}_{label}.jpg")
        cv2.imwrite(output_path, obj_img)

# 모든 이미지 처리
image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
for filename in tqdm(image_files, desc="Processing images"):
    process_image(filename)

print("모든 이미지 처리 완료")
