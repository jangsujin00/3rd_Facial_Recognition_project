import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

### 카메라로 찍은 사진을 학습

# (1번과 cv 동일)

# numpy(numerical python) : Python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수(Linear algebra) 라이브러리

# os모듈 : os에 의존하는 다양한 기능을 제공하는 모듈.
#
# 1) 파일이나 디렉토리 조작이 가능하고, 2) 파일의 목록이나 path를 얻을 수 있거나, 3) 새로운 파일 혹은 디렉토리를 작성하는 것도 가능.
#
# - os.listdir() : 파일이나 디렉토리의 목록을 확인하기 위해 사용.
# - os.path()모듈 메소드
#
# ㄴisfile() : 파일의 존재여부를 확인. return값은 Bool형.
# ㄴjoin() : 경로와 파일명 결합
# ㄴdir( ) [directory] : 해당 객체가 어떤 변수와 메소드(method)를 가지고 있는지 **나열**



data_path = 'faces/'
#faces폴더에 있는 파일 리스트 얻기
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]    # isfile(경로)
#데이터와 매칭될 라벨 변수
Training_Data, Labels = [], []

#파일 개수 만큼 루프
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    # 이미지 불러오기
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Training_Data 리스트에 이미지를 바이트 배열로 추가
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # Labels 리스트엔 카운트 번호 추가
    Labels.append(i)

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)
#모델 생성
model = cv2.face.LBPHFaceRecognizer_create()        # LBP[Local-Binary-Pattern] 알고리즘 : 주변 값을 2진수로 표현한 뒤, 값을 계산 (영상의 질감을 256개의 숫자로 표현)
#학습 시작
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")