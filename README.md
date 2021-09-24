# Rice-Images-for-Object-Detection
### ㅤ
## 1.Dataset
### 1-1. Image Labeling
![image](https://user-images.githubusercontent.com/80669371/134620740-c08ac457-782b-4373-8112-4b8d5658ddbb.png)
#### labelImg를 사용하여 각각 이미지의 라벨링 정보를 xml 형태로 저장 : Pascal VOC 형식
#### ㅤ
### 1-2. Data Sample
![image](https://user-images.githubusercontent.com/80669371/134619825-6507cd10-ff7f-4440-8203-150584c3937e.png) ![image](https://user-images.githubusercontent.com/80669371/134619832-44c5affd-8da4-44cc-a0d4-4ef10d72d84f.png)
#### ●Train dataset : 32
#### ●Validation dataset : 8
#### ●Test data set : 10
### ㅤ
## 2.Faster R-CNN Model Architecture
#### :CNN 특징 추출 및 Classification, Bounding Box Regression을 모두 하나의 모델에서 학습시키고자 한 모델
![image](https://user-images.githubusercontent.com/80669371/134619982-1bafbf32-26e0-4898-ac6f-63b4d6819f7e.png)
#### ㅤ
### 2-1.Network Architecture
![image](https://user-images.githubusercontent.com/80669371/134620132-42bdf204-8a90-4bfa-9c3f-38515f4801cc.png)
##### 1) 이미지 입력
##### 2) pre-train CNN 모델의 Feature map 추출
##### 3) 전체 이미지의 feature map을 통해 Region Proposal Network를 실행시켜 후보군 BoundingBox좌표를 예측
##### 4) RPN에서 나온 BB의 좌표에 따라, feature map의 ROI를 잘라내어 ROI pooling
##### 5) 이를 Image classification과 BoundingBox regression에 각각 넣어주어 최종 판단
#### ㅤ
#### 2-1-1.RPN(Region Proposal Network)
###### :object 경계와 각 위치에서의 사물성의 스코어를 동시에 예측하는 fully conv 네트워크
###### ->질 좋은 region proposal을 생성하기 위해 end-to-end로 학습
![image](https://user-images.githubusercontent.com/80669371/134620215-7643578a-16c0-4526-a2bb-37127d99ad5d.png)
#### ㅤ
#### 2-1-2.RoI Pooling
#####  :RoI 영역에 해당하는 부분만 max-pooling을 통해 feature map으로부터 고정된 길이의 저차원 벡터로 축소하는 단계를 의미
![image](https://user-images.githubusercontent.com/80669371/134620244-95b3cb41-0623-4e26-9d9e-9ec673fb9a7f.png)
### ㅤ
## 3.Model Training
![image](https://user-images.githubusercontent.com/80669371/134620886-81abeb80-fd72-41fe-ac6a-7c79a3031262.png)
### ㅤ
## 4.Model Validation
#### : :IoU값 설정에 따른 모델 검증(IoU값이 특정 값 이상일 때 해당하는 바운딩 박스만 Ouput에 표시)
![image](https://user-images.githubusercontent.com/80669371/134620955-2bd2c76b-810b-4d65-847f-fd645f9e16d9.png)
### ㅤ
## 5.Model Evaluation
#### :Test dataset을 이용하여 학습한 모델로 데이터 평가
![image](https://user-images.githubusercontent.com/80669371/134621119-33ca9dd1-bcf4-473f-b5b3-d17f0e2a064a.png)
