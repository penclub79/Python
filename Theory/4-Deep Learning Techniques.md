# Deep-theory
# Transfer Learning & Fine-Tuning

## 1. 용어 정리
- Overfitting(과적합) : 학습 데이터를 과하게 학습하는 것
- Feature Extraction(특징추출) : 고차원의 원본 Feature 공간을 저차원의 새로운 Feature 공간으로 투영
- Backpropagation(역전파) : 다중 퍼셉트론 학습에 사용되는 통계적 기법
- Autoencoder(오토인코더) : Unsupervised Learning을 위한 Neural Network 구조 중 하나로, Hidden Layer에 Input 데이터를 압축적으로
                           저장함으로써 원본 데이터의 특징(Feature)을 효과적으로 추출할 수 있는 기법
 
## 2. Transfer Learning
### 2.1 정의
- 기존의 만들어진 모델을 사용하여 새로운 모델을 만들때 학습을 빠르게 하며, 예측을 더 높이는 방법
- 딥러닝을 특징 추출(Feature Extractor)로만 사용하고 그렇게 추출한 특징을 가지고 다른 모델을 학습하는 것
- 이미 학습된 weight들을 Transfer(전송)하여 자신의 model에 맞게 학습 시키는 방법

### 2.2 사용 목적
1) 실질적으로 Convolution Network를 처음부터 학습시키는 일은 많지 않고 대부분의 문제는 이미 학습된 모델을 사용해서 문제를 해결할 때
2) 복잡한 모델일수록 학습시키기 어려움
3) Layer의 개수, Activation, Hyper Parameters 등등 고려해야 할 사항들이 많아, 실질적으로 처음부터 학습시키려면 많은 시도가 필요하므로 사용

### 2.3 사용 경우
#### 1) 새로 훈련할 데이터가 적지만 Original 데이터와 유사할 경우
데이터의 양이 적어 Fine Tuning(전체 모델의 대해서 역전파(Backpropagation)를 진행하는 것)은 오버피팅의 위험이 있기에 하지 않고 새로 학습할
데이터는 Original 데이터와 유사하기 때문에 이 경우 최종 Linear Classifier 레이어만 학습한다.

#### 2) 새로 훈련할 데이터가 적으며 Original 데이터와 다른 경우
네트워크 초기 어딘가 Activation 이후에 특정 레이어를 학습

#### 이미 잘 훈련된 모델이 있고, 특히 해당 모델과 유사한 문제를 해결 시 사용

### 2.4 모델
#### 1) VGG
- 많은 하이퍼 파라미터를 사용하지 않기 때문에 단순한 아키텍처 모델
- 3 X 3 필터의 스트라이드 1인 Convolution Layer와 2 X 2의 스트라이드 2인 Pooling Layer 사용
![vgg](https://user-images.githubusercontent.com/40047360/44298241-6b54d900-a31a-11e8-9dc0-bf403bfb6415.png)

#### 2) GoogleNet
- 빨간색 동그라미 : 인셉션 모듈로 총 9개의 인셉션 모듈 적용
- 파란색 유닛 : Convolutional Layer
- 빨간색 유닛 : Max-Pooling Layer
- 노란색 유닛 : Softmax Layer
- 녹색 유닛 : 기타 function
- 동그라미 위에 있는 숫자는 각 단계에서 얻어지는 Feature-Map의 수
<img width="907" alt="googlenet2" src="https://user-images.githubusercontent.com/40047360/44298242-6d1e9c80-a31a-11e8-875e-3f2640242691.png">

#### 3) ResNet
![resnet](https://user-images.githubusercontent.com/40047360/44298245-6f80f680-a31a-11e8-9aee-ecac474a01ca.png)
- Activation을 직접적으로 transformation하는 weight를 학습하는게 아니라 Input과 Output의 차이인 residual을 학습시키는 방식
- Input에 Additive 한 값을 Convolution으로 학습
- 원활한 Gradient Flow로 인해 깊이가 152까지 되는 등 성능이 좋다.

![resnet2](https://user-images.githubusercontent.com/40047360/44298246-70b22380-a31a-11e8-9dd4-01cf68d6509f.png)
- 층을 건너뛰어 입력을 바로 출력부로 연결한다. 이렇게 유닛을 연결하게 되면 입력은 그대로 출력으로 나가게 되므로 identify 매핑이 만들어지고
  중간에 거치는 층들은 weight 값들이 0이나 0근처의 값을 가져도 될 것이다.

## 3. Fine-Tuning
### 3.1 정의
- 기존에 학습되어져 있는 모델을 기반으로 아키텍처를 새로운 목적(나의 이미지 데이터에 맞게) 변형하고 이미 학습된 모델 파라미터(Weights)로부터
  학습을 업데이트하는 방법
- 모델의 파라미터를 미세하게 조정하는 행위(이미 존재하는 모델에 추가 데이터를 투입하여 파라미터를 업데이트하는 것)

### 3.2 사용 경우
#### 1)새로 훈련할 데이터가 매우 많으며 Original데이터와 유사할 경우
- 새로 학습할 데이터의 양이 많다는 것은 오버피팅의 위험이 낮다는 뜻으로, 전체 레이어에 대해서 Fine Tuning을 한다

#### 2) 새로 훈련할 데이터가 많지만 Original 데이터와 다른 경우
- 전체 네트워크에 대해서 Fine Tuning을 한다.

