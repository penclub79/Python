# Deep-theory
## 1. 머신러닝
 
1.1 머신러닝 정의
- 데이터를 이용해서 명시적으로 정의되지 않은 패턴을 컴퓨터로 학습하여 결과를 만들어 내는 학문이다.
- 머신러닝은 항상 데이터를 기반으로 하며, 데이터를 보고 패턴을 추리하는 것이 머신러닝의 핵심.

1.2 머신러닝의 원리'
1.2.1 모델(문제를 바라보는 관점)
- 데이터가 어떤 패턴을 가질 것이라는 가정을 이용하여 현재 상태를 어떠한 시각으로 바라보고 어떠한 기대를 하고 있는가 하는 것이 모델이다.

![default](https://user-images.githubusercontent.com/40047360/43994469-2737733c-9dd8-11e8-981e-4f80c99a7f19.png)

* 간단한 모델
  * '데이터의 구조가 간단하다'라는 뜻으로 강력한 가정을 한다고 해석하면 된다. 가장 간단하지만 아주 효과적인 모델로 '선형모델'이 있다
* 복잡한 모델
  * 데이터가 상대적으로 더 복잡하게 생겼다고 가정할 때 사용하는 모델이다. 복잡한 모델은 모델의 유연성을 더 중요시 한다. 'SVM'
* 순차 모델
  * 연속된 관측값이 서로 연관성이 있을 때 주로 사용한다. 가장 큰 특징은 특정 시점에서 상태를 저장하고, 상태가 각 시점의 입력과 출력에 따라 
    변화한다는 점이다.
* 그래프 모델
  * 그래프를 이용해서 순차 모델보다 좀 더 복잡한 구조를 모델링 한다.'CNN'


1.2.3 최적화
- 실제로 학습을 하는 방법이다. 손실함수의 결과값을 최소화하는 모델의 인자를 찾는 것을 최적화라고 한다.

* 경사 하강법
경사하강법은 간단한 최적화 방법 중 하나이다. 임의의 지점에서 시작해서 경사를 따라 내려갈 수 없을 때까지 반복적으로 내려가면 최적화를 수행한다.
![default](https://user-images.githubusercontent.com/40047360/43995070-96556468-9de2-11e8-84c8-27f3e73b5ae9.png)

초기값으로부터 경사를 따라 차츰차츰 내려가서 결국 최저점에 도달한다. 경사는 곡선의 접선방향으로 결정되는데, 이를 계산하려면 손실함수를 인자에 대해 미분하면 된다.

1.2.4 모델평가 (실제 활용에서 성능을 평가하는 방법)
- 모델 평가는 모델이 얼마나 좋은 성능을 보일지 평가하는 방법이다. 손실함수와 모델 평가 방식이 정확히 일치하는 경우도 있지만, 개념이 서로 다르고, 
  모델 평가 방식을 수학적으로 직접 최적화하기 어려운 경우가 많아서 별도로 취급한다.

- 모델 평가를 할 때는 학습 데이터뿐만 아니라 학습 데이터가 아닌 새로운 데이터가 들어 왔을 때도 잘 동작하는지 측정한다. 이를 일반화라고 하며, 실제 
  머신러닝 시스템을 구축할 때 굉장히 중요한 요소이다.

 - 일반화가 중요한 이유는 학습에 사용되는 관측된 데이터들을 한정된 패턴들만 보여주기 때문이다. 따라서 관측된 데이터에 지나치게 의존해 학습하면 진짜 
   분포에서 오히려 멀어 질 수 있다. 이런 문제를 과학습(오버피팅)이라고 한다.

1.3 머신러닝 분류

1.3.1 지도학습
지도학습은 주어진 데이터와 레이블(정답)을 이용해서 미지의 상태나 값을 예측하는 방법이다. 대부분의 머신러닝 문제는 지도학습에 해당한다. 
예를 들어 '예전의 주식 시장 변화를 보고 내일의 주식 시장 변화 예측', '문서에 사용된 단어를 보고 해당 문서의 카테고리 분류' 등이 여기 속한다.

* 회귀와 분류
  * 회귀의 경우에는 숫자값을 예측한다. 대개 연속된 숫자를 예측하는데 예를 들어 기존 온도 추이를 보고 내일 온도를 예측하는 일을 들 수 있다.
  * 분류는 입력데이터들을 주어진 항목들로 나누는 방법이다. 예를 들어 어떤 문서가 도서관 어떤 분류에 해당하는지 고르는 경우를 들 수 있다.
  * 회귀를 통해 손쉽게 분류를 구현하거나 분류를 통해 회귀를 구현할 수 있을 만큼 서로 유사하다. 예를 들어 어제의 온도와 구름의 양으로 내일의 날씨가 
    좋을지 안 좋을지를 예측하는 분류 시스템을 만든다고 할 때 날시가 좋은 경우를 1, 나쁜 경우를 0으로 두어서 이 값이 0.8이상이면 좋음, 아니면 나쁨
    으로 분류할 수 있다.

* 추천시스템과 랭킹 학습
  * 추천시스템은 상춤에 대한 사용자 선호도를 예측하는 시스템이다. 상품과 사용자 데이터를 이용하여 값 혹은 레이블을 예측하는 것이므로 회귀의 
     일종으로 볼 수 있다.
  * 랭킹학습은 데이터의 순위를 예측합니다. 좋아할만한 영화 10편을 추천한다면 랭킹학습에 해당 한다.

1.3.2 비지도 학습
* 군집화와 토픽 모델링
군집화(클러스터링)은 비슷한 데이터들을 묶어서 큰 단위로 만드는 기법이다. 
토픽모델링은 군집화와 매우 유사하지만 주로 텍스트 데이터에 대해 사용된다. 토픽모델링은 보통 한 문서가 토픽에 따라 만들어 지고 그에 따라 단어가 생성되어 문서가 씅진다는 가정하에 접근한다.

* 밀도 추정
관측한 데이터로부터 데이터를 생성한 원래의 분포를 추측하는 방법이다. 예를 들어 각국의 학생들의 키와 몸무게를 모아놓은 통계자료에서 키와 몸무게의 관계를 분석 할 때, 여러가지 다른 기법으로 더 정확한 분포를 얻을 수 있다. 커널 밀도 추정, 가우스 혼합모델이 대표적인 기법이다.

* 차원 축소
말 그대로 '데이터의 차원을 낮추는 기법.' 보통은 데이터가 복잡하고 높은 차원을 가져서 시각화하기 어려울 때 2차원이나 3차원으로 표현하기 위해서 사용. 일반적으로 주요 패턴을 찾아서 해당 패턴을 낮은 차원에서 보존하는 방식으로 이루어 진다.

1.4 머신러닝 활용 예시

* 사기방지
연간 2천억 달러 이상의 경제를 처리하는 Paypal의 창업초기 월별 사기 피해 금액이 1,000만 달러에 이르렀었다. Paypal은 이 문제를 해결하기 위해 최고의 연구원들로 팀을 꾸렸고 이팀은 최신 머신러닝 기법을 사용해 사기성 결제를 실시간으로 식별하는 모델을 구축했다.

* 타겟팅 디지털 디스플레이
광고 기술 기업 Dstillery는 머신러닝 기법을 사용하여 기업의 실시간 입찰 플랫폼에서 타겟 디지털 디스플레이 광고를 진행하도록 한다. 디스틸러리는 개인의 브라우징 내력, 방문, 클릭 및 구매에 대해 수집된 데이터를 사용하여 한번에 수백 개의 광고 캠페인을 처리하여 초당 수천 건의 예측을 실행한다.

* 콘텐츠 추천
Comcast는 인터랙티브 TV서비스를 고객을 위해 각 고객의 이전 시청 습관을 기반으로 하여 실시간 개인 맞춤화 콘텐츠를 추천한다. Comcast가 운영하는 머신러닝은 수십 억개의 내역 기록을 상ㅇ하여 각 고객별로 고유한 취향 프로필을 작성한 다음, 공통적인 취향을 가진 고객을 클러스터로 묶는다. 그런 다음 각 고객 클러스터를 대상으로 가장 인기있는 콘텐츠를 실시간으로 추적, 표시하여 고객이 현재 인기있는 콘텐츠를 볼 수 있도록 한다.

## 2. 딥러닝(Deep Learning)
2.1 딥러닝 정의 
신경망을 층층이 쌓아서 문제를 해결하는 기법의 총칭이다. 사용하는 기법이 특정 형태를 가지는 것을 말하는 것으로 이는 데이터의 양에 의존하는 기법으로 
다른 머신러닝 기법보다 문제에 대한 가정이 적은 대신 다양한 패턴과 경우에 유연하게 대응하는 구조를 만들어 많은 데이터를 이용하는 학습시키는 것으로 
모델의 성능을 향상 시킨다. 즉, 큰 데이터에서 잘 동작하는 방법이다. (약 5G이상)

2.2 딥러닝 원리
- 초기 머신러닝 연구자들이 만들어 낸 또 다른 알고리즘인 인공 신경망(Artificial Neural Network)에 영감을 준 것은 인간의 뇌가 지닌 생물학적 특성, 
  특히 뉴런의 연결 구조였다. 그러나 물리적으로 근접한 어떤 뉴런이든 상호 연결이 가능한 뇌와는 달리, 인공 신경망은 레이어 연결 및 데이터 전파 방향이 
  일정하다.

- 딥러닝은 인공신경망에서 발전한 형태의 인공 지능으로, 뇌의 뉴런과 유사한 정보 입출력 계츨을 활용해 데이터를 학습한다. 그러나 기본적인 신경망조차 
  굉장한 양의 연산을 필요로 하는 탓에 딥러닝의 상용화는 초기부터 난관에 부딪혔다. 그럼에도 토론토대 제프리 힌튼(Geoffrey Hinton)교수 연구팀과 같은 
  일부 기관에서는 연구를 지속했고, 슈퍼컴퓨터를 기반으로 딥러닝 개념을 증명하는 알고리즘을 병렬화하는데 성공했다. 그리고 병렬연상에 최적화된 GPU의 
  등장은 신경망의 연산 속도를 획기적으로 증가시키며 진정한 딥러닝 기반 인공 지능의 등장을 불러왔다.

- 딥러닝은 선형 맞춤과 비선형 변환을 반복하여 쌓아올린 구조이다. 다시 말해서, 인공 신경망은 데이터를 잘 구분할 수 있는 선들을 긋고 이 공간들을 
  잘 왜곡해 합하는 것을 반복하는 구조라고 할 수 있다.

- 컴퓨터가 사진 속에서 고양이를 검출해내야 할 때, '고양이'라는 추상적 이미지는 선, 면, 형상, 색깔, 크기 등 다양한 요소들이 조합된 결과물이다. 
  '갈색'은 고양이, '빨간색'은 고양이가 아니다 라고 간단한 선형 구분만으로는 식별해 낼 수 없는 문제가 있다. 딥러닝은 이 과제를 선을 긋고, 
  왜곡하고 합하고를 반복하며 복잡한 공간 속에서의 최적의 구분선을 만들어 내는 목적을 가지고 있다.

2.3 딥러닝 분류
2.3.1다층 신경망
 - 다층 신경망은 하나 혹은 그 이상의 '은닉층'이 있는 신경망이다. 출력층은 은닉층의 출력 신호를 받아들이고 전체 신경망의 출력 패턴을 정한다. 
   은닉층 의 뉴런은 신호의 특성을 파악한다. 가중치를 통해 입력패턴에 숨겨져 있는 특성을 알 수 있다. 출력층은 이 특성을 사용하여 출력패턴을 
   결정한다.

2.3.2 컨볼루션 신경망
- 컨볼루션 신경망은 영상 인식에 특화된 심층 신경망이다. 컨볼루션 신경망은 전처리를 추가한 다층퍼셉트론의 한 종류이지만 2차원 데이터의 입력이 
  용이하고 훈련이 용이하며 적은 매개변수라는 장점을 가지고 있어 많이 사용된다.

2.3.3 순환신경망
- 인공 신경망의 한 종류로써, 출력된 정보가 입력으로 재사용될 수 있는 신경망이다. 유닛간의 연결이 순환적 구조를 갖는 특징이 있다. 또한 시변적 동적 
  특징을 모델링할 수 있도록 신경망 내부에 상태를 저장할 수 있게 해준다. 내부의 메모리를 이용해서 시퀀스 형태의 입력을 처리한다. 필기체 인식이나 
  음성 인식과 같이 시간의 흐름에 따라 변하는 특징을 가지는 데이터를 처리할 수 있다.

2.4 딥러닝 활용 예시
- 2012년, 구글과 스탠퍼드대 앤드류 응(Andrew NG) 교수는 1만 6,000개의 컴퓨터로 약 10억 개 이상의 신경망으로 이뤄진 
  '심층신경망(Deep Neural Network)'을 구현했다. 이를 통해 유튜브에서 이미지 1,000만 개를 뽑아 분석한 뒤, 컴퓨터가 사람과 고양이 사진을 
  분류하도록 하는데 성공했다. 컴퓨터가 영상에서 나온 고양이의 형태와 생김새를 인식하고 판단하는 과정을 스스로 학습하게 한 것이다.

딥러닝으로 훈련된 시스템의 이미지 인식 능력은 이미 인간을 앞서고 있다. 이 밖에도 딥러닝의 영역에는 혈액의 암세포, MRI 스캔에서의 종양 식별 능력 등이 포함된다. 구글의 알파고는 바둑의 기초를 배우고, 자신과 같은 AI를 상대로 반복적으로 대국을 벌이는 과정에서 그 신경망을 더욱 강화해 나갔다.
