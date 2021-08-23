## 선형 회귀 Linear Regression

**머신 러닝의 가장 큰 목적은 실제 데이터를 바탕으로 모델을 생성해서 입력 값을 넣었을때의 결과를 예측하는 데에 있습니다.**

저희가 찾아낼 수 있는 가장 직관적이고 간단한 모델은 선입니다.

그래서 데이터를 놓고 그걸 가장 잘 설명할 수 있는 선을 찾는 분석하는 방법을 선형 회귀(Linear Regression) 분석이라 합니다.

예를 들어 공부 시간과 성적에 데이터를 펼쳐 놓고 그것에 대해 가장 잘 설명할 수 있는 선을 그리면, 공부를 몇 시간 해서 성적이 몇점이 

나올지 예측을 할 수 있습니다.




![img](./ML_Models\1.png)

**y = Wx + b**

기울기 W 절편 b에 따라 모양이 정해지기 때문에 x를 넣었을때 y를 구할 수 있습니다.

선형 회귀의 목적도 우리가 가진 데이터를 가장 잘 설명 할 수 있는 1차 함수 W 기울기와 b 절편을 구하는 것 입니다.



### Loss

![평균제곱오차 MSE - 제타위키](./ML_Models\2.png)

선형 회귀에서 발생하는 오차, 손실을 Loss 라고 합니다.

* Loss Function MSE ( Mean Squared Error )

  손실을 구할때 가장 널리 쓰이는 방법으로 실제 값 - 예측 데이터를 제곱을 해서 평균을 한 것으로

   ![image-20210819151051481](./ML_Models\3.png)

  위 사진의 Loss는 0.4 입니다.

  각 오류 값에 제곱을 하는 이유는 마이너스 값 제거, 큰 오류의 Loss를 더 크게 만들어 주기 위함 입니다.

  MSE 말고도 RMSE MAE 등등 많은 손실 함수가 있습니다.

  ![MSE(Mean Squared Error) 간단 정리](./ML_Models\4.png)

### 경사하강법 Gradient Descent

경사하강법 ( Gradient Descent)는 손실을 최소화 하기 위한 방법 입니다.

![Gradient Descent and Stochastic Gradient Descent - mlxtend](./ML_Models\5.png)

파라미터를 임의로 정한 후에 조금씩 변화 시키며 손실을 점점 줄여가는 방법으로 최적의 파라미터를 찾아갑니다.



### 수렴 ( Convergence )

![Linear regression with multiple features - Do It Easy With ScienceProg](./ML_Models\6.png)

선형회귀 분석을 수행하면 기울기와 절편을 변경해 가면서 최적의 값에 수렴 하게 됩니다.

### 학습률 Learning Rate



![Setting the learning rate of your neural network.](./ML_Models\7.png)

경사하강법 알고리즘은 기울기에 학습률을 곱해서 다음 지점을 결정 합니다.

**학습률이 큰 경우 : 데이터가 무질서 하게 이탈하며, 최저점에 수렴하지 못합니다.**

**학습률이 작은 경우 : 학습시간이 매우 오래 걸리며, 최저점에 도달하지 못합니다.**

![학습률 (Learning rate)](./ML_Models\8.png)

**low learning rate**: 손실 감소가 선형의 형태를 보이면서 천천히 학습됩니다.

**high learning rate**: 손실 감소가 지수적인 형태를 보이며, 구간에 따라 빠른 학습 혹은 정체가 보입니다.

**very high learning rate**: 매우 높은 학습률은 경우에 따라, 손실을 오히려 증가시키는 상황을 발생시킵니다.

**good learning rate**: 적절한 학습 곡선의 형태로, Learning rate를 조절하면서 찾아내야 합니다.



## 로지스틱 회귀 Logistic Regression

![image-20210819171843323](./ML_Models\9.png)

* **로지스틱 회귀 모델의 필요성**

  Y 값이 카테고리인 즉 연속성이 없어 선형회귀 모델과 다른 방식으로 접근해야 합니다. 

  새로운 관측치가 왔을 때 이를 기존 범주 중 하나로 예측하는 즉 분류를 하는 것 입니다.

* **로지스틱 함수**

  ![image-20210819173820049](./ML_Models\10.png)

  베르누이 시도에서 1이 나올 확률 μ와 0이 나올 확률 1−μ의 비율(ratio)을 승산비(odds ratio)라고 합니다.

  ![image-20210819174124446](./ML_Models\11.png)

  위 식인 승산비를 로그 변환한 것이 로지트함수(Logit function)입니다.

  로지트함수의 값은 로그 변환에 의해 **음의 무한대(−∞)부터 양의 무한대 (+∞)**까지의 값을 가질 수 있습니다.

  ![image-20210819174222293](./ML_Models\12.png)

  로지스틱함수는 로지트함수의 역함수로 **음의 무한대(−∞)부터 양의 무한대 (+∞)까지의 값을 가지는 입력변수를 0부터 1사의 값을 가지는 출력변수로 변환한 것 입니다.**

  

  로지스틱함수도 역시 구해야 할 것은 계수와 절편 즉 가중치(Weight)와 편향(Bias)를 인공 지능 알고리즘이 구하는 것 입니다.
  
  ### 비용 함수 Cost Function
  
  로지스틱 회귀 또한 가중치를 찾아내지만 비용 함수로는 MSE를 사용하지 않습니다.
  
  시그모이드에 MSE로 그래프를 그리면 다음 그림과 같이 로컬 미니멈에 빠질 수 있습니다.
  
  ![img](./ML_Models\13.png)

시그모이드 함수는 0과 1사이의 y값을 반환합니다. 이는 실제 값이 0일때 y값이 1에 가까워지면 오차가 커지며 실제값이 1일 때 y 값이

0에 가까워지면 오차가 커짐을 의미합니다. 그리고 이를 로그 함수로 표현이 가능합니다.

![image-20210819185752630](ML_Models\37.png)

![img](./ML_Models\14.png)

의 실제값이 1일 때 −logH(x) 그래프를 사용하고 y의 실제값이 0일 때 −log(1−H(X)) 그래프를 사용해야 합니다. 위의 두 식을 그래프 상으로 표현하면 아래와 같습니다.

![image-20210819190148683](./ML_Models\15.png)

y가 0이면 ylogH(X)가 없어지고, y가 1이면 (1−y)log(1−H(X))가 없어지는데 이는 각각 y가 1일 때와 y가 0일 때의 앞서 본 식과 같습니다.

![image-20210819190323931](./ML_Models\16.png)

이때 이 로지스틱 회귀에서 찾아낸 비용 함수를 크로스 엔트로피(Cross Entropy)함수 라고 합니다. 결과적으로 로지스틱 회귀는 크로스 엔트로피 함수를 비용 함수로 사용하고 가중치를 찾기 위해서 크로스 엔트로피의 평균을 취한 함수를 사용합니다.



## 결정 트리 Decision Tree

결정 트리(Decision Tree, 의사결정트리, 의사결정나무)는 분류(Classification)와 회귀(Regression) 모두 가능한 지도 학습 모델 입니다.

결정 트리는 스무 고개 처럼 예/아니오 질문을 이어가면서 학습을 합니다.

![image-20210819223143689](./ML_Models\17.png)

이렇게 특정 질문에 따라 데이터를 구분하는 모델을 결정 트리 모델이라고 합니다. 한번의 분기 때마다 변수 영역을 두개로 구분합니다.

결정 트리에서 질문이나 정답을 담은 네모 상자를 노드(Node)라고 합니다. 맨 처음 분류 기준을 Root Node라고 하고, 맨 마지막 노드를 

Terminal Node 혹은 Leaf Node 라고 합니다.



![image-20210819223356021](./ML_Models\18.png)

우선 위와 같이 데이터를 가장 잘 구분 할 수 있는 질문을 기준으로 나눕니다.

![image-20210819223424419](./ML_Models\19.png)

나뉜 각 범주에서 가장 잘 구분 할수 있는 질문을 기준으로 또 나눕니다.

![image-20210819223501738](./ML_Models\20.png)

하지만 이것을 지나치게 많이하면 (층이 많이 쌓이면) 보는 것과 같이 Overfitting 이 됩니다.

여기서 하이퍼파라미터(Hyperparameter)를 조정하여 Overfitting을 막을 수 있습니다.



### 엔트로피, 불순도

불순도(Imputity)란 해당 범주 안에 서로 다른 데이터가 얼마나 섞여 있는지를 뜻합니다. 아래 그림에서 위쪽 범주는 불순도가 낮고

아래 범주는 불순도가 높습니다. 

![image-20210819223736757](./ML_Models\21.png)

한 범주에 하나의 데이터만 있다면 순도가 최고이고 , 한 범주 안에 서로 다른 두 데이터가 정확히 반반 있다면 불순도가 최대가 됩니다. 

결정 트리는 불순도를 최소화 혹은 순도를 최대화 하는 방향으로 학습을 진행합니다.

 

엔트로피 (Entropy)는 불순도(Imputity)를 수치적으로 나타낸 척도입니다. 엔트로피가 높다는 것은 불순도가 높다는 것 입니다.

엔트로피를 구하는 공식은 다음과 같습니다.

![image-20210819224127565](./ML_Models\22.png)

Pi = 한 영역 안에 존재하는 데이터 가운데 범주 i에 속하는 데이터의 비율



### 정보 획득 Information gain

엔트로피가 1인 상태에서 0.7인 상태로 바꿧다면 정보 획득 (Information gain)은 0.3 입니다. 분기 이전의 엔트로피에서 

분기 이후의 엔트로피를 뺀 수치를 정보 획득이라고 합니다. 정보 획득은 아래와 같이 공식화를 할 수 있습니다.

**Information gain = entropy(parent) - [weighted average] entropy(children)**

entropy(parent)는 분기 이전 엔트로피고 , entropy(children)은 분기 이후 엔트로피 입니다.

이때 weighted average는 가중 평균을 뜻 합니다. 분기 이후 엔트로피에 가중 평균을 하는 이유는 분기를 하면 범주가 2개 이상으로

쪼개지기 때문입니다. 범주가 하나라면 위 엔트로피 공식으로 바로 엔트로피를 구할 수 있지만, 범주가 2개 이상일 경우 가중 평균을

활용하여 분기 이후 엔트로피를 구하는 것입니다.



**결정트리 알고리즘은 정보 획득을 최대화 하는 방향으로 학습이 진행 됩니다.** 어느 feature의 어느 분기점에서 정보 획득이 

최대화되는지 판단을 해서 분기가 진행됩니다.

## 앙상블 Ensemble

앙상블은 조화 또는 통일을 의미합니다.

앙상블 학습은 여러개의 결정 트리(Descision Tree)를 결합하여 하나의 결정 트리보다 더 좋은 성능을 내는 머신러닝 기법입니다.

앙상블 학습의 핵심은 여러 개의  약 분류기 (Weak Classifier) 를 결합하여 강 분류기 (Strong Classifier)를 만드는 것 입니다.

그리하여 모델의 정확성이 향상됩니다.

* 앙상블 학습 유형

  앙상블 학습은 일반적으로 보팅(Voting), 배깅(Bagging), 부스팅(Boosting) 세 가지의 유형으로 나눌 수 있습니다.

  * 보팅(Voting)
    * 하드 보팅(Hard Voting)
      * 다수의 분류기가 예측한 결과값을 최종 결과로 선정
      * ![앙상블(Ensemble) 기법](./ML_Models\23.png)
    * 소프트 보팅(Soft Voting)
      * 모든 분류기가 예측한 레이블 값의 결정 확률 평균을 구한 뒤 가장 확률이 높은 레이블 값을 최종 결과로 선정
      * ![앙상블(Ensemble) 기법](./ML_Models\24.png)

  * 배깅(Bagging)

    * ![머신러닝 - 11. 앙상블 학습 (Ensemble Learning): 배깅(Bagging)과 부스팅(Boosting)](./ML_Models\25.png)
    * **배깅(Bagging)은 Bootstrap Aggregating의 줄임말로, 부트스트래핑을 이용한 앙상블 학습법**
    * 부트스트래핑과 패이스팅
      - 부트스트래핑 : 학습 데이터셋에서 중복을 허용하여 랜덤하게 추출하는 방식 (리샘플링)
      - 페이스팅 : 학습 데이터셋에서 중복 없이 랜덤하게 추출하는 방식
    * 부트스트래핑 장단점
      - 장점: 분산 감소
      - 단점 : 중복으로 인해, 특정 샘플은 사용되지 않고 특정 샘플은 여러번 사용되어 편향될 가능성
        - OOB(Out-of-Bag) 샘플: 샘플링 되지 않은 나머지 샘플

    * 배깅을 적용한 대표적인 기법으로 Random Forest 가 있습니다.

  * 부스팅(Boosting)

    * 부스팅은 가중치를 활용하여 약 분류기를 강 분류기로 만드는 방법입니다. 

      부스팅은 모델 간 팀워크가 이루어집니다. 처음 모델이 예측을 하면 그 예측 결과에 따라 데이터에 가중치가 부여되고,

      부여된 가중치가 다음 모델에 영향을 줍니다. 잘못 분류된 데이터에 집중하여 새로운 분류 규칙을 
    
      만드는 단계를 반복합니다.
    
    * ![img](./ML_Models\26.png)
    
    * 장단점
      - **장점**: 오답에 대해 높은 가중치를 부여하고 정답에 대해 낮은 가중치를 부여하여 오답에 더욱 집중
        - **단점**: 이상치(Outlier)에 취약
      
    * 부스팅의 대표적인 알고리즘
    
      * AdaBoost
      * Gradient Boost Machine
      * XGBoost
      * LightBGM

## 랜덤 포레스트 Random Forest

랜덤 포레스트의 포레스트는 숲(Forest)입니다. 결정 트리의 트리는 나무(Tree)입니다.

이름 처럼 나무가 모여 숲을 이룹니다. 즉 결정 트리(Decision Tree)가 모여 랜덤 포레스트(Random Forest)를 구성합니다.

결정트리 하나만으로 머신러닝을 할 수 있지만, 훈련 데이터에 Overfitting 되는 경향이 있습니다.

하지만 이 랜덤 포레스트(Random Forest)에선 여러개의 결정 트리(Decision Tree)를 사용해 Overfitting 되는

단점을 해결 할 수 있습니다.



Feature가 많아지면 많아질수록 트리의 가지가 많아질 것 이고 이는 Overfitting이 될 수 있습니다.

하지만 여러개의 Feature중 몇개의 Feature만 선택을 하여 결정트리들을 만들면 이 예측 값들을 모와 과적합을 방지하면서

결과 값을 예측 할 수 있습니다. 

이렇게 의견을 통합해서 다수결의 원칙으로 Ensemble을 합니다.

**문제를 풀 때도 한명의 똑똑한 사람보다 100명의 평범한 사람이 더 잘 푸는 원리입니다.**
![image-20210819225520070](./ML_Models\27.png)

기존 각 결정 트리(Decision Tree)의 경계는 다소 모호하고 Overfitting이 된 모습을 볼 수 있는데

5개의 결정 트리(Decision Tree)를 기반으로 평균을 내어서 만든 랜덤 포레스트는 비교적 깔끔한 것을 볼 수 있습니다.

## 에이다 부스트 AdaBoost

에이다 부스트는 부스팅 기법 중 가장 기본이 되는 알고리즘입니다.



아래와 같이 노드 하나에 두개의 리프(Leaf)를 가지는 트리를 stump라고 합니다.

![image-20210819215922023](./ML_Models\28.png)

AdaBoost는 아래와 같이 여러 개의 stump로 구성이 되어있습니다. 이를 Forest of stumps라고 합니다.

![image-20210819220002848](./ML_Models\29.png)

Random Forest는 여러개의 트리를 합산해 최종 결과는 도출 합니다. 다수결의 원칙 처럼 각 최종분류를 하는데 있어 모두

동등한 가중치를 가지고 있습니다. 

하지만 AdaBoost 에선 각각의 가중치가 다릅니다. 여기서 가중치가 높다는 것을 Amount of Say가 높다고 표현을 합니다.

![image-20210819220345382](./ML_Models\30.png)



AdaBoost는 3가지의 특징을 가지고 있습니다.

 * 약한 학습기(Weak Learner)으로 구성되어 있고 Stump의 형태 입니다.
 * 어떤 Stump는 다른 Stump 보다 가중치가 높습니다.
 * 각 Stump의 Error은 다음 Stump의 결과에 영향을 줍니다.



### AdaBoost 작동 원리

![image-20210819221139788](./ML_Models\31.png)

Chest Pain, Blocked Arteries, Patient Weight에 따른 Heart Disease 여부에 대한 데이터입니다. 맨 처음 Sample Weight는 8개의 

데이터 모두 동일하게 1/8입니다.



각 Stump를 만든 후 지니 계수를 구합니다.

![image-20210819221345580](./ML_Models\32.png)



마지막 Stump의 지니 계수가 가장 작기 때문에 Forest의 첫 Stump로 지정합니다.

### Amount Of Say 구하기

틀리게 분류한 것이 No Heart Disease의 Incorrect로 1개밖에 없습니다. 따라서 Total Error = 1/8입니다.

![image-20210819221657766](./ML_Models\33.png)

모든 Sample Weight의 합은 1이기 때문에, Total Error은 0과 1사이의 값을 갖습니다. Total Error가 Amount of Say를 결정하고. 

Amount of Say는 최종 분류에 있어 해당 Stump가 얼마나 영향을 주는지를 뜻합니다. 



* Amount of Say 공식

![image-20210819221857800](./ML_Models\38.png)

Amount of Say를 그래프로 그리면 아래와 같습니다.

![img](./ML_Models\34.png)

X 축은 Total Error Y 축은 Amount of Say 입니다 Total Error가 0이면 굉장히 큰 양수이고

Total Error가 1이면 굉장히 작은 음수가 됩니다. 따라서 Total Error가 0이면 분류를 올바르게 했다는 뜻이고

1이면 분류를 반대로 한다는 뜻 입니다.



위 Stump 에서 Total Error = 1/8이라고 했으니

![img](./ML_Models\35.png)

Amount of Say 가 0.98인 지점 입니다.



### 샘플 가중치

Adaboost에서는 하나의 Stump가 잘못 분류한 sample에 대해서는 다음 Stump로 넘겨줄 때 가중치를 더 높여서 넘겨줍니다. 그래야 다음 Stump에서 해당 Sample에 더 집중해서 올바로 분류해주기 때문입니다.

다시 맨 처음에 했던 방식 대로 계속 반복을 하면 됩니다.



### 최종 분류

여러 차례 진행 하다보면 아래와 같이 각 Stump 마다의 Amount of Say 가 나오게 됩니다.

왼쪽은 Heart Disease가 있다고 한 Stump이고 오른쪽은 없다고 판단한 Stump 입니다.

결국 Has Heart Disease의 Total 이 Does Not Have Heart Disease 보다 크니 Has Heart Disease 라고 예측 할 수 있게 됩니다.

![image-20210819222505829](./ML_Models\36.png)

각 Stump 하나하나의 분류력은 굉장히 약하지만 여러 Stump의 결과를 종합하면 강한 학습기(Strong Learner)가 됩니다.

## Gradient Boosting Machines

Gradient Boosting Machines (GBM)은 Boosting의 개념을 Gradinet Descent라는 최적화 방법으로 이해하는 방법입니다.

GBM 간략하게 설명하면 다음과 같습니다.

1. ![image-20210821142833807](./ML_Models\39.png)에 대한 초기 예측값으로 ![image-20210821142906588](./ML_Models\40.png)값들의 평균인 ![image-20210821142938046](./ML_Models\41.png)를 택합니다.
2. ![image-20210821143102712](./ML_Models\42.png)만큼 반복
   	1. 각 관측지마다 잔차![image-20210821143155795](./ML_Models\43.png)를 구하고, 이 잔차에 첫 번째 약한 학습기를 적합시킵니다.
    	2. 적합된 결과에 따라 잔차의 예측값 (마지막 노드 (leaf)에 속한 잔차들의 평균)을 구하고
    	3. 잔차 예측값 * 학습률을 이전 예측값과 더해 새롭게 예측값을 업데이트합니다.

### Gradient Descent 란?

![image-20210821143333024](./ML_Models/5.png)

경사하강법은 **미분 가능한 볼록함수**, 즉 y = x^2과 같은 오목 함수에서 y가 최소값을 갖게 하는 x를 찾기 위한 알고리즘입니다.

이를 위해 Gradient Descent는 함수의 기울기를 구하여 기울기가 낮은 쪽으로 x값을 이동시켜서 y가 **극소값**을 갖도록 합니다.

* 극소값은 Local Minimum으로 가장 작은 함수값이라 말할 수는 없지만 주변에 비해 작은 값 입니다.
* 최소값은 Global Minimum으로 전체 범위에서 가장 작은 함수 값 입니다.



Gradient Descent 알고리즘은 다음과 같습니다.

초기값 ![image-20210821143835829](./ML_Models\44.png)을 선택하고, 다음의 식을 반복적으로 실행해서 임의의 조건을 만족하면 종료하게 됩니다.
$$
x_t=x_{t−1}−ηΔf(x_{t−1}), t=1,2,...
$$


이처럼 t번째 단계의 ![image-20210821144314408](./ML_Models\45.png)는 이전 단계의 값![image-20210821144415771](./ML_Models\46.png)에서 η와 ![image-20210821144503495](./ML_Models\47.png)를 곱한 값을 빼주어 값을 지속적으로 업데이트합니다.

여기서 η은 학습률(Learning rate)라 불리우는데, 한번 업데이트 할 때 얼마나 큰 크기로 업데이트할지 조정하는 값 입니다.

그리고 ![image-20210821144503495](./ML_Models\47.png)이 기울기(Gradient)에 해당합니다. 기울기는 함수를 **1차 미분**한 것과 같습니다.



즉 t번째 값의 ![image-20210821144314408](./ML_Models\45.png)는 

* 기울기 > 0 이면 ![image-20210821220932821](./ML_Models\48.png)보다 더 작은 값으로 업데이트되고
* 기울기가 < 0 이면 ![image-20210821220932821](./ML_Models\48.png)보다 큰 값으로 업데이트 됩니다.

기울기 = 0일때 극값을 갖기 때문에 기울기 > 0 이면 x가 극소값보다 오른쪽에 있다는 뜻이므로 학습률 * 기울기 만큼 빼주어

왼쪽으로 이동시켜주는 것이고, 기울기 < 0 이면 극소값보다 왼쪽에 있다는 뜻이므로 학습률 * 기울기만큼 더해주어

오른쪽으로 이동시켜주는 원리입니다.



계속 반복하는 것이 아니라 t번째 값인 ![image-20210821144314408](./ML_Models\45.png)와 t - 1번째 값인 ![image-20210821220932821](./ML_Models\48.png)이 **거의 같으면** 그 값이 수렴했음을 의미하기 때문에 반복을 멈춥니다.



손실 함수(Loss Function)를 MSE를 2로 나눈 것으로 정의하면, 이 손실 함수의 1차 미분값(Gradient)의 음수를 취한게

바로 잔차 (residual)이 됩니다.

손실함수는 예측값과 실제값의 차이를 나타내는 함수이기 때문에 값이 작을수록 좋습니다. 그래서 이 손실 함수의 극소값을 찾는데

경사 하강법을 이용한 최적화 문제로 바꿀 수 있습니다.



손실 함수를![image-20210821222121825](./ML_Models\49.png)로 정의하면 최소화해야 할 목적 함수(Objective Function)은

![image-20210821222235905](./ML_Models\50.png)가 됩니다.



이 목적 함수에서 극소값을 찾기 위해서 1차 미분을 하면

![image-20210821222358138](./ML_Models\51.png)

따라서 **잔차를 1차미분의 음수값(negative gradients)**라고 할 수 있습니다.
$$
y_i−F(x_i)=−\frac{∂J}{∂F(x_i)}
$$

결국 ![image-20210823165529508](./ML_Models\52.png)는 (t - 1번째 예측값 + 학습률 * 잔차)로 업데이트가 됩니다.

![image-20210823180147686](./ML_Models\53.png)

이 처럼 남은 잔차를 이용해 계속해서 학습합니다.

## XGBoost

* XGBoost란 ?

  "XGBoost: A Scalable Tree Boosting System" 논문에서 소개된 "Extreme Gradient Boosting"을 의미합니다.



* XGBoost를 쓰는 이유

  * 여러 파라미터를 조절해가면서 최적의 모델을 만들 수 있습니다.

  * Overfitting을 방지할 수 있습니다.

  * 자원이 많으면 많으면 많을수록 빠르게 학습하고 예측할 수 있습니다.

  * Cross validation을 지원합니다.

  * 높은 성능을 나타냅니다. Kaggle에서 XGboost를 쓴 결과들이 상위권을 차지하고, 대부분의 데이터 셋에서 90%의 이상이

    GBM 보다도 더 좋은 성능을 보여줍니다.

  * C언어로 작성해 수행 속도가 빠릅니다.

* 학습

  ![image-20210823191207279](./ML_Models\54.png)

  위와 같은 데이터가 있다고 하면 X축은 복용량이고 Y축은 효과입니다.

  ![image-20210823191302849](./ML_Models\55.png)

  초기 예측값은 0.5로 설정합니다.



​		**Gain을 가장 극대화**하는 방법으로 트리구조 처럼 여러 예측값으로 나눕니다.

​		![image-20210823191443340](./ML_Models\56.png)

​		Similarity Score를 계산하는 공식은 다음과 같습니다.

​		![image-20210823191528678](./ML_Models\57.png)

​		여기서 λ는 Overfitting을 다루기 위한 정규화 파라미터입니다.



​		이 케이스에서 Old Similarity Score는 다음과 같이 계산됩니다.

![image-20210823191651511](./ML_Models\58.png)

![image-20210823191727630](./ML_Models\59.png)

​		트리의 분기 조건을 각각 빨간 줄, 초록 줄, 파란 줄로 설정하고 나누면 New Similarity Score를 계산합니다.

​		**총 유사도(Similarity Score)는 모든 리프 노드의 유사도의 합입니다.**

​		이 예제에서 빨간 줄로 나누었을 때 총 유사도는 110.25 + 14.08이 됩니다.

​		![image-20210823193338408](./ML_Models\60.png)

그래서 Gain값은 (110.25 + 14.08) - 4가 됩니다.

빨강, 초록, 파랑으로 각각 나누어보고 **Gain 값이 가장 높은 쪽으로 분기합니다.**

![image-20210823194611997](./ML_Models\61.png)

빨간색을 기준으로 **분기한 Tree를 만들고 리프 노드(Leaf node)에 대해서 위 작업을 재귀적으로 반복**합니다.
만약 **모든 조건에서 Gain값이 음수일 경우 분기**를 중지합니다. 이 예제에서는 위와 같이 완성됩니다.

![image-20210823194737995](./ML_Models\62.png)

**지금까지의 과정을 통해 새로 만들어진 Tree에 대해서 반복합니다. N회 반복하거나 M회 이상 validation 결과가 개선되지 않을 때 종료**합니다.

![a](./ML_Models\63.png)

이렇게 만들어진 모든 트리마다 각각의 Weight를 갖고 부스팅 방식으로 최종 Output을 결정합니다.

## LightGBM

* LGBM 알고리즘

  ![image-20210823215026208](./ML_Models\64.png)

  LightGBM은 나무를 수직으로 확장합니다. 반면 다른 알고리즘은 나무를 수평으로 확장합니다.

  따라서 기존의 알고리즘은 수평으로 확장을 하여 포화트리를 만드는 방향으로 학습하는 반면 LGBM은 최대 delta loss가

  증가하도록 잎의 개수를 정합니다. 

  leaf-wise알고리즘은 다른 level-wise 알고리즘 보다 낮은 loss를 달성하는 경향이 있습니다. 

  데이터의 크기가 작은 경우 leaf-wise 알고리즘은 Overfitting되기 쉬운 경향이 있어서 max_depth를 줄여줘야 합니다.



* 장점 

  LightGBM은 Light 단어 처럼 속도가 빠릅니다.

  메모리를 적게 차지하고 속도가 빠르면서도 LightGBM은 정확도가 높습니다.