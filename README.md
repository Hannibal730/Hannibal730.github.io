---
layout: post
title: "Stealing Hyperparameters in Machine Learning"
date: 2025-07-17
author: 최대승 학부인턴
meta: 건국대학교 응용통계학과
description: "Model Stealing and Application 논문 세미나 자료"
tags: lab-seminar hyperparameter-stealing MLaas
categories: privacy
related_posts: true
toc:
  beginning: true
---
> 논문명: EModel Stealing and Application
> 
> 저자: Binghui Wang, Neil Zhenqiang Gong
> 
> 게재지: Symposium on Security and Privacy SP, San Francisco, CA, 2018. IEEE.
> 
>URL:[Stealing Hyperparameters in Machine Learning](https://arxiv.org/abs/1802.05351)
{: .block-warning }

## 서론
<hr>


**MLaaS (Machine Learning as a Service)** 는 머신러닝 모델 개발, 학습, 배포, 관리를 클라우드 플랫폼(cloud platform)에서 제공하는 서비스입니다. MLaaS 덕분에 기업과 개인은 고성능 머신러닝 모델을 저비용으로 쉽게 활용할 수 있습니다. 그러나 MLaaS의 모델은 **하이퍼파라미터 도용 (Hyperparameter stealing)** 위험에 노출될 수 있습니다.

하이퍼파라미터는 모델 성능을 결정하는 핵심 요소이며, 다양한 알고리즘에서 모델의 학습 방향을 조정합니다. 하이퍼파라미터를 학습하고 최적화하는 과정은 고비용을 수반하기 때문에 **상업적 기밀**로써 간주됩니다. 하지만 사용자가 학습 데이터와 모델 파라미터를 MLaaS와 공유하는 과정에서, 악의적 사용자가 하이퍼파라미터를 추출할 수 있습니다. 그리고 이는 MLaaS의 비용 구조를 악용하여 고성능 모델을 저비용으로 학습하려는 시도로 이어질 수 있습니다. 이러한 보안 위협은 MLaaS의 상업적 가치를 저해할 수 있으므로, 이에 대한 체계적인 분석과 대응이 필요합니다.

이러한 배경 속에서 본 논문은 하이퍼파라미터 도용이라는 상업적 위협을 분석하고 방어 기법을 제안합니다. 본 연구는 Amazon Machine Learning과 같은 실제 MLaaS 플랫폼에서 다양한 알고리즘을 대상으로 하이퍼파라미터 도용 공격을 실험하고 그 효과를 입증합니다. 이를 통해 MLaaS 비용 구조의 취약점을 드러내고, 모델 파라미터 **반올림 (Rounding)** 방어 기법의 효과와 한계를 평가하여 새로운 대응책의 필요성을 제안합니다.

## 사전지식
<hr>

#### 하이퍼파라미터 (Hyperparameter) vs 모델 파라미터 (Model Parameter)

**하이퍼파라미터**는 머신러닝 모델의 학습 과정을 조정하는 변수로, 모델 성능에 직접적인 영향을 미칩니다. 예를 들어, 정규화 파라미터 $\lambda$는 목적 함수에서 손실 함수와 정규화 항의 균형을 조정하여 과적합을 방지합니다.
$$
\begin{equation}
\mathcal{L}(\mathbf{w}) = \mathcal{L}_{\text{loss}}(\mathbf{w}) + \lambda \mathcal{R}(\mathbf{w})
\end{equation}
$$

**모델 파라미터**는 학습 과정에서 목적 함수를 최소화하여 결정된 값으로, 기울기 $\nabla \mathcal{L}(\mathbf{w}) = 0$를 만족하는 지점에서의 파라미터를 의미합니다. 이때 목적함수의 종류에 따라서 모델 파라미터가 달라집니다. 


#### 비커널 알고리즘 (Non-Kernel Algorithm) vs 커널 알고리즘 (Kernel Algorithm)

목적함수가 어떤 알고리즘인지에 따라서 모델 파라미터 표현방식이 다릅니다.

**비커널 알고리즘**은 선형 결합 ($\mathbf{w}^\top \mathbf{x}$)을 사용하는 알고리즘입니다. 목적함수가 비커널 알고리즘이면 모델 파라미터는 $\mathbf{w}$로 표현됩니다. 
    
    RR (Ridge Regression), LASSO, LR (Logistic Regression), SVM (Support Vector Machine) 등이 포함됩니다. 

**커널 알고리즘**은 데이터 $\mathbf{x}_i$를 비선형 고차원 공간으로 매핑 $\phi(\mathbf{x}_i$)하는 알고리즘입니다.  목적함수가 아래처럼 커널 알고리즘이며 노름 기반 정규화 항을 사용한다면, Representer theorem을 사용할 수 있습니다.


$$
\begin{equation}
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \ell(f(\mathbf{x}_i), y_i) + \lambda R(\mathbf{w})
\end{equation}
$$

**Representer theorem**을 사용하면, 최적화된 모델 파라미터 $\mathbf{w}$가 학습 데이터 $\phi(\mathbf{x}_i)$의 선형 결합 ($w = \sum_i \alpha_i \phi(x_i)$)으로 표현됩니다. 그 결과, 목적함수가 커널 알고리즘이면 모델 파라미터는 $\alpha$로 대체됩니다.

    KRR (Kernel Ridge Regression), KLR (Kernel Logistic Regression), KSVM (Kernel Support Vector) 등이 포함됩니다.



#### 위협 모델 (Threat Model)

공격자는 학습 데이터셋($ X, y $), 학습 알고리즘, 그리고 경우에 따라 모델 파라미터($ w $ 또는 $ \alpha $)를 알고 있다고 가정합니다.
화이트박스 설정: 모델 파라미터가 완전히 공개된 경우로, 공격자는 이를 직접 사용해 하이퍼파라미터를 추정합니다.
블랙박스 설정: 모델 파라미터가 알려지지 않은 경우, 기존 연구(예: Tramer et al.)의 모델 파라미터 도용 기법을 활용해 추정 후 하이퍼파라미터를 계산합니다.
데이터셋: 실험에 사용된 데이터셋은 Diabetes(질병 진행 예측, 회귀), GeoOrig(지리적 변수 예측, 회귀), UJIIndoor(실내 위치 예측, 회귀), Iris(꽃 분류, 분류), Madelon(합성 이진 분류), Bank(정기예금 가입 예측, 분류)입니다. 이들은 다양한 특징과 규모를 대표합니다.


#### 하이퍼파라미터 도용 메커니즘

공격은 학습 데이터와 모델 파라미터 간의 수학적 관계를 활용합니다. 머신러닝 알고리즘의 목적 함수는 일반적으로 $ \text{Loss}(w) + \lambda \|w\|_p $ 형태로, 모델 파라미터는 기울기 $ \nabla (\text{Loss}(w) + \lambda \|w\|_p) = 0 $를 만족합니다. 이를 통해 $ \lambda $를 포함한 선형 방정식을 구성하고, 최소제곱법으로 하이퍼파라미터를 추정합니다.
예: 리지 회귀에서는 $ X^T (y - Xw) = \lambda w $를 통해 $ \lambda $를 계산. 커널 알고리즘에서는 $ \alpha $를 사용해 유사한 방정식을 만듭니다.
L1 정규화(LASSO 등)에서는 비미분 가능 지점($ w = 0 $)을 제외하고 추정하며, 다중 하이퍼파라미터(예: Elastic Net의 $ \lambda_1, \lambda_2 $)는 행렬 방정식으로 해결합니다.


#### MLaaS에서의 하이퍼파라미터 도용 프로토콜

프로토콜 1: 사용자가 전체 학습 데이터셋과 알고리즘을 업로드하면, MLaaS 플랫폼이 하이퍼파라미터(예: λ)와 모델 파라미터를 학습합니다. 이 과정은 계산 비용이 높으며, 플랫폼은 선택적으로 모델 파라미터를 반환합니다.
프로토콜 2: 사용자가 데이터셋과 하이퍼파라미터를 직접 지정하면, MLaaS는 모델 파라미터만 학습해 비용이 낮습니다.
공격 전략: 공격자는 데이터셋 일부를 프로토콜 1에 업로드해 모델 파라미터를 얻고, 이를 사용해 하이퍼파라미터를 도용합니다. 이후 전체 데이터셋과 도용한 하이퍼파라미터를 프로토콜 2에 적용해 저비용으로 고성능 모델을 학습합니다(Train-Steal-Retrain, M3 전략). 이는 고비용의 하이퍼파라미터 학습을 우회해 플랫폼의 비용 구조를 악용합니다.


##### 방어 메커니즘

모델 파라미터 반올림(rounding): 모델 파라미터를 소수점 한 자리로 반올림해 공격자의 하이퍼파라미터 추정 오류를 증가시키는 기법입니다. 이는 기존 모델 역전 및 모델 도용 공격 방지에 사용되었으며, L2 정규화 알고리즘에서 효과적이나 L1 정규화에는 한계가 있습니다.
배경: 반올림은 파라미터의 정밀도를 낮춰 수학적 방정식의 정확성을 떨어뜨리지만, 모델 성능에 미치는 영향은 최소화해야 합니다. 논문은 이 기법의 효과와 한계를 실험적으로 평가합니다.




-----------------------------
기존의 FFT는 신호의 전체적인 주파수 성분을 분석할 수 있으나 시간이 지남에 따라 변화하는 주파수 패턴을 포착하는 것이 어렵다는 단점이 있습니다. 이때 spectrogram은 시간에 따른 주파수 변화를 분석할 수 있어 베어링 결함의 특성을 자세히 파악할 수 있습니다. 
spectrogram을 CNN(Convolutional Neural Network) 모델의 입력으로 사용하여 STFT 기반보다 베어링 결함의 특성을 잘 반영하여 베어링 결함 진단의 정확도를 향상시킬 수 있습니다.

#### 적대적 공격

적대적 공격은 머신러닝을 속이기 위해 악의적인 노이즈 또는 작은 변형(perbutation)을 생성하는 방법입니다. 즉 f가 주어졌을 때 다음과 같은 최적화 문제를 최대화합니다.

$$
\begin{equation}
d(x; w_u) = D_{KL} \big( p(f(x; w_o)) \parallel p(f(x; w_u)) \big)
\end{equation}
$$

즉, 적대적 공격의 목표는 원래 샘플 x의 정답 라벨 y를 변경하도록 모델을 속이는 것입니다.
많이 알려진 적대적 공격의 예시로는 FGSM(Fast Gradient Sign Method과 PGD(Projected Gradient Descent)가 있습니다.

$$
\begin{equation}
\delta = \epsilon \cdot \text{sign}(\nabla \mathcal{L}(f(x + \delta), y))
\end{equation}
$$

FGSM은 다음 수식과 같이 기울기를 이용하여 빠르게 적대적 노이즈를 생성하는 방법으로 손실 함수의 기울기를 따라 단 한 번의 스텝으로 노이즈를 생성하는 기법입니다.
PGD는 FGSM의 다중 스텝 버전으로 한 번이 아닌 여러 번 기울기를 업데이트하여 더 강력한 적대적 예제를 생성하는 공격 기법입니다.

$$
\begin{equation}
\delta(t+1) = \Pi \left[ \delta(t) + \alpha \cdot \text{sign}(\nabla \mathcal{L}(f(x + \delta(t)), y)) \right]
\end{equation}
$$


$$
\begin{equation}

\end{equation}
$$

해당 수식은 t+1번째 스텝에서의 적대적 노이즈를 업데이트하는 과정이며 각 스텝 t에서의 손실함수의 기울기 및 방향을 계산하고 알파만큼 이동하며 노이즈가 엡실론을 초과하지 않도록 수행합니다.

#### 적대적 공격 및 블랙박스 모델

FGSM와 PGD와 같은 간단한 알고리즘을 통해 다양한 분야에서 기계 학습 모델의 성능을 효과적으로 저하시킬 수 있습니다. 특히 베어링 결함 진단 시스템 역시 적대적 공격이 효과적임이 확인되었습니다.

화이트 박스 환경은 공격자가 훈련된 모델에 대한 모든 정보(모델 구조, 가중치, 기울기 정보)를 알고 있다고 가정하는 환경입니다. 
하지만 FGSM과  PGD는 결국 손실함수를 기반으로 기울기 정보를 활용하여 공격을 수행해야 하므로 모델 f의 내부 정보를 알아야 효과적으로 수행할 수 있습니다. 

따라서 **블랙박스 환경**에서의 공격을 수행해야 합니다. 일반적으로 훈련된 모델은 공개되지 않는 경우가 더 많으며 특히 산업용 결함 진단 시스템의 경우 모델이 직접 접근 가능한 형태로 제공되지 않으며 IoT 장치를 통해 입력을 제공하는 방식으로만 접근할 수 있습니다. 

블랙박스에서의 적대적 공격은 전이 공격(transfer attack)의 방식으로 진행되며 다음과 같습니다. 

$$
\begin{equation}
f(x') \neq y, \quad x' = x + \arg\max \mathcal{L}(g(x + \delta), y)
\end{equation}
$$

이때 g는 소스 모델로 하여 공격자가 접근할 수 있는 공개된 모델 또는 스스로 훈련한 모델이며 특정 회사의 산업용 고장 진단 모델을 목표로 한다면 해당 도메인의 공개된 신경망 모델을 소스 모델로 사용할 수 있습니다.  

**소스 모델에서 적대적 노이즈를 최적화**하여 소스 모델의 손실을 최대화하여 적대적 예제를 생성하여 이를 target model 인 f에 입력합니다.

화이트박스 환경에서는 모델의 강건성을 정확하게 측정할 수 있으나 블랙박스에서는 직접적으로 불가능하므로 1-P(P=공격 성공률)을 근사값으로 사용하였습니다. 하지만 기존의 FGSM, PGD 기법을 단순히 블랙박스 환경에 적용할 경우 실질적인 강건성을 과대평가하게 되어 **실제 환경에서 목표 모델이 얼마나 취약한지 정확하게 평가하기 어렵습니다.**

#### 도메인 특화 정보를 활용한 적대적 공격

블랙박스 환경에서의 **도메인 특화 정보**를 반영한 적대적 공격이 효과적이라는 연구가 기존 연구에서 입증되었습니다. 특히 다음 수식과 같이 적대적 예제 최적화 과정에서 공통적인 시각적 변환 T를 적용하면 공격 성능이 향상됩니다.

$$
\begin{equation}
\max \mathcal{L}(g(T(x + \delta)), y)
\end{equation}
$$

또한 **음향 도메인 지식**을 활용하여 공격 과정에서 연속적인 노이즈를 추가하면 공격 성능이 향상되는 것을 보입니다.  
이처럼 다양한 연구에서 블랙박스 환경의 적대적 공격이 연구되고 있으나 산업용 결함 진단 시스템에서의 블랙박스 환경의 실용적인 강건성을 평가한 연구는 현재 존재하지 않습니다.
따라서 본 논문은 적대적 공격을 최적화하는 과정에서 **spectrum** 변환을 적용하는 새로운 공격 기법을 제안합니다. 


<div class="row mt-3 text-center justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/students/2025-03-15-robustness-of-fault-diagnosis-systems/picture_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Spectrogram of Normal and Fault signal
</div>


다음 그림과 같이 베어링 결함 진단 시스템에서 spectrum 정보는 강력한 도메인 특성을 제공하며 이를 통해 기존보다 정밀한 모델의 강건성을 평가할 수 있습니다. 
본 논문은 블랙박스 환경에서의 적대적 예제 생성 과정에서 spectrum 정보를 조작하는 새로운 정규화 기법을 다음 수식처럼 제안하고 있습니다.

$$
\begin{equation}
\mathcal{L}_{SC}(x, x^*, \delta) = 1 - \cos(\text{Spec}(x + \delta), \text{Spec}(x)) 
+ \max\left( 0, \cos(\text{Spec}(x + \delta), \text{Spec}(x^*)) - \gamma \right)
\end{equation}
$$

해당 수식은 코사인 임베딩 손실 함수(cosine embedding loss function)로 소스 모델의 노이즈와 원본 모델의 cosine 유사도가 최소가 되며 소스 모델의 노이즈와 목표 모델의 노이즈의 코사인 유사도가 최대가 되도록 최적화합니다. 또한 다음 수식과 같이 크로스 엔트로피와 코사인 임베딩 손실을 결합하여 적대적 노이즈를 업데이트합니다.

$$
\begin{equation}
\
\delta^{(t+1)} = \Pi_{\|\delta\| \leq \epsilon} \left[ \delta^{(t)} + \alpha \cdot \text{sign} \left( \nabla_{\delta} \left( \mathcal{L}_{CE}(g(x), y) + \beta \mathcal{L}_{SC}(x, x^*, \delta) \right) \right) \right]
\
\end{equation}
$$

#### Spectrogram 기반 앙상블 기법

spectrogram은 코사인 임베딩 손실에 강한 영향을 미치므로 최적화 과정에서 중요한 역할을 합니다. 본 논문은 추가적으로 윈도우 크기가 적대적 노이즈 생성 성능에 미치는 영향을 추가적으로 분석합니다. 
이때 윈도우는 주파수 해상도(frequency resolution)와 시간-주파수 특성 간의 균형을 결정하며 윈도우가 작은 경우 시간 해상도(time resolution)는 높아지나 주파수 정보가 손실되며 윈도우가 큰 경우는 주파수 해상도는 높아지나 시간 정보가 희생되는 트레이드오프가 발생합니다.


<div class="row mt-3 text-center justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/students/2025-03-15-robustness-of-fault-diagnosis-systems/picture_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Window and Attack Performance
</div>


다음 그림은 다양한 윈도우 크기에 따른 소스 모델과 목표 모델 조합의 적대적 공격의 성공률을 분석하였습니다. 
윈도우가 너무 작은 경우는 오히려 PGD보다도 낮은 공격 성능을 보이나 소스 모델이 WDCNN(Wide deep Convolutional Neural Network), 목표 모델이 CNN인 경우는 윈도우 크기가 128인 경우가 가장 높은 공격 성능, 소스 모델이 CNN, 목표 모델이 MLP(Multi-layer Perception)인 경우는 윈도우 크기가 1024인 경우가 가장 높은 공격 성능을 보이고 있습니다.
즉, **최적의 w값은 소스 모델과 목표 모델의 조합에 따라 다르다**는 것을 알 수 있습니다.
따라서 본 논문은 w값을 단일 값으로 고정하는 것이 아닌 여러 개의 w를 동시에 고려하는 방법을 제안하고 있습니다.

$$
\begin{align}
\mathcal{L}_{SC}(x, x^*, \delta; w) := 1 & -\cos(\text{Spec}(x + \delta; w), \text{Spec}(x; w)) 
\\&+ \max \left( 0, \cos(\text{Spec}(x + \delta; w), \text{Spec}(x^*; w)) - \gamma \right)
\end{align}
$$

다음 수식과 같이 spectrogram 정보와 함께 윈도우 정보를 함께 활용하여 각 윈도우 크기에 따라 원본 모델과 소스 모델의 차이를 최대화하면서 목표 모델과의 차이를 최소화하는 방향으로 최적화합니다. 
이처럼 **다중 윈도우 크기를 활용**하는 방식을 통해  본 논문은 SAEM(Spectrogram-aware Ensemble Method)으로 명명하며 **코사인 임베딩 손실**을 기반으로 도메인 지식을 활용하며 **크로스 엔트로피** 손실과 결합해 모델을 속이는 최적의 적대적 신호를 생성합니다.

## Experiment
<hr>
본 논문은 실험을 통해 PGD 공격의 한계와 SAEM이 블랙박스 환경에서 뛰어난 공격 성능을 달성한다는 것을 보입니다.

#### Results
다음 실험은 목표 모델의 조합에 따른 SAEM과 PGD의 공격 성공률을 계산한 실험입니다. 


<div class="row mt-3 text-center justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/students/2025-03-15-robustness-of-fault-diagnosis-systems/picture_4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Loss of PGD and SAEM
</div>


해당 그래프를 보면 SAEM는 스텝 수가 증가함에 따라 공격 성공률이 증가하고 있으며 PGD에 비해 월등히 높은 공격 성공률을 보이고 있습니다.

다음 실험은 PGD와 SAEM으로 생성된 적대적 예제와 원본 예제의 spectrum 정보의 차이가 최대화 되었는지의 여부를 실험하였습니다.


<div class="row mt-3 text-center justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/students/2025-03-15-robustness-of-fault-diagnosis-systems/picture_5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Distance of Spectrum
</div>


해당 그래프를 보면 SAEM는 적대적 예제와 원본 예제 사이의 spectrum 거리가 더 빠르게 증가하는 모습을 보이며 PGD는 spectrum 정보를 최대화하는데 실패하는 것을 볼 수 있습니다. 
또한(c)를 보면 SAEM은 spectrum manifold 상에서 목표 모델과 소스 모델의 거리를 성공적으로 최소화한 것을 볼 수 있습니다.

다음 실험은 Spectrogram 앙상블 조합에 따른 모델의 공격 성능을 비교한 실험입니다.


<div class="row mt-3 text-center justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/students/2025-03-15-robustness-of-fault-diagnosis-systems/picture_6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Spectrogram Ensemble and Attack Performance
</div>


해당 그래프를 보면 w가 [128, 1024] 조합이 가장 높은 공격 성공률을 보이며 평균적으로 작은 윈도우 크기와 큰 윈도우 크기를 결합하는 것이 악의적인 적대적 예제를 생성하는데 유리하다는 것을 알 수 있다.

## 결론
<hr>

SAEM은 블랙박스 환경에서 뛰어난 공격 성능을 달성합니다. 추가적으로 산업용 결함 진단 시스템의 실무 환경을 탐색하며 산업 환경에서 발생할 수 있는 잠재적 위험 요소를 식별할 필요가 있습니다. 또한 spectrum 정보를 활용한 방어 메커니즘 연구를 통해서 고장 진단 시스템의 강건성을 향상시키는 연구 및 훈련 데이터셋의 접근성을 포함한 실무 환경에서의 연구가 필요할 것으로 보입니다.


