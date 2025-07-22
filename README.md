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


**MLaaS (Machine Learning as a Service)** 는 머신러닝 모델 개발, 학습, 배포, 관리를 클라우드 플랫폼(cloud platform)에서 제공하는 서비스입니다. MLaaS 덕분에 기업과 개인은 고성능 머신러닝 모델을 저비용으로 쉽게 활용할 수 있습니다.
그러나 MLaaS의 모델은 **하이퍼파라미터 도용 (Hyperparameter Stealing)** 위험에 노출될 수 있습니다.

하이퍼파라미터는 모델 성능을 결정하는 핵심 요소이며, 다양한 알고리즘에서 모델의 학습 방향을 조정합니다. 하이퍼파라미터를 학습하는 과정은 고비용을 수반하기 때문에 **상업적 기밀**로써 간주됩니다.
하지만 사용자가 학습 데이터와 모델 파라미터를 MLaaS와 공유하는 과정에서, 악의적 사용자가 하이퍼파라미터를 추출할 수 있습니다. 그리고 이는 MLaaS의 비용 구조를 악용하여 고성능 모델을 저비용으로 학습하려는 시도로 이어질 수 있습니다. 이러한 보안 위협은 MLaaS의 상업적 가치를 저해할 수 있으므로, 이에 대한 체계적인 분석과 대응이 필요합니다.

이러한 배경 속에서 본 논문은 하이퍼파라미터 도용이라는 상업적 위협을 분석하고 방어 기법을 제안합니다. 본 연구는 Amazon Machine Learning과 같은 실제 MLaaS 플랫폼에서 다양한 알고리즘을 대상으로 하이퍼파라미터 도용 공격을 실험하고 그 효과를 입증합니다.
나아가 모델 파라미터 **반올림 (Rounding)** 방어 기법의 효과와 한계를 평가하여 새로운 대응책의 필요성까지 제안합니다.

## 사전지식
<hr>

#### 하이퍼파라미터 vs 모델 파라미터

**하이퍼파라미터**(Hyperparameter)는 머신러닝 모델의 학습 과정을 조정하는 변수로, 모델 성능에 직접적인 영향을 미칩니다.
예를 들어, 다음 수식에서 하이퍼파라미터 $\lambda$는 목적 함수$\mathcal{L}(\mathbf{w})$에서 손실 함수와 정규화 항의 균형을 조정하여 과적합을 방지합니다.
$$
\begin{equation}
\mathcal{L}(\mathbf{w}) = \mathcal{L}_{\text{loss}}(\mathbf{w}) + \lambda \mathcal{R}(\mathbf{w})
\end{equation}
$$

**모델 파라미터**(Model Parameter)는 학습 과정에서 목적 함수를 최소화하여 결정된 값으로, 기울기 $\nabla \mathcal{L}(\mathbf{w}) = 0$를 만족하는 지점에서의 파라미터를 의미합니다. 모델 파라미터의 표현방식은 목적 함수의 종류에 따라  달라집니다. 


#### 비커널 알고리즘 vs 커널 알고리즘

**비커널 알고리즘**(Non-Kernel Algorithm)은 선형 결합 ($\mathbf{w}^\top \mathbf{x}$)을 사용하는 알고리즘입니다.
목적 함수$\mathcal{L}(\mathbf{w})$가 비커널 알고리즘이면 모델 파라미터는 $\mathbf{w}$로 표현됩니다.

$$
\begin{equation}
\mathcal{L}(\mathbf{w})=\mathcal{L}(X,y,\mathbf{w})+\lambda\,\mathcal{R}(\mathbf{w})
\end{equation}
$$

**커널 알고리즘**(Kernel Algorithm)은 데이터 $\mathbf{x}_i$를 비선형 고차원 공간으로 매핑 $\phi(\mathbf{x}_i$)하는 알고리즘입니다.
이때의 목적 함수$\mathcal{L}(\mathbf{w})$는 다음과 같습니다.
$$
\begin{equation}
\mathcal{L}(\mathbf{w})=\mathcal{L}\bigl(\phi(X),y,\mathbf{w}\bigr)+\lambda\,\mathcal{R}(\mathbf{w})
\end{equation}
$$

위 수식처럼 $\mathcal{L}(\mathbf{w})$가 커널 알고리즘이고 노름 정규화를 포함한다면, **Representer theorem**을 적용할 수 있습니다.
Representer theorem에 따르면, 최적화된 모델 파라미터 $\mathbf{w}$는 $\phi(\mathbf{x}_i)$의 선형 결합으로 표현됩니다. 따라서 커널 알고리즘일 때는 실제로 최적화해야 대상이 $\mathbf{w}$에서 $\alpha$로 대체됩니다. 그 결과, $\alpha$도 $\mathbf{w}$와 동등하게 모델 파라미터로 사용됩니다.

$$
\begin{equation}
\mathbf{w} = \sum_i \alpha_i \phi(x_i)
\end{equation}
$$

|목적 함수| 모델 파라미터 | 예시 |
|---|---|---|
|**비커널 알고리즘**   | $\mathbf{w}$  | RR(Ridge Regression), LASSO, LR(Logistic Regression), SVM(Support Vector Machine)        |
| **커널 알고리즘**    | $\mathbf{w}= \sum_i \alpha_i \phi(x_i)$ 또는 $\alpha$ | KLR(Kernel Logistic Regression), KSVM(Kernel Support Vector) |


#### Threat Model

공격 대상인 알고리즘은 다음과 같습니다. Linear regression, Kernel regression, Linear classification, Kernel classfication, Neural networks

공격자는 학습 데이터셋($X, y$), 학습 알고리즘, 그리고 모델 파라미터($\mathbf{w}$ 또는 $\alpha$)를 알고 있다고 가정합니다. 만약 모델 파라미터가 알려지지 않았다면, 기존 연구에서 제안된 모델 파라미터 도용 기법을 활용한다고 가정합니다.



#### 하이퍼파라미터$\lambda$ 추정

공격자는 목적 함수의 기울기가 0일 때를 선형 방정식으로 설정하고, 이를 만족시키는 하이퍼파라미터 $\lambda$를 계산합니다.

**비커널 목적 함수**$\mathcal{L}(\mathbf{w})$에서 $\lambda$ 계산
$$
\begin{equation}
  \frac{\partial \mathcal{L}_{\text{loss}}(X,\,y,\,\mathbf{w})}{\partial \mathbf{w}}
  \;+\;
  \lambda\,\frac{\partial \mathcal{R}(\mathbf{w})}{\partial \mathbf{w}}
  \;=\; 0
\end{equation}
$$

$$
\begin{equation}
\mathbf{b} \;=\;
\begin{bmatrix}
  \dfrac{\partial \mathcal{L}(X,y,\mathbf{w})}{\partial w_1} \\
  \dfrac{\partial \mathcal{L}(X,y,\mathbf{w})}{\partial w_2} \\
  \vdots \\[2pt]
  \dfrac{\partial \mathcal{L}(X,y,\mathbf{w})}{\partial w_m}
\end{bmatrix},
\qquad
\mathbf{a} \;=\;
\begin{bmatrix}
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial w_1} \\
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial w_2} \\
  \vdots \\[2pt]
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial w_m}
\end{bmatrix}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}}
  \;=\;
  \mathbf{b} \;+\; \lambda\,\mathbf{a}
  \;=\; 0
\end{equation}
$$

**커널 목적 함수**$\mathcal{L}(\alpha)$에서 $\lambda$ 계산

$$
\begin{equation}
  \frac{\partial \mathcal{L}_{\text{loss}}\bigl(\phi(\mathbf{X}),\,y,\,\mathbf{w}\bigr)}
       {\partial \boldsymbol{\alpha}}
  \;+\;
  \lambda\,
  \frac{\partial \mathcal{R}(\mathbf{w})}
       {\partial \boldsymbol{\alpha}}
  \;=\; 0
\end{equation}
$$

$$
\mathbf{b} \;=\;
\begin{bmatrix}
  \dfrac{\partial \mathcal{L}\bigl(\phi(X),y,\mathbf{w}\bigr)}{\partial \alpha_1} \\
  \dfrac{\partial \mathcal{L}\bigl(\phi(X),y,\mathbf{w}\bigr)}{\partial \alpha_2} \\
  \vdots \\[2pt]
  \dfrac{\partial \mathcal{L}\bigl(\phi(X),y,\mathbf{w}\bigr)}{\partial \alpha_n}
\end{bmatrix},
\qquad
\mathbf{a} \;=\;
\begin{bmatrix}
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial \alpha_1} \\
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial \alpha_2} \\
  \vdots \\[2pt]
  \dfrac{\partial \mathcal{R}(\mathbf{w})}{\partial \alpha_n}
\end{bmatrix}
$$

$$
\begin{equation}
\frac{\partial \mathcal{L}(\alpha)}{\partial \alpha}
  \;=\;
  \mathbf{b} \;+\; \lambda\,\mathbf{a}
  \;=\; 0
\end{equation}
$$

<!-- **다중 하이퍼파라미터**일 때는 $\mathbf{a}$를 행렬으로 변환한 후 계산합니다.

$$
\begin{equation}

  \mathbf{b}
  \;+\;
  \lambda_{1}\mathbf{a}_{1}
  \;+\;
  \lambda_{2}\mathbf{a}_{2}
  \;+\;
  \lambda_{3}\mathbf{a}_{3}
  \;=\; 0.
\end{equation}
$$

$$
\begin{equation}
A \;=\;[\,\mathbf{a}_{1}\; \mathbf{a}_{2}\; \mathbf{a}_{3}\,]\;\in\;\mathbb{R}^{d\times 3},
\qquad
\boldsymbol{\lambda}\;=\;
\begin{bmatrix}
  \lambda_{1}\\[2pt]
  \lambda_{2}\\[2pt]
  \lambda_{3}
\end{bmatrix}
\;\in\;\mathbb{R}^{3\times 1}.
\end{equation}
$$

$$
\begin{equation}
\mathbf{b} \;+\; A\boldsymbol{\lambda} \;=\; 0
\end{equation}
$$ -->


$\lambda$를 계산하기 위해 필요한 $\mathbf{a}$, $\mathbf{b}$는 목적함수의 종류에 따라 다르게 계산됩니다.
만약 $\lambda$의 개수보다 선형 방정식의 개수가 더 많은 경우(Over‑determined)라면, $\lambda$를 계산하기 어렵습니다. 이때는 최소자승법(Least-Squares solution)을 사용해 $\hat{\lambda}$를 추정합니다.

$$
\begin{equation}
\hat{\lambda} \;=\; -\,(\mathbf{a}^\top \mathbf{a})^{-1}\,\mathbf{a}^\top \mathbf{b}
\end{equation}
$$



<!-- $$
\begin{align}
\Delta\hat{\lambda}
  &= \hat{\lambda}-\lambda
     = \Delta\mathbf{w}^{\top}\,
       \nabla\hat{\lambda}\!\bigl(\mathbf{w}^{\star}\bigr)
     + O\!\bigl(\lVert\Delta\mathbf{w}\rVert_{2}^{2}\bigr)
\end{align}
$$

$$
\begin{align}
\Delta\hat{\lambda}
  &= \hat{\lambda}-\lambda
     = \Delta\boldsymbol{\alpha}^{\top}\,
       \nabla\hat{\lambda}\!\bigl(\boldsymbol{\alpha}^{\star}\bigr)
     + O\!\bigl(\lVert\Delta\boldsymbol{\alpha}\rVert_{2}^{2}\bigr)
\end{align}
$$

$$
\begin{align}
\hat{\lambda}-\lambda
    &= \Delta\hat{\lambda} \rightarrow 0
\end{align}
$$ -->


추정한 $\hat{\lambda}$에 대해서는 **상대추정오차**(Relative estimation error) $\epsilon$를 지표삼아 공격자의 하이퍼파라미터 추정 성능을 평가합니다.

$$
\begin{equation}
\epsilon \;=\; \frac{\lvert \hat{\lambda}-\lambda \rvert}{\lambda}
\end{equation}
$$



#### MLaaS에서 하이퍼파라미터 도용 방법

머신러닝은 기본적으로 아래의 과정을 따릅니다.
![alt text](image-1.png)

공격자는 이 과정 속에서 두 가지 Protocol을 활용해 하이퍼파라미터를 도용합니다.

**Protocol 1**: 사용자가 전체 학습 데이터셋과 학습 알고리즘을 MLaaS 플랫폼에 업로드하면, MLaaS는 하이퍼파라미터와 모델 파라미터를 학습한 뒤 모델 파라미터를 반환합니다.

**Protocol 2**: 사용자가 전체 학습 데이터셋과 학습 알고리즘, 하이퍼파라미터를 MLaaS 플랫폼에 업로드하면, MLaaS는 모델 파라미터를 학습합니다.

공격자는 이 Protocol을 다음과 같은 방식으로 활용합니다.
먼저, 공격자는 데이터셋 일부만 Protocol 1에 적용해 모델 파라미터를 얻고, MLaaS가 학습한 하이퍼파라미터를 추정합니다.
그 다음, 데이터셋 전체와 추정한 하이퍼파라미터를 Protocol 2에 적용해 저비용으로 고성능 모델을 학습합니다. 이 과정에서 공격자는 고비용의 하이퍼파라미터 학습 단계를 생략할 수 있습니다.
본 논문은 이 과정을 **Train-Steal-Retrain** 전략으로 부르며, 이를 MLaaS 플랫폼의 비용 구조를 악용하는 방식으로 설명합니다.


##### 방어 방법

본 논문이 소개하는 하이퍼파라미터 도용 방어 기법은 다음과 같습니다.
MLaaS가 Protocol 1에서 학습한 모델 파라미터를 사용자에게 반환할 때, 반올림한 값으로 제공하게 합니다. 이 방식은 공격자의 하이퍼파라미터 추정 오류를 증가시켜 도용 성능을 낮추는 데 효과적입니다.


## Experiment
<hr>

아래 실험은 여러 목적 함수의 하이퍼파라미터를 추정하고, 실제 하이퍼파라미터와의 상대추정오차를 분석합니다.
나아가 MLaaS 플랫폼에서의 하이퍼파라미터 추정 성능까지 분석합니다.
또한 반올림 기반 방어법의 성능과 한계도 밝힙니다.

#### 학습 데이터셋
| 이름 | 샘플 수 | 차원 | 유형 | 내용 |
|---|---|---|---|--|
|Diabetes|442|10|Regression|질병 진행 예측|
|GeoOrig|1059|68|Regression|지리적 변수 예측|
| UJIIndoor| 19937 | 529 | Regression |실내 위치 예측|
|Iris|100|4|Classification|붓꽃 분류|
|Madelon|4400|500|Classification| 이진 분류|
|Bank|45210|16|Classification| 정기예금 가입 예측|


#### 실험결과

**Regression**
| 목적 함수 | 손실 함수 | 정규화 |
|--|--|--|
| RR    | Least square | L2 | 
| LASSO | Least square | L1 | 
| KRR   | Least square | L2 |

![alt text](image-6.png)

위 그래프를 보면 목적함수가 RR, KRR일 때 상대추정오차가 비교적 작습니다. 이는 두 알고리즘이 학습한 모델 파라미터가 **해석적 해**(Analytical solutions)이기 때문입니다.
이처럼 모델 파라미터 $\mathbf{w}$ 또는 ${\alpha}$가 목적함수의 정확한 최소지점에서의 결과라면, 공격자가 추정한 하이퍼파라미터$\hat{\lambda}$는 실제$\lambda$와 일치하게 됩니다.
반면 다른 알고리즘들은 모델 파라미터에 대한 해석적 해가 없기 때문에 상대추정오차가 비교적 큽니다.

**Logistic Regression**
| 목적 함수 | 손실 함수 | 정규화 |
|--|--|--|
| L2-LR | Cross Entropy | L2 |
| L1-LR | Cross Entropy | L1 |
| L2-KLR | Cross Entropy | L2 |
| L1-KLR | Cross Entropy | L1 |

![alt text](image-7.png)



**SVM**
| 목적 함수 | 손실 함수 | 정규화 |
|--|--|--|
| SVM-RHL | Regular Hinge Loss | L2 |
| SVM-SHL | Square Hinge Loss  | L2 |
| KSVM-RHL | Regular Hinge Loss| L2 |
| KSVM-SHL | Square Hinge Loss | L2 |

![alt text](image-8.png)

**Neural Netwok**
| 목적 함수 | 손실 함수 | 정규화 |
|--|--|--|
| Regression     | Least Square  | L2|
| Classification | Cross Entropy | L2|

![alt text](image-9.png)






위 그래프를 보면 각각 

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


