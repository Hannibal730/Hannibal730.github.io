[---
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

**하이퍼파라미터**는 머신러닝 모델의 학습 과정을 조정하는 변수로, 모델 성능에 직접적인 영향을 미칩니다. 예를 들어, 정규화 파라미터 $\lambda$는 목적 함수$\mathcal{L}(\mathbf{w})$에서 손실 함수와 정규화 항의 균형을 조정하여 과적합을 방지합니다.
$$
\begin{equation}
\mathcal{L}(\mathbf{w}) = \mathcal{L}_{\text{loss}}(\mathbf{w}) + \lambda \mathcal{R}(\mathbf{w})
\end{equation}
$$

**모델 파라미터**는 학습 과정에서 목적 함수를 최소화하여 결정된 값으로, 기울기 $\nabla \mathcal{L}(\mathbf{w}) = 0$를 만족하는 지점에서의 파라미터를 의미합니다. 이때 목적 함수의 종류에 따라서 모델 파라미터의 표현방식이 달라집니다. 


#### 비커널 알고리즘 (Non-Kernel Algorithm) vs 커널 알고리즘 (Kernel Algorithm)

**비커널 알고리즘**은 선형 결합 ($\mathbf{w}^\top \mathbf{x}$)을 사용하는 알고리즘입니다. 목적 함수$\mathcal{L}(\mathbf{w})$가 비커널 알고리즘이면 모델 파라미터는 $\mathbf{w}$로 표현됩니다.

$$
\begin{equation}
\mathcal{L}(\mathbf{w})=\mathcal{L}(X,y,\mathbf{w})+\lambda\,\mathcal{R}(\mathbf{w})
\end{equation}
$$

**커널 알고리즘**은 데이터 $\mathbf{x}_i$를 비선형 고차원 공간으로 매핑 $\phi(\mathbf{x}_i$)하는 알고리즘입니다. 이때의 목적 함수$\mathcal{L}(\mathbf{w})$는 다음과 같습니다.
$$
\begin{equation}
\mathcal{L}(\mathbf{w})=\mathcal{L}\bigl(\phi(X),y,\mathbf{w}\bigr)+\lambda\,\mathcal{R}(\mathbf{w})
\end{equation}
$$

이 목적 함수처럼 $\mathcal{L}(\mathbf{w})$가 커널 알고리즘이고 노름 정규화를 포함한다면, **Representer theorem**을 적용할 수 있습니다. Representer theorem에 따르면 최적화된 모델 파라미터 $\mathbf{w}$는 $\phi(\mathbf{x}_i)$의 선형 결합으로 표현됩니다.

$$
\begin{equation}
\mathbf{w} = \sum_i \alpha_i \phi(x_i)
\end{equation}
$$

따라서 커널 알고리즘일 때는 실제로 최적화해야 대상이 $\mathbf{w}$에서 $\alpha$로 대체됩니다. 그 결과, $\alpha$도 $\mathbf{w}$와 동등하게 모델 파라미터로 사용된다.



|목적 함수| 모델 파라미터 | 예시 |
|---|---|---|
|**비커널 알고리즘**   | $\mathbf{w}$  | RR(Ridge Regression), LASSO, LR(Logistic Regression), SVM(Support Vector Machine)        |
| **커널 알고리즘**    | $\mathbf{w}= \sum_i \alpha_i \phi(x_i)$ 또는 $\alpha$ | KLR(Kernel Logistic Regression), KSVM(Kernel Support Vector) |


#### 위협 모델 (Threat Model)

공격 대상은 다음과 같습니다. Linear regression, Kernel regression, Linear classification, Kernel classfication, Neural networks

공격자는 학습 데이터셋($X, y$), 학습 알고리즘, 그리고 주로 모델 파라미터($\mathbf{w}$ 또는 $\alpha$)를 안다고 가정합니다. 만약 모델 파라미터가 알려지지 않은 경우, 기존 연구에서 제안된 모델 파라미터 도용 기법을 활용한다고 가정합니다.

| 학습 데이터셋 | 샘플 수 | 차원 | 유형 | 내용 |
|---|---|---|---|--|
|Diabetes|442|10|Regression|질병 진행 예측|
|GeoOrig|1059|68|Regression|지리적 변수 예측|
| UJIIndoor| 19937 | 529 | Regression |실내 위치 예측|
|Iris|100|4|Classification|붓꽃 분류|
|Madelon|4400|500|Classification| 이진 분류|
|Bank|45210|16|Classification| 정기예금 가입 예측|


#### 하이퍼파라미터$\lambda$ 도용 방법

목적 함수의 기울기가 0일 때를 선형 방정식으로 설정하고, 이를 만족시키는 하이퍼파라미터 $\lambda$를 계산합니다.

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
\frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}}
  \;=\;
  \mathbf{b} \;+\; \lambda\,\mathbf{a}
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
\end{bmatrix}.
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
\begin{equation}
\frac{\partial \mathcal{L}(\alpha)}{\partial \alpha}
  \;=\;
  \mathbf{b} \;+\; \lambda\,\mathbf{a}
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

이때 $\lambda$의 개수보다 선형 방정식의 개수가 더 많은 Over‑determined 경우라면, $\lambda$를 계산하기 어렵습니다. 이때는 최소자승법(Least-Squares solution)으로 $\hat{\lambda}$를 추정합니다.

$$
\begin{equation}
\hat{\lambda} \;=\; -\,(\mathbf{a}^\top \mathbf{a})^{-1}\,\mathbf{a}^\top \mathbf{b}.
\end{equation}
$$

만약 하이퍼파라미터가 여러 개라면, 이를 행렬으로 변환한 후 계산합니다.

$$
\begin{equation}
\frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}}
  \;=\;
  \mathbf{b}
  \;+\;
  \lambda_{1}\mathbf{a}_{1}
  \;+\;
  \lambda_{2}\mathbf{a}_{2}
  \;+\;
  \lambda_{3}\mathbf{a}_{3}
  \;=\; 0
\end{equation}
$$

$$
\begin{equation}
A \;=\;[\,a_{1}\; a_{2}\; a_{3}\,]\;\in\;\mathbb{R}^{d\times 3},
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
$$


#### MLaaS에서 하이퍼파라미터 도용 방법

하이퍼파라미터 도용에는 두 가지 프로토콜(Protocol)이 활용됩니다.

**프로토콜 1**: 사용자가 전체 학습 데이터셋과 학습 알고리즘을 MLaaS 플랫폼에 업로드하면, MLaaS는 하이퍼파라미터와 모델 파라미터를 학습한 뒤 모델 파라미터를 반환합니다. 

**프로토콜 2**: 사용자가 전체 학습 데이터셋과 학습 알고리즘, 하이퍼파라미터를 MLaaS 플랫폼에 업로드하면, MLaaS는 모델 파라미터를 학습합니다.

**공격 전략**: 공격자는 데이터셋 일부만 프로토콜 1에 적용해 모델 파라미터를 얻고, 이를 사용해 하이퍼파라미터를 도용합니다. 이후 전체 데이터셋과 도용한 하이퍼파라미터를 프로토콜 2에 적용해 저비용으로 고성능 모델을 학습합니다(Train-Steal-Retrain 전략). 이는 고비용의 하이퍼파라미터 학습을 우회해 MLaaS 플랫폼의 비용 구조를 악용합니다.


##### 방어 방법

본 논문이 소개하는 하이퍼파라미터 도용 방어법은 다음과 같습니다. 프로토콜 1에서 MLaaS가 학습한 모델 파라미터를 사용자에게 반환할 때, 이를 반올림한 값을 제공하는 방법입니다. 이 방법은 공격자의 하이퍼파라미터 추정 오류를 증가시킵니다. 본 연구는 실험을 통해 이 기법의 효과와 한계를 제시합니다.
