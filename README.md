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

**모델 파라미터**는 학습 과정에서 목적 함수를 최소화하여 결정된 값으로, 기울기 $\nabla \mathcal{L}(\mathbf{w}) = 0$를 만족하는 지점에서의 파라미터를 의미합니다. 이때 목적함수의 종류에 따라서 모델 파라미터의 표현방식이 달라집니다. 


#### 비커널 알고리즘 (Non-Kernel Algorithm) vs 커널 알고리즘 (Kernel Algorithm)

**비커널 알고리즘**은 선형 결합 ($\mathbf{w}^\top \mathbf{x}$)을 사용하는 알고리즘입니다. 목적함수가 비커널 알고리즘이면 모델 파라미터는 $\mathbf{w}$로 표현됩니다. 
    
**커널 알고리즘**은 데이터 $\mathbf{x}_i$를 비선형 고차원 공간으로 매핑 $\phi(\mathbf{x}_i$)하는 알고리즘입니다. 목적함수가 아래처럼 커널 알고리즘이며 노름 기반 정규화 항을 사용한다면, **Representer theorem**을 사용할 수 있습니다.

$$
\begin{equation}
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \ell(f(\mathbf{x}_i), y_i) + \lambda R(\mathbf{w})
\end{equation}
$$

Representer theorem에 따르면 최적화된 모델 파라미터 $\mathbf{w}$는 학습 데이터 $\phi(\mathbf{x}_i)$의 선형 결합 ($\mathbf{w} = \sum_i \alpha_i \phi(x_i)$)으로 표현됩니다. 그 결과, 목적함수가 커널 알고리즘이면 모델 파라미터는 $\alpha$로 대체됩니다.

|목적 함수| 모델 파라미터 | 예시 |
|-----| ---- | ---- |
|**비커널**    | $\mathbf{w}$ | RR(Ridge Regression), LASSO, LR(Logistic Regression), SVM(Support Vector Machine)        |
| **커널**     | $\mathbf{w} = \sum_i \alpha_i \phi(x_i)$ | KLR(Kernel Logistic Regression), KSVM(Kernel Support Vector) |


#### 위협 모델 (Threat Model)

공격 대상: Linear Regression, Kernel Regression, Linear Classification, Kernel Classfication, Neural networks

공격자는 학습 데이터셋($X, y$), 학습 알고리즘, 그리고 주로 모델 파라미터($\mathbf{w}$ 또는 $\alpha$)를 알고 있다고 가정합니다. 

만약 모델 파라미터가 알려지지 않은 경우, 기존 연구에서 제안된 모델 파라미터 도용 기법을 활용한다고 가정합니다.

| 데이터셋 | 샘플 수 | 차원 | 유형 | 내용 |
|---|---|---|---|--|
|Diabetes|442|10|Regression|질병 진행 예측|
|GeoOrig|1059|68|Regression|지리적 변수 예측|
| UJIIndoor| 19937 | 529 | Regression |실내 위치 예측|
|Iris|100|4|Classification|붓꽃 분류|
|Madelon|4400|500|Classification| 이진 분류|
|Bank|45210|16|Classification| 정기예금 가입 예측|


#### 하이퍼파라미터 도용 메커니즘

공격은 학습 데이터와 모델 파라미터 간의 수학적 관계를 활용합니다. 머신러닝 알고리즘의 목적 함수는 일반적으로 $ \text{Loss}(w) + \lambda \|w\|_p $ 형태로, 모델 파라미터는 기울기 $ \nabla (\text{Loss}(w) + \lambda \|w\|_p) = 0 $를 만족합니다. 이를 통해 $ \lambda $를 포함한 선형 방정식을 구성하고, 최소제곱법으로 하이퍼파라미터를 추정합니다.
예: 리지 회귀에서는 $ X^T (y - Xw) = \lambda w $를 통해 $ \lambda $를 계산. 커널 알고리즘에서는 $ \alpha $를 사용해 유사한 방정식을 만듭니다.
L1 정규화(LASSO 등)에서는 비미분 가능 지점($ w = 0 $)을 제외하고 추정하며, 다중 하이퍼파라미터(예: Elastic Net의 $ \lambda_1, \lambda_2 $)는 행렬 방정식으로 해결합니다.
