---
{"dg-publish":true,"permalink":"/paper-review/generalization-in-diffusion-models-arises-from-geometry-adaptive-harmonic-representations/"}
---


## **요약**

딥 뉴럴 네트워크(DNN)는 이미지 디노이징을 학습한 후, 스코어 기반 역확산(reverse diffusion) 알고리즘을 사용하여 고품질의 샘플을 생성할 수 있다. 이는 고차원 공간에서의 **차원의 저주(curse of dimensionality)** 를 극복한 듯 보이지만, 최근 연구들은 훈련 데이터의 **암기(memorization)** 문제를 제기하며, DNN이 실제로 연속적인 데이터 밀도를 학습하는지에 대한 의문을 던진다.

본 연구에서는 **충분히 큰 데이터셋**에서 훈련된 두 개의 DNN이 **거의 동일한 스코어 함수**를 학습한다는 사실을 발견했다. 이는 모델이 특정 훈련 샘플을 암기하는 것이 아니라, **일반화(generalization)** 를 통해 실제 데이터 밀도를 학습했음을 의미한다. 이때 생성된 이미지는 원본 훈련 데이터와 구별되며, 높은 시각적 품질을 유지한다.

추가적으로, 연구진은 DNN 디노이저가 특정한 **귀납적 편향(inductive bias)** 을 갖고 있으며, 이는 이미지의 기하학적 구조에 맞춰 **기하 적응 조화 기저(GAHB, Geometry-Adaptive Harmonic Bases)** 를 형성하는 방식으로 나타난다는 것을 발견했다. 이러한 편향은 훈련 데이터가 사진 이미지이든, 저차원 다양체(manifold)에 놓인 이미지이든 일관되게 나타났다. 또한, 밴들렛(bandlets)과 같은 **최적 기저가 존재하는 이미지 클래스**에서는 DNN 디노이저가 거의 최적의 성능을 보였지만, GAHB가 최적 기저가 아닌 경우 성능이 저하되었다.

이러한 결과는 DNN이 특정한 Inductive bias을 통해 **고차원 공간에서 효과적으로 데이터 밀도를 학습할 수 있도록 유도되는 메커니즘**을 보여주며, 이는 확산 모델의 강력한 샘플링 성능을 설명하는 중요한 단서가 된다.

# Diffusion Model Variance and denoising Generalization

알려지지 않은 이미지 확률 밀도 함수 , $p(x)$
diffusion 모델은 바로 근사하는 것이 아니라 노이즈 섞인 이미지 분포의 ***score*** 를 학습함.
1. denoising error가 density modeling error 의 bound를 설정함 보이기
2. 이걸로 density model의 수렴을 분석할거임
## Diffusion model and Denoising

Let $y = x + z$ , where $z \sim \mathcal{N}(0, \sigma^2 \mathbf{Id})$
density of noise image,  ${p}_\sigma(y)$ related to $p(x)$ through marginalization over $x$.

###### (1)  $p_{\sigma}(y) = \int p(y|x) p(x) \,dx = \int g_{\sigma} (y - x) p(x) \,dx$

$g_\sigma(z)$ : density of $z$. 
$p_{\sigma}(y)$ 는 $p(y)$에 $\sigma$ 표준편차로 가우시안 적용  

$\{p_\sigma(y);\sigma \geq 0\}$ family는 scale-space representation of $p(x)$이며, diffusion 과정과 유사함.
	보통 2차원임. 
	주어진 이미지 $f(x,y)$ 에 대해 $L(x,y;t)$의 합성으로 정의되며,
	2차원 가우시안 커널을 이용해
	$g(x,y;t)=\frac{1}{2\pi t}e^{-(x^2+y^2)/2t}$    이때,   $L(\cdot,\cdot;t) = g(\cdot,\cdot;t)*f(\cdot,\cdot)$
	$t$는 scale param이고, 가우시안 필터의 $t=\sigma^2$임.
	![Scale_space.png](/img/user/blog%20asset/Scale_space.png)
	https://en.wikipedia.org/wiki/Scale_space



###### (2)  $D_{\text{KL}}(p(x) \| p_{\sigma}(x)) \leq \int_{0}^{\infty} \mathbb{E}_{y} \left[ \left\| \nabla \log p_{\sigma}(y) - s_{\theta}(y) \right\|^2 \right] \sigma \, d\sigma.$

Diffusion 모델은 모든 노이즈 수준 $\sigma$ 에서 $p_\sigma(y)$의 scores $\triangledown \log p_\sigma(y)$ 의 근사 $s_\theta (y)$ 를 학습함.
이 스코어 모델들은 결국 역확산을 통해 $p(x)$의 모델인 $p_\theta(x)$를 표현함.

###### (3)  $\nabla \log p_{\sigma} (y) = ({\mathbb{E}_{x} [x | y] - y})/{\sigma^2}$

유도임.
MIYASAWA Relationship.
자주나와서 $\sigma$ dependence 생략

score $\nabla \log p (y)$ , Jacobian $\nabla
{ #2}
 \log p (y)$
measurement  $p(y|x)$, posterior $p(x|y)$

노이즈 이미지의 확률분포는 다음과 같음.(**bayes' rule** 및 **marginalize**)
$p(y) = \int p(x)p(y|x)dx$

score를 구하기 위해 델 log p(x) 를 아래 식을 이용해 표현하면 ,  
(∇h(y) = h(y) ∇ log h(y) , ∇ log h(y) = (∇h(y))/h(y), h(y)=p(y))

$$
\begin{aligned}  
\nabla \log p(y) & =\int p(x) p(y \mid x) \nabla_y \log p(y \mid x) \mathrm{d} x / p(y) \\  
& =\int p(x \mid y) \nabla_y \log p(y \mid x) \mathrm{d} x \\  
& =\mathbb{E}\left[\nabla_y \log p(y \mid x) \mid y\right]  
\end{aligned}
$$ 
(14)   조건부 기대값


**y에 대해 미분 한번 더하면,** 
$\nabla^2 \log p(y) = \int p(x|y) \left( \nabla_y \log p(x|y) \nabla_y \log p(y|x)^{\top} + \nabla^2 \log p(y|x) \right) dx.$    (15)


**Bayes rule의 로그 버전 이용**
$\log p(x | y) = \log p(y | x) - \log p(y) + \log p(x),$ 
$\nabla_y \log p(x | y) = \nabla_y \log p(y | x) - \nabla_y \log p(y)$         (16)

(15)에 (16) 적용
$\nabla^2 \log p(y) = \int p(x|y) \left( (\nabla_y \log p(y|x) - \nabla_y \log p(y)) \nabla_y \log p(y|x)^{\top} + \nabla^2 \log p(y) \right) dx$
   $= \mathbb{E} \left[ (\nabla_y \log p(y|x) - \nabla_y \log p(y)) \nabla_y \log p(y|x)^{\top} \mid y \right] + \mathbb{E} \left[ \nabla^2 \log p(y|x) \mid y \right]$
   $= \text{Cov} \left[ \nabla_y \log p(y|x) \mid y \right] + \mathbb{E} \left[ \nabla^2 \log p(y|x) \mid y \right]$      (17)
   $\nabla \log p(y) = \mathbb{E} \left[ \nabla \log p(y|x) \mid y \right]$   이기 때문에.. 왜? y=x+z다.


근데 $y$ = $x$ + $\sigma^2$ Id 다. 가우시안으로다가:
$$
\begin{aligned}
\log p(y \mid x) & =-\frac{1}{2 \sigma^2}\|y-x\|^2+\mathrm{cst} \\
\nabla_y \log p(y \mid x) & =-\frac{1}{\sigma^2}(y-x) \\
\nabla_y^2 \log p(y \mid x) & =-\frac{1}{\sigma^2} \mathrm{Id}
\end{aligned}
$$
 (14) and (17) 는
$$
\begin{aligned}
\nabla \log p(y) & =\frac{1}{\sigma^2}(\mathbb{E}[x \mid y]-y) \\
\nabla^2 \log p(y) & =\frac{1}{\sigma^4} \operatorname{Cov}[x \mid y]-\frac{1}{\sigma^2} \mathrm{Id}
\end{aligned}
$$

Finally, the above identities can be rearranged to yield the first- and second-order Miyasawa relationships:
$$
\begin{aligned}
\mathbb{E}[x \mid y] & =y+\sigma^2 \nabla \log p(y) \\
\operatorname{Cov}[x \mid y] & =\sigma^2\left(\operatorname{Id}+\sigma^2 \nabla^2 \log p(y)\right)
\end{aligned}
$$

Note that the optimal denoising error satisfies
$$
\mathbb{E}\left[\|x-\mathbb{E}[x \mid y]\|^2\right]=\mathbb{E}\left[\mathbb{E}\left[\operatorname{tr}(x-\mathbb{E}[x \mid y])(x-\mathbb{E}[x \mid y])^{\mathrm{T}} \mid y\right]\right]=\mathbb{E}[\operatorname{tr} \operatorname{Cov}[x \mid y]]
$$
###### (4)  $\text{MSE}(f_{\theta}, \sigma^2) = \mathbb{E}_{x,y} \left[ \| x - f_{\theta}(y) \|^2 \right]$

###### (5)  asd
so that $\left.f_\theta(y) \approx \mathbb{E}_x|x| y\right]$. This estimated conditional mean is used to recover the estimated score using eq. (3): $s_\theta(y)=\left(f_\theta(y)-y\right) / \sigma^2$. As we show in Appendix D.2, the error in estimating the density $p(x)$ is bounded by the integrated optimality gap of the denoiser across noise levels:
$$
D_{\mathrm{KL}}\left(p(x) \| p_\theta(x)\right) \leq \int_0^{\infty}\left(\operatorname{MSE}\left(f_\theta, \sigma^2\right)-\operatorname{MSE}\left(f^{\star}, \sigma^2\right)\right) \sigma^{-3} \mathrm{~d} \sigma
$$
where $f^{\star}(y)=\mathbb{E}_x[x \mid y]$ is the optimal denoiser. Thus, learning the true density model is equivalent to performing optimal denoising at all noise levels. Conversely, a suboptimal denoiser introduces a score approximation error, which in turn can result in an error in the modeled density.
Generally, the optimal denoising function $f^{\star}$ (as well as the "true" distribution, $p(x)$ ) is unknown

## Transition From Memorization to Generalization

*DNN* 은 Overfitting 되기 쉬움. - 훈련 데이터의 개수가 모델 capacity에 비해 작기 때문.
또 차원의 저주(curse of dimensionality)로 인해, 생성형 모델에선 특히 문제가 있음.

Diffusion 모델이 memorization이 학계에 보고 되고 실험설계함.

Size N = 10^[0, 1, 2, 3, 4, 5]
3 composed conv 인코더 - 디코더 U-Net. Noise 레벨 Input 안받고 모든 노이즈 레벨서 연산.(universal blind)
![Pasted image 20250319233320.png](/img/user/blog%20asset/Pasted%20image%2020250319233320.png)
$PSNR = 10 \cdot\log_{10} \frac {MAX^2}{MSE}$
높을수록 좋음. 40db 되면 거의 원본과 비슷하다고 함.
train 보면 input PSNR에 비해 output이 엄청 높음 - memorization인거임. 근데 Test를 보면 성능 망해버림.
N이 1000일 때부터, 좀 바뀌더니, N에 100000일때는 empirical 하게 test와 train error가 모든 노이즈 레벨에서 동일함.

![Pasted image 20250319233342.png](/img/user/blog%20asset/Pasted%20image%2020250319233342.png)
이번에는 분리된 데이터셋(S1, S2)에서 훈련함. 
non-overlapping CelebA 데이터임에도
동일한 score 함소가 학습되었다는 걸 볼수 있음.
2,3 행은 생성된 것. 1,4행은 각 데이터셋에서 가상 유사한 데이터
2행, 3행 보고, 1,2 3,4 행 묶어보면됨. 

# Inductive Biases
임의의 확률 분포를 추정하는 데 필요한 샘플 개수가 지수적으로 증가한다.(curse of dimension)
-> 고차원 분포 추정은 가설공간에 강력한 제약이나 사전지식이 필요하다.
-> Inductive bias라고한다.
2.2로 적은 데이터 일반화에 성공했으니, 모델의 귀납적 편향이 이미지의 진짜 분포와 잘 부합한거다.(빠르게 좋은 해)
아니면, 높은 편향을 가진 안 좋은 해에 도달했을 거니까.

확산모델에서 옳은 확률밀도모델 학습은 모든 노이즈 수준에서의 최적의 디노이징과 동등하므로(2.1), density model의 inductive bias는 denoiser의 inductive bias로부터 직접 나타난다.
-> 고차원 공간에서의 어려운 확률모델의 정확도를 평가한다.
## Denoising As Shrinkage in and adaptive basis
DNN 디노이저의 inductive bias는 Jacobian 의 eigendecomposition로 연구되었음.
그럼 최적 디노이저의 general property를 부분적으로 알고있는 최적해의 몇 가지 특정 사례로 살펴보고자함.

#### Jacobian eigenvectors as an adaptive basis
local analysis of a denoising astimator $\hat{x} = f(y)$ by looking at its Jacobian $\nabla f(y)$ 

symmetric에 non-negative 가정(결국 최적 디노이저 특징이긴함.)
이거 diagonalize 해서 eigenvalues$(\lambda_k(y))_{1 \leq k  \leq d}$랑 eigenvectors$(e_k(y))_{1 \leq k  \leq d}$ 볼거임.

affine말고 linear하게 i/o 매핑된 DNN denoiser 계산하는 f(y)임.

$$
f(y)=\nabla f(y) y=\sum_k \lambda_k(y)\left\langle y, e_k(y)\right\rangle e_k(y)
$$

$$
\nabla f(y) = \begin{bmatrix} \frac{\partial f_1(y)}{\partial y_1} & \cdots & \frac{\partial f_1(y)}{\partial y_d} \\[6pt] \frac{\partial f_2(y)}{\partial y_1} & \cdots & \frac{\partial f_2(y)}{\partial y_d} \\[6pt] \vdots & \vdots & \vdots \\ \frac{\partial f_d(y)}{\partial y_1} & \cdots & \frac{\partial f_d(y)}{\partial y_d} \end{bmatrix}​​​
$$
야코비안은 다시 말하지만, 함수의 국소적인 변화를 나타내는 행렬이고
위 식에서는 비선형이든 뭐든간에 선형화 시켜서 보기위해(선형 근사) 쓴거다.
그리고 이걸 다시 고유벡터 기준으로 나타내어 쓴거다.
여기서 shrinkage factor가 바로 고유값(eigenvalue, $\lambda$)이다.  의미없는 ~ 작은 eigenvalue ~ 축소된다.

근데 잘생각해보면 이러한 작은 eigenvalue는 bias 있어도 정의 되긴한다. -> 국소적으로 불변성을 나타낸다(local invariance)ㅡ 입력데이터가 고유벡터 방향으로 흔들려도 무시된다 -> 노이즈 제거랑 직관적으로 통하죠? 
small eigenvalue가 매우 중요하다.
inductive bias 이해의 핵심적인 단서다.

또 자연스럽게 MSE를 줄이면서 얻어진다. Stein's unbiased risk estimate 형으로 표현하면 잘 보임.
$$
\operatorname{MSE}\left(f, \sigma^2\right)=\underset{y}{\mathbb{E}}\left[2 \sigma^2 \operatorname{tr} \nabla f(y)+\|y-f(y)\|^2-\sigma^2 d\right] .
$$
왜냐하면, denoiser는 Jacobian의 rank(tr은 eigenvalue들의 합이므로)와   디노이징 에러 추정(위식에서 뒤쪽항 두개)사이의 trade-off관계를 가지기 때문이다.
그러면 상상해봐라. 국소적으로 Jacobian의 rank따라 차원이 결정되는 부분공간에 입력을 soft하게 사영되는 식으로 행동할거다. 

이 부분공간은 posterior distribution $p(x|y)$의 support를 근사하는 영역인 것이며, 결국 p(x)의 support를 국소 근사하는 거다.


$$
\begin{aligned}
f^{\star}(y) & =y+\sigma^2 \nabla \log p_\sigma(y)=\underset{x}{\mathbb{E}}[x \mid y], \\
\nabla f^{\star}(y) & =\mathrm{Id}+\sigma^2 \nabla^2 \log p_\sigma(y)=\sigma^{-2} \operatorname{Cov}[x \mid y]
\end{aligned}
$$

최적 디노이저의 야코비안은 posterior 공분산 행렬(대칭 행렬, non-negative(고윳값 0보다큼))과 비례관계이다. 다시말해, 이 adaptive eigenvector들이 우리가 알 수 없는 x에 대한 optimal approximation basis를 제공하는 것이다.
-> 작은 eigenvalue 방향 - posterior의 공분산이 작음 - 해당 방향 변화는 이미지 변화 거의 없음
-> 큰 eigenvalue 방향 - posterior 공분산 큼 - 해당 방향 변화를 통해 의미있는 신호
$$
\operatorname{MSE}\left(f^{\star}, \sigma^2\right)=\underset{y}{\mathbb{E}}[\operatorname{tr} \operatorname{Cov}[x \mid y]]=\sigma^2 \underset{y}{\mathbb{E}}\left[\operatorname{tr} \nabla f^{\star}(y)\right]=\sigma^2 \underset{y}{\mathbb{E}}\left[\sum_k \lambda_k^{\star}(y)\right]
$$
야코비안 rank 낮을수록 국소적으로 의미있는 신호를 잘 보존한다는 것이고, 노이즈를 잘 제거한다는 것이다.


근데, 대부분의 경우 최적 적응 기저 $e_k^{\star}(y)_{1\leq k \leq d}$ 는 모르는 값이어서 classical 하게는 노이즈($\sigma^2$)가 작아질 때 디노이징 오차 감소 속도의 asymptotic decay를 봤다. 꼭 딱 들어맞지는 않지만 figure 1이 그렇다.
이러면 실제로 가장 좋은 기저를 정확히 구하지 않아도 디노이징 성능이 특정한 속도로 향상됨을 보인다.

y넣어서 고정된 기저 선택 후 고정된 dictionary  중 가장 좋은 기저 고르기 같이 같은 짓을 할 수 있는거지

#### Denoising in fixed basis.
고정된 basis $e_k$와 축소 계수 $\lambda_k(y)$를 생각해보자. 디노이징 에러의 하한 (PSNR slope의 상한) oracle denoiser의 성능 측정으로 획득된다.
$$
\underset{y}{\mathbb{E}} [\underset{k}\sum((1-\lambda_k(x))^2\left\langle x, e_k(x)\right\rangle^2) + \lambda_k(x)^2\sigma^2]
$$
가장 작아지는건  $\lambda_k(x) = \frac{\left\langle x, e_k\right\rangle^2}{\left\langle x, e_k\right\rangle^2 + \sigma^2}$
이러면 일종의 soft threshold처럼 기능하는데 1이면 살고 0이면 죽는다.(노이즈와 신호 비교했을 때)
아래는 이상적일 때 디노이징 에러다.

$$
\sigma^2 \underset{k}\sum \lambda_k(x) = \underset{k} \sum \frac{\sigma^2 \left\langle x, e_k\right\rangle^2}{\left\langle x, e_k\right\rangle^2 + \sigma^2} \backsim \underset{k}\sum \operatorname{min}(\left\langle x, e_k\right\rangle^2,\sigma^2) = M\sigma^2 + \|x - x_m\|^2
$$
where $x_M = \sum_{\left\langle x, e_k\right\rangle^2>\sigma^2}\left\langle x, e_k\right\rangle e_k$ 는 M항 근사임.
⟨x, ek ⟩ 에서 $\sigma^2$ 보다 큰놈들만 살린거  이게 오른쪽이랑 거의 동등하다는거임.(최대 2배)
근데 여기서 x가 sparse 하므로 M하고 approximation error도 작아짐/. 이게 뭔소리냐
$$
\left\langle x, e_k\right\rangle^2 \backsim k^{-(\alpha+1)} 를 따른다고 했을 때,
M\sigma^2 + \|x - x_M\|^2 \backsim \sigma^{2\alpha/(\alpha+1)}
$$
이꼴로 나타낼 수 있음.
디노이징 성능을 봤을 때, 입력 PSNR이 증가할 수록 MSE 감소율이 $\alpha/(\alpha + 1)$ 인거임.
sparsity/regularity exponent $\alpha$가 클수록 작은 계수가 더 빠르게 감소하면서 디노이징 성능향상으로 이어지는 거임.

#### Best adaptive bases
-> 최적의 적응 기저를 계산하기 위해 현실 디노이저와 MSE가 동일한 oracle 디노이저를 찾아야 asymptotic MSE의 상한과 하한이 일치하게된다. 
최적의 오라클 기저는 $x/\|x\|$겠지만 이건 못구하는 거고, 얘도 아까 가정 넣으면 여전히 점근 PSNR 기울기는 $\alpha/(\alpha+1)$이다. 그래서 ($e_k(x)$)는 고정 사전에서 제한해야한다. dictionary 가 커지면 적응을 잘하겠지만, 노이즈가 있는 y의 최적기저 추정이 어려워 질거고, 기저 개수는 d에 따라  polynomial 하게 구성되면, bases의 개수는 지수적으로 증가하고(d에 따라.) 오라클 디노이저와 같은 기울기를 달성한다. 

(차원 d에대해 polynormail하게 생기는)$e_k$ dictionary에서 threshold로 잘라 쓰면, oracle 디노이저 같은 기울기를 달성하는 거다. 여기서 사전에 있는 기저 벡터의 개수가 제한되어있으므로, 최적기저 추정의 변동이 제한된다.(있는 거에서 고르니까)

이제 최적 PSNR 경사를 위해 작으면서도 최적 이미지 표현 기저 사전을 만드는게 문제다

## Geometry-adaptive Harmonic bases in DNNs

![Pasted image 20250320210142.png](/img/user/blog%20asset/Pasted%20image%2020250320210142.png)
![Pasted image 20250320212219.png](/img/user/blog%20asset/Pasted%20image%2020250320212219.png)
기저 벡터들은 이미지에 따라(geometry-adaptive) 윤곽선(contours)과 균일한(regular) 영역 모두에서 진동하는 패턴을 보임. 윤곽선과 평탄한 배경 내부 세부 구조를 효과적으로 잘 표현한다.
GAHB! Geometry adaptive Harmonic basis

수축계수 $\lambda_k(y)$ 적응기저벡터$e_k(y)$ 신호계수 $\left\langle x, e_k\right\rangle$
이 basis에서는 신호 계수가 희소하므로 디노이징 성능이 향상되는 것이다.
성능 잘나오고 일반화도 잘되는거보면 DNN 이 사진 이미지 분포에 잘맞는 inductive bias라고 해석해야한다,.

#### $C^\alpha$ images and bandlet bases.
DNN은 그냥 기하적응조화벡터 내재적 편향이 있으면, 일반화와 최적 디노이징 성능을 기대할 수 있겠지, bases가 최적이라면. 그치?


$\alpha$ 에 의해 degree of regularity가 결정되는 상황에서 regular 배경들에 regular 윤곽선들로 이뤄진 이미지의 geometric $C^\alpha$ class 로 부를거야. synthetic 데이터 만든거야. 최적은 bandlet basis인데, 이미지의 윤곽선 방향에 맞춰 조정되는 harmonic function이다. 낮은 주파수로 진동하지만, 윤곽선 주변에선 높은 주파수를 나타낸다. 여기서도 점근적으로 $\alpha/(\alpha+1)$의 기울기를 가진다.

![Pasted image 20250320213522.png](/img/user/blog%20asset/Pasted%20image%2020250320213522.png)

#### Low-dimensional manifolds
![Pasted image 20250320215011.png](/img/user/blog%20asset/Pasted%20image%2020250320215011.png)
데이터는 이동 확장되는 원판이고, foreground와 background의 intencity가 가변적인 데이터임. 즉 5개의 파라미터로 이미지 생성. 따라서 최적이라면 5개의 교유값을 기대하고 나머지는 수축계수 0으로 날려버려야하고 MSE는 $5\sigma^2$ PSNR은 기울기 1급이어야함. 근데 접공간 잘 나타나지만, 배경과 윤곽선에도 진동하는 놈이 있고 결과도 서브옵티멀했음. DNN은 저차원 매니폴드에선 완벽히 최적화되기 어렵고, 노이즈 레벨이 낮을 수록 서브옵티멀리티가 증가함을 보여줌



#### shuffled 얼굴
![Pasted image 20250320215552.png](/img/user/blog%20asset/Pasted%20image%2020250320215552.png)
픽셀간 locality가 유지가 단되어서 최적기저가 harmonic하지 않음 -> 귀납적 편향이 데이터 분포간 불일치하여 성능저하됨.으로 해석




마무리- Diffusion model 성능은 반복적으로 적용해 고품질 이미지 생성하고 단순하고 야무지게 MSE 오차 최소화하며 학습됨. 데이터셋 크기가 증가하면서 특정 샘플에 의존안하는 density model로 수렴함. 데이터 적당히 써도 수렴잘함.
DNN은 GAHB에서 노이즈 계수를 축소함 $C^\alpha$ 하는거보면 최적급임. 
GAHB 클래스에 대한 정식 수학적 정의를 주진 않는다. 