---
{"dg-publish":true,"permalink":"/paper-review/learning-without-training-the-implicit-dynamics-of-in-context-learning/"}
---


새로 제안한 개념들.
contextual layer. 
context에 관한 처리를 하는 레이어. transformer의 self-attention layer 역할 등을 일반화하여 생각

 $A(\cdot)$ : 기본적으로 single vector $x$를 input으로 받는 함수, 선택적으로 추가적인 context $C$를 받을 수도 있다.(sequence of tokens, image etc). so, output - $A([C,x])$ 간단하게 $A(C,x)$

이때, $A(C,x)$는 self-attention layer(transformer의 경우)의 output을 통해서, $A(C)$와 같은 vector space를 점유하므로, 다음과 같이 contextual vector를 정의할 수 있다.
$$
\Delta A(C) \coloneqq A(C,x) - A(x)
$$


contextual block.
$T_W = M_W \circ A$   
contextual block $T_W$는 contextual layer $A$와 neural network $M_W$의 합성함수이다.
추가로, $M_W(z) = f_\theta(Wz + b)$이다. 논의에서 layer 개수 안 중요하다는 것. 모든 dense-layer 포함하는 개념


Theorem 2.2
contexual block $T_W$는 weight matrix $W$를 포함하는 fully connected layer $M_W$와 contexual layer $A$로 구성되어있다. 주어진 context $C$와 input $x$에 대해,  $Y \subset C$인 $Y$가 $T_W$에 미치는 효과는 $M_W$의 첫번째 레이어에 대한 rank 1 업데이트 $W+\Delta W(Y)$에 암묵적으로 대응된다.
수식적으로는 다음과 같다.
$$
T_W(C,x) = T_{W + \Delta W(Y)}(C\backslash Y,x) \quad \text{where} \quad \Delta W(Y) = \frac{(W\Delta A(Y))A(C\backslash Y, x)^T}{\|A(C\backslash Y,x)\|^2}
$$
이때, $\Delta A(Y) = A(C,x) - A(C\backslash Y,x)$는 $Y$에 대응하는 context vector이다. 또한, $\Delta W(Y)$의 rank는 1이다.
$C\backslash Y$는 Y에 속하지 않는 모든 C의 원소 집합이다.

$$
T_W(D U Y, X) = T_{W+\Delta W(Y)}(D,x)
$$
논문에 증명이 함께 써있는데, 간단하니 재료만 써 두면,
2.2의 $\Delta W(Y)$와 $\Delta A(Y)$의 정의를 통해 구할 수 있다.
$$
\begin{align}

\end{align}
$$

다시 말하자면,
어떤 contextual layer라도, prompt로 부터 첫번째 신경망 층의 implicit weight transfer를 만들어 냄이라고 해석할 수 있다. contextual layer는 self-attention, RNN, local attention을 활용하는 recurrent layer 등이 있다. Y=C면 다음과 같이 쓸 수 있다.

$$
T_W(C,x) = T_{W+\Delta W(C)}(x), \quad \text{with} \quad \Delta W(C) = \frac{(W\Delta A)A(x)^T}{\|A(x)\|^2},
$$
where, context vector $\Delta A = A(C,x) - A(x)$ , $\Delta W$는 rank 1이다.

Appendix A에는 skip-connection 일반화한 거 있는데 이거는 정리를 추가로 해보자.
	$T(C,x) = x + A(C,x) + W'g_{\theta}(WA(C,x)+b)+b'$
	$g_\theta$는 any differential model.
	$W(Y) = W+\Delta W(Y)$
	$b'(Y) = b' + \Delta b'(Y)$ 
	
	$T_{W,b'}(C,x) = T_{W(Y),b'(Y)}(C\backslash Y,x)$
	$\Delta b'(Y) = \Delta A(Y)$
	$\Delta W(Y) = \frac{(W\Delta A(Y))A(C\backslash Y, x)^T}{\|A(C\backslash Y, x)\|^2}$
	이 식들을 가지고 아까보다는 귀찮지만, 풀면 풀린다.
	$T_{W(Y), b'(Y)}(C\backslash Y,x) = x + A(C\backslash Y, x) + \Delta A(Y) + W'g_\theta \big(W(A(C\backslash Y, x) + \Delta A(Y)) + b \big) + b' = T_{W,b'}(C,x)$
	이러면, 최근 세 논문에서 제안된 몇가지 개념과  $\Delta b'(Y)$이 연결된다고 하는데... 인용된 논문의 개념들은 이 논문의 Appendix A 말미의 인용을 참고하시길 바란다.
	 또한 pre-LN 트랜스포머 블록, 로컬 어텐션을 이용하는 Griffin recurrent model 등에도 적용이 가능해지는 것이다.
	

## The implicit learning dynamics of ICL

sequence of tokens $C = [c_1, ..., c_n]$

위의 표기를 반복 적용하면, implicit dynamics가 드러나게 된다.
초기 가중치  $W_0$, NN의 첫 dense layer $M_W$
$$
\begin{align}
T_{W_0}(c_1,x) &= T_{W_0+\Delta W_0(c_1)}(x)\\
T_{W_0}(c_1,c_2,x) &= T_{W_0+\Delta W_0(c_1,c_2)}(x)\\
&\vdots\\
T_{W_0}(c_1,\ldots,c_n,x) &= T_{W_0+\Delta W_0(c_1,\ldots,c_n)}(x)
\end{align}
$$
이때, $C$에 대해 수렴하는 weight를 다음과 같이 표기하면,
$$
\begin{align}
W_1 &= W_0 + \Delta W_0(c_1)\\
W_2 &= W_0 + \Delta W_0(c_1,c_2)\\
&\vdots\\
W_n &= W_0 + \Delta W_0(c_1,\ldots,c_n)
\end{align}
$$
아래와 같아진다.

$$
T_{W_n}(x) = T_{W_0}(c_1,\ldots,c_n)
$$

이걸 online gradient descent 학습 역학과 비교하여 Proposition 3.1을 얻는다.
토큰을 데이터 포인트로 고려하는 것이다.
proposition 3.1

$$
W_i = W_{i-1} - h \nabla_W L_i(W_{i-1})
$$
learning rate $h = 1/\|A(x)\|^2$ 이고, step $i$에서의 loss는  $L_i(W) = \text{trace}(\Delta_i^T W)$이다.
where, $\Delta_i = W_0 \big(A(c_1,\ldots,c_i,x) - A(c_1,\ldots,c_{i+1},x)\big) A(x)^T$


$W_{i+1}-W_i$를 정리한 식을 정리하면, 쉽게 아래 식을 유도가능하다.
$$
W_{i+1} = W_i - h\Delta_i = W_i - h\Delta_W \text{trace}(\Delta^T_i W)
$$
이때, $\nabla_W \text{trace}(A^T W) = A$이다.

직관적으로 $\Delta_i$는 $i+1$번째의 컨텍스트 토큰이 추가되는 효과를 측정하는 것이다.



Appendix B에서는 각 단계마다 변화하지 않는 다른 형태인 dynamics를 기술하는데, 이건 분해 공식을 통해 얻는다.
	$T_{W_0}(c_1,\ldots,c_n,x) = T_{W_0+\Delta W_0(c_1)}(c2,\ldots,c_n,x)$
	$W_1 = W_0 + \frac{(W_0\Delta A(c_1))A(c_2,\ldots,c_n,x)^T}{\|A(c_2,\ldots,c_n,x)\|^2}$
	iteration 가능하니...
	$W_i = W_{i-1}+h_i W_{i-2}A_i = W_{i-1}(1+h_i A_i)$
	$A_i \coloneqq \Delta A(c_i)A(c_{i+1},\ldots,c_n,x)^T$
	$h_i \coloneqq \frac{1}{\|A(c_{i+1},\ldots,c_n,x)\|^2}$
	  따라서 context C가 x에 미치는 영향은
	  $W_n = W_0(1+h_1 A_1)(1 + h_2 A_2) \ldots (1+ h_n A_n)$




## 실험
기존 ICL 관련 이론을 살펴보지 않았지만, 본 논문에 제안된 이론을 통해 더 적은 제약 조건으로 contextual layer와 NN 조합으로 만들어지는 Context block의 dynamics를 기술할 수 있다는 것을 알았다.

그 방식은 context 토큰을 데이터 포인트로 보고, 첫 번째 context block의 학습으로서 표현하는 것이다.

정리 2.2에 관하여, 프롬프트가 주어졌을 때 예측 $T_W(C,x)$이, 프롬프트 없이도 rank 1이었던 $\Delta W$를 MLP에 적용한 예측 $T_{W+\Delta W}(x)$와 사실상 동일함을 보이는 것이 목적이다.
이때, $\Delta W$는 context token을 순차로 학습하면서 정의되는데, 이것이 소멸하는지.(수렴하는지)를 측정하여, 이러한 Dynamics가 동작하는 지 확인하는 것이다.

프롬프트는  $(x_1,h(x_1)), \ldots, (x_N,h(x_N)),x_{query}$이고, $h(x) = \langle w, x \rangle$ 이때, $w,x_i,x_{query} \sim \mathcal N(0, I_d)$
모델은 하나의 트랜스포머 블록에, MLP skip connection 제거.
예측은 query토큰 출력의 마지막 성분.
결과적으로 i가 커질수록,  $\|(\Delta W)_{i+1} - (\Delta W)_i \|_2$가 감소하고 소멸하였다. 이러한 수렴성은 경사하강법의 Dynamics와 유사하게 나타났다. (batch loss가 $(\hat y_{query}) - \langle w, x_{query}\rangle)^2$) 
![Pasted image 20250729181039.png](/img/user/Pasted%20image%2020250729181039.png)
pretrain 후 새롭게 만든 세트에서 앞 $i$개의 예시만으로 SGD 파인튜닝을 수행하여, test loss를 계산한 결과와, $\Delta W$를 계산하여 얻은 test loss와 비교해도, 100회 평균 서로 유사하게 loss를 줄여나갔음.
![Pasted image 20250729181559.png](/img/user/Pasted%20image%2020250729181559.png)


## 한계:
단일 블록, 마지막 입력토큰의 출력과 첫 생성 토큰에 한정된 등가성.
제한적인 실험 세팅.

