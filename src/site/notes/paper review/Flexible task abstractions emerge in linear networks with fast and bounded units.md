---
{"dg-publish":true,"permalink":"/paper-review/flexible-task-abstractions-emerge-in-linear-networks-with-fast-and-bounded-units/"}
---


요약


- **전문화된 weight**는 gate가 더 빠르게 작동할 수 있게 하고,
- **유연한 gate**는 weight의 전문화 과정을 보호하고 가속하는 **선순환(상호 강화)** 구조가 나타난다.  
    이는 task가 바뀔 때마다 지식이 계속 잊히고 다시 학습해야 하는 **망각(forgetful) 모드**와 뚜렷하게 대조된다.


**유연한 regime**에 도달하기 위해서는

- 적절한 정규화(regularization),
- 차등적인 학습률(differential learning rates),
- 충분한 task block 길이가 필요하며,  이는 생물학적 뉴런과 인지과학에서 밝혀진 바람직한 학습 환경과도 일치한다.

# Introduction
#### 동물과 인공지능의 차이
**동물**의 경우, 상황에 따라 유연하고 적응적으로 행동을 하게 된다.
	갑자기 내가 바이올린을 켜는 법을 배운다고 했을 때(생애 최초)
	나는, 미적분학, 탑에서 라인전 하는 법, 에에올에 대한 감상, 한국형 세계 각지의 요리 레시피를 손대지 않고 활을 현에 비비고 있을 것이다.
	오히려 기계진동과 고체역학의 지식들을 통해 현의 길이에 따른 고유 진동수, 요리와 게임, 클라이밍 등을 통해 얻은 손놀림, 음악을 들으며 쌓인 교양을 통해 영어유치원 다음 스케쥴로 바이올린 학원에 있는 6세 영아보다 빠르게 바이올린 켜는 법을 습득할 것이다.
	
	이해에 도움이 안되었다면... 죄송 ㅎㅎ

-> 동물은 감각경험을 하면서, 그것을 세부 단위(task)로 학습하여 인식한다. 각 과제에 따르는 '내부적인 추상화'(representation, task abstraction)을 만들어 빠르게 취사선택해 사용하며, 저차원의 추상적인(입력에 달라지지 않는) task representation이 생긴다고 보는 것이다.

-> 동물은 추상화(abstract)와 조합(compositional)을 통해 환경에 대응한다.

AI의 경우에 dataset이 기본적으로 iid 가정. 같은 분포로 가정하며 학습되기 때문에, 현실처럼 시간적으로 순차적이거나 distribution shift에 취약하다. 
기존 모델은 모델 전체 파라미터를 직접 고치면서 적응을 시도하는데, 이러면 예전에 배춘 내용이 쉽게 잊게 되는 것이다.(Catastrophic Forgetting)
![Pasted image 20250708150045.png](/img/user/blog%20asset/Pasted%20image%2020250708150045.png)
### 분석
동물의 뇌에서 출발한 Task abstraction이라는 아이디어는
여러 태스크를 배운 후, 이건 무슨 과제인거 같은데...? 하고 어떤 task를 저차원의 task들로 분해해 인식하는 것.
이를 통해 과제별 추상화를 자연스럽게 만든다.
 장점: 새로운 과제가 오면, 파라미터 전체를 다시 고치지 않고, 필요한 부분(저차원 representation)만 빠르게 바꿔 적응.
 여러 과제 빠르게 전환하고, 기존 지식을 재조합해 새로운 문제도 잘 해결 가능

#### AI는 어떤 상황이지?
기존 finetuning은 데이터 요구, 느린 적응, 망각 등의 문제가 있음.)
현재 ANN은 데이터 흐름(환경이 순차적으로 변한다던가..)을 스스로 task로 분할해 내부적으로 만드는, 원리들이 잘 알려져 있지 않으며, 대부분은 task를 ID나 구분자를 통해 인위적으로 알려주거나 복잡한 방법을 써야했다.
(MoE 같이 routing수행하는 gating network를 이용하기, meta learning, representation 학습할때 통계 뜯어보기 등)

"어떻게 ANN이 뇌처럼 자연스럽게  task를 전환하며 기존 지식도 잘 보존할 수 있을까?" 
-> task abstraction을 **자연스레 유도**하고 활용

이 논문의 기여
- 신경망에서 **flexible** 모드와 **forgetful** 모드의 태스크 전환 방식을 설명하고, 유연한 모드를 유도하는 **effective dynamics를 분석**적으로 규명하였다.
- 그 모델이 data shift와 loger task blocks에서 기존처럼 데이터를 섞어서(inverleaved) 훈련하는 것보다 더 좋은 성능을 냈다. 또한 학습이 지속될수록, 태스크 전환이 빨라지는 현상도 관찰했다.
- FNN에도 일반화 해봤는데, 실험 결과 **두번째 레이어 weight에 대해 differential learning rates와 regularization을 적용**하는 것이, 앞 레이어에서 태스크 관련 모듈이 형성되고, 뒷 레이어가 태스크별 모듈을 선택하는 **gating 기반의 해법으로써 필요충분함**을 발견했다.
- non-linear networks로도 이 결과를 확장하여, PoC처럼, **비선형 CNN에서도 이런 구조**를 통해 두 자리 숫자 분류 과제 학습 실험을 진행했다.

# Approach
## 문제 setting
- $M$개의 서로 다른 task가 순차적으로 주어지는 dynamic learning problem
- 각 시간 $t$마다, 현재 태스크 $m$에 해당되는 입출력 쌍($x(t), y^*_m(x(t))$)을 받음
- 한 task는 일정 시간 주기 $\tau_B$동안 유지되다가 다음 태스크로 전환됨.
- 네트워크는 task Id나 task의 경계를 전혀 모름!
![Pasted image 20250708184200.png](/img/user/blog%20asset/Pasted%20image%2020250708184200.png)

구체적으로 멀티태스크 teacher-student 설정을 고려
각 task가 teacher임. $W^*_m$
그리고 정답 레이블은 $y^*_m=W^*_mx$ 
x는 매 시점 가우시안 iid에서 샘플됨.
어떤 입력에 대해 orthogonal한 응답을 내도록 teacher를 무작위로 생성하였음.$w^{*1}_i \cdot w^{*2}_i=0$(appendix 9에선 non orthogonal에도 일반화함.) 

## 모델 구조
Linear gated neural network 기반으로 이용
- $P$개의 student
- student weight matrices.   $W^P \in \mathbb R^{d_{out}\times d_{in}}$ , scalar variables $c^p \in \mathbb R$
- model output $y \in \mathbb R^{d_{out}}$
$$
y=\Sigma^P_{p=1} c^p W^p x.
$$
$c^p$를 통해 activation 되면서 *task abstraction*의 패턴을 구현하는 것이다. 이런 방식으로 path가 형성된다.
그런데 이 Neural Task Abstraction 구조를 학습시키려면,(자기가 이름 붙임)
- $W^p$와 $c^p$를 regularized loss function $\mathcal L = \mathcal L_{task} + \mathcal L_{reg}$ 로 gradient descent를 통해 업데이트 해야한다.
- 우리는 gate에 weight 보다 더 짧은 시간 주기를 부과한다. $\tau_c< \tau_w$ 이다. (근데 이건 task가 충분히 고차원이라면 비필수적이라고 함.)

![Pasted image 20250708184217.png](/img/user/blog%20asset/Pasted%20image%2020250708184217.png)

## Loss 디자인
task loss
- $\mathcal L_{task} = \frac{1}{2}\Sigma^{d_{out}}_i \langle (y^{*m}_i - y_i)^2 \rangle$   
그리고 게이트 값이 양수/크기 제한/ 경쟁 구조를 가지기 위해 다음과 같이 설계함
$$
\mathcal L_{reg} = \lambda_{norm}\mathcal L_{norm} + \lambda_{nonneg}\mathcal L_{nonneg} \qquad \mathcal L_{norm} = \frac{1}{2}(\|c\|_k-1)^2 \quad \mathcal L_{nonneg} = \Sigma^P_{p=1} \max(0,-c^p)
$$
$k \in 1,2$ 
해설 -> 
norm loss는 미분 깔끔해져서 0.5 붙임. 게이트 백터 c의 k-norm이 1이 되게끔 만듦.
nonneg loss는 gate output에서 음수가 나오면 커짐. - > 양수가 되도록 만듦

이러면, 특정 경로에 치우치지 않고 임의의 convex combination으로 해를 찾도록 유도함.
Appendix B.3. 참고.
	solution 공간의 덜 관측된 공간에서의 효과를 경감시키고, 서로 다른 component를 specialization하게만듦
	$y = c^1 w^1 + c^2 w^2$
	 Nonnegativity -> 생물학적 뉴런의 발화율은 0보다 큼
	 $w^1, w^2 \text{ are basis of } \mathbb R^2$ 
	 $\mathcal L_{nonneg} = \Sigma^P_{p=1}\max(0,-c^p)$
	 Competition으로 invariance 해소.
	$c^p, W^p \rightarrow ac^p, W^p/a$면 model은 invariant임.
	c를 통해 gating으로 분업이 드러나야함.
	그렇기 때문에 vector $c = (c^1, c^2)^T$의 norm으로 bounding하여서 한정된 pi를 가져오기 위해 경쟁시키는거. -> symmetry breaking 유도.
	 동시에 여러 게이트가 동시에 활성화될 수 있는 조합적 성질(compositionality)도 허용한다.
	 ![Pasted image 20250709150602.png](/img/user/blog%20asset/Pasted%20image%2020250709150602.png)
small learning rates (gradient flow)
$$
\tau_c \frac{d}{dt} c^p = - \nabla_{c^p} \mathcal L, \qquad\tau_w \frac{d}{dt} W^p = - \nabla_{W^p} \mathcal L.
$$
$\tau$는 모델의 시간 상수고, 
$W^p_{init} ~ \sigma^2/d_{in}, \quad \sigma \sim N(0,0.01),\quad  c^p=1/2$  

# joint gradient descent 를 통한 Task abstraction 

"동시에 gate와 weight를 gradient descent로 학습시켜도 내부적으로 task abstraction(task별 modul)이 저절로 유도될 수 있을까?"
-> 그렇다.

![Pasted image 20250709172628.png](/img/user/blog%20asset/Pasted%20image%2020250709172628.png)
task 두 개(M=2), path 수도 두 개(P=2)로 설정.
gate regulization과 gate/weight timescale 차이를 적용하지 않은 모델을 사용.
A와 F를 보면, 제안한 방법론인 flexible NTA만 block 변화에 점점 빠르게 적응함을 볼 수 있음.(A, 제안 방법론(검정)은 block이 바뀔 때마다 점점 더 뾰족해지지만, 아닌 것은 기울기가 비슷함., f는 loss가 .1에 도달하는 시간이므로 같은 인사이트를 제공.)

C와 D 그리고 B를 함께 보면, C, D를 통해 블록을 거칠수록 student가 teacher에 잘 맞춰지고(실선은 1에 가깝고 점선은 0에 가깝고), c는 의미없이 요동치나, 이후 c가 수렴하는 모습을 볼 수 있음.그리고 c가 어느정도 수렴하면서, student의 수렴도 더 빨라지는 것을 볼 수 있음..(한 10 이후 즈음)  
W가 업데이트 된 후 task별 specialization이 일어나고, c가 학습되며, flexible하게 network가 구성됨.

![Pasted image 20250708194934.png](/img/user/blog%20asset/Pasted%20image%2020250708194934.png)
![Pasted image 20250709172350.png](/img/user/blog%20asset/Pasted%20image%2020250709172350.png)
"gating 을 통해(c^p, task abstraction) compositional generalization을 지원할 수 있을까?"
아마 그런듯..?

Task composition experiment setting
- 세개의 path와 세 개의 teacher(A, B, C)로 번갈아(blocks) 학습
- 그 다음에는 새로운 조건으로, teacher들을 더한 조합(A+B, B+C, C+A)를 학습시킴.(Fig.3A)
- 결과적으로 flexible NTA 모델은 composite task를 훨씬 더 빠르게 학습하였음.

Subtask composition
- 각 teacher(A, B, C)의 행(row) 단위로 쪼개
- 서로 다른 teacher의 행을 섞어 새로운 task를 구성(Fig. 3B)
- 이때, 모델의 gate도 expressive하게 각 행마다 제어할 수 있도록 설계하였고, flexible regime에서 이런 compositional task도 잘 적응하였음.

하지만, 정규화가 제거된 forgetful model은 이런 조합적 task에 잘 적응하지 못했음.
Figure A.12를 보자.
이 파트의 핵심 질문은 compositional generalization을 달성할 수 있는지를 평가하는 것이다. 
task composition(Fig. A.12A), subtask composition(부분 작업 조합, Fig. A.12.B) 환경이다.
task composition은 전체 student matrix에 대해 gating을 했고, 
subtask compostion은 student matrix의 각 row 단위로 gating을 진행했다.
이렇게 각 뉴런별 gate를 두는 버전을 per-neuron gating NTA라고 하자.

per-neuron gating NTA는 $Pd_{out}$만큼의 경로가 gate에 의해 조절된다.
각 teacher에 대해 specialization과 gating이 잘 이뤄지는지 확인하기 위해,
$Pd_{out}$개의 path를 $d_{out}$크기의 $P$개 path로 분류한다.
-> W의 첫 계층 각 행과 각 teacher 행 사이 cosine similarity, 잘 맞는 것끼리 sorting.
이렇게 각 student의 특화여부와 student-teacher alignment를 시각화하였음.(Fig.A.12F)

이렇게 per-student gating NTA와 per-neuron gating NTA 모두 각 task 및 subtask 일반화 과제를 잘 해결하였고, compositional setting에서도 specialization을 유지함을 확인하였음. gate 또한 조합 과제에서, 균등하게 가중치가 켜지는 모습을 보여주었음.


## Mechanism 분석 
joint gradient descent를 통해 flexible regime으로의 수렴을 관측함.
-> 모델의 선형성을 활용해 teacher의 SVD(singular value decomposition) 공간에서 learning dynamics를 찾아봤음.

$M=P=2$ case,
각 mode $\alpha$에 대해서 student의 weight를 teacher 의 SVD space로 projection하면
$$w^p_{m\alpha} \coloneqq u^{*mT}_\alpha W^p v^{*m}_\alpha$$
$y = c^1w^1 + c^2w^2 \in \mathbb R^2$ 
![Pasted image 20250709220104.png](/img/user/blog%20asset/Pasted%20image%2020250709220104.png)

$$
\tau_w \frac{d}{dt}w^p = c^p(y^{*m}-y) \quad(1)\qquad  \tau_c \frac{d}{dt} c^p = w^{pT}(y^{*m}-y)-\lambda \nabla_{c^p} \mathcal L_{reg} \quad(2)
$$


"자기 강화 피드백 루프로 인해 Specialization이 일어난다."


-> Specialized 네트워크(student)와 regularization이 fast and separated gates를 야기한다.
	$\bar w_1 \coloneqq w^{p=1}_{m=1} - w^{p=2}_{m=1}$ 
	$\bar w_2 \coloneqq w^{p=2}_{m=2} - w^{p=1}_{m=2}$ 
	$$\bar w \coloneqq \frac{1}{2}(\bar w_1 + \bar w_2) \quad(3) \qquad \bar c \coloneqq c^1 - c^2 \quad(4)$$
	이러면 $\bar w$와 $\bar c$가 클수록 specialization 됐다고 볼 수 있다.
	$\epsilon \coloneqq y^{*m}-y$ 이라고 하고, teacher basis 계수로서 표현하면 $\epsilon_m \coloneqq \epsilon^T w^{*m}$
	현재 M=2인 세팅이므로 다음과 같이 표현된다.
	$$\tau_c \frac{d}{dt}c^p = \epsilon_1 w^{pT}w^{*1}+\epsilon_2 w^{pT}w^{*2} - \nabla_{c^p} \mathcal L_{reg}$$
	앞에 $\epsilon$계수는 상수고 $w^{pT}w^{*m}$은 cosine similarity이므로
	얼마나 유사한지(각도)와 $w^p$의 크기로 gate switching speed가 결정됨.
	이때, $w^p$는 student의 singular value(특이값)으로부터 만들어지므로
	초기 $W^p$의 원소 값에 비례해 크기가 정해짐. ~~(Marcenko-Pastur distribution)을 따름~~
	
	학습초기 작은 값으로 초기화 되므로, 게이트 변화도 작아짐.(fig. 4.D 참고)
-> Flexible gates는 학습된 특성을 보호한다.
	$$
	\tau_w\frac{d}{dt}w^p \simeq c^p \epsilon + \frac{1}{2}\bigg((\frac{d}{dt}c^p)\epsilon + c^p (\frac{d}{dt}\epsilon)\bigg)
	$$
	student gates는 결국 두 컴포넌트간의 오차 차이 ($\epsilon_1 - \epsilon_2$)와 게이트 활성화 차이($\bar c = c^1 - c^2$)에 의해 촉진된다.
	$$
	\tau_w \frac{d\bar w}{dt} = \frac{1}{2}\bar c(\epsilon_1 - \epsilon_2)
	$$
-> dynamics를 학습하는 정확한 솔루션이 flexible regime에서 symmetry하의 보호와 적응을 의미한다.
	flexible regim에서 해석적 해를 구하면
	$$\frac{\tau_c}{\tau_w}\frac{d\bar c}{d\bar w} = 2 \frac{\bar w(\epsilon_1 - \epsilon 2)}{\bar c (\epsilon_1 - \epsilon_2)}$$
	$$
	\bar w=\sqrt{1-\frac{1}{2}\frac{\tau_c}{\tau_w}(1-\bar c^2)}
	$$
	$\bar w = \bar w_1 = \bar w_2$인 대칭 조건과 오차의 대칭 조건($\epsilon_1 = -\epsilon_2$) 그리고 강한 L1 정규화에서, 게이트와 가중치 변화 비율에 대한 미분방정식을 정확하게 풀 수 있었다.
	게이트 타임스케일(\tau_c)기 찗을 수록 학생의 기존 지식을 잘 보호할 수 있음을 알 수 있다.




![Pasted image 20250715190059.png](/img/user/Pasted%20image%2020250715190059.png)
블록 길이, 정규화, 그리고 게이트 학습률이 모델의 specialization에 어떻게 영향을 주는지 분석하면 다음과 같다. 전체 데이터 양을 고정했을 때, 블록 길이가 길어질 수록 모델의 전문화가 더 잘 나타났으며, 이는 게이팅이 시간-역전 대칭성을 깨기 때문에 발생함.
만약 각 태스크를 일정 시간 학습하고 다음에 다른 태스크를 같은 시간 학습한다면, 원래 첫번째 태스크에서 배운것이 두번째에서 forgetting 됨. -> 변화 없음 -> 시간 역전 대칭성
하지만 게이팅이 있으면, 두번째 블록에서도 첫번째 학습된것이 남아있으므로... 블록길이가 길수록 specialization이 잘 나타나는 것.




![Pasted image 20250715191243.png](/img/user/Pasted%20image%2020250715191243.png)
![Pasted image 20250715191317.png](/img/user/Pasted%20image%2020250715191317.png)
