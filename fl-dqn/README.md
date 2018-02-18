# Flappy Bird를 플레이하는 방법을 학습하기 위한 Deep Q-Network를 사용하기

<img src="./images/flappy_bird_demp.gif" width="250">

7분 버전: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)

## 개요
이 프로젝트는 딥 강화학습과 함께 아타리를 플레이하는데 묘사된 딥 Q러닝의 방법을 따르고 이러한 학습 알고리즘이 더 나아가 악명높은 Flappy Bird를 일반화할수 있다는 것을 보여준다.

## 설치에 필요한 것:
* Python 2.7 또는 3
* TensorFlow 0.7
* pygame
* OpenCV-Python

## 어떻게 실행하나요?
```
git clone https://github.com/yenchenlin1994/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python deep_q_network.py
```

## 무엇이 Deep Q-Network인가요?
그것은 합성곱 뉴럴 네트워크입니다. 그리고 그것은 input값이 raw 픽셀이고 output값이 미래의 보상을 예측하는 가치 함수값인 Q러닝의 변종으로 훈련되었다.

딥 강화학습에 관심있는 사람들에게 아래의 포스트를 읽기를 강하게 추천드린다.

[Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Deep Q-Network 알고리즘

주어진 딥 Q러닝 알고리즘을 위한 모의 코드는 아래에서 볼 수 있다.

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```

## 실험들

#### 환경
딥 Q 러닝이 매 시간 단계에서 게임 화면으로부터 관찰된 raw 픽셀 값들에서 훈련된 이후로 기본 게임에서 나타나는 배경을 제거하는 것이 그것이 더 빠르게 수렴되게 만들 수 있다는 것을 찾아내었다. 이러한 과정은 아래의 그림으로서 시각화된다.

<img src="./images/preprocess.png" width="450">

#### 네트워크 구조
[1]번 논문에 따라서, 나는 먼저 아래의 단계들로 게임 화면들을 미리 처리했다.:

1. 이미지를 회색조로 변환한다.
2. 이미지를 80x80으로 리사이징한다.
3. 네트워크를 위한 80x80x4 input 배열을 만들어 내기 위하여 전의 4 프레임을 쌓아(Stack)둔다.

그 네트워크의 구조는 아래 그림에서 보여진다. 첫 번째 layer은 4 stride 크기에서 8x8x4x32의 kernel(\*핵심)으로 input이미지를 합성 한다. 그러면 output은 2x2 max pooling 층(max pooling은 layer의 최댓값을 갖는 요소를 모아서 resize 하는 것을 의미한다) 을 만들어낸다. 두 번째 층은 2 stride에서 4x4x32x64 kernel 로 합성한다. 그리고 다시 max pool 한다. 세 번째 층은 1 stride에서 3x3x64x64로 합성한다. 다시 한번 max pool 한다. 마지막 숨겨진 층이 완전히 256 개의 연결된 ReLU(*sigmoid 보다 개선된 함수) nodes로 구성된다.
<img src="./images/network.png">

마지막 output 층은 그 게임에서 수행되어질수 있는 유효한 액션들의 수로써 같은 차원수를 가진다. 그리고 그곳은 0번째 인덱스가 아무것도 하지 않는것과 일치하는 곳이다. 이 마지막 층에서 값들은 각각의 유효한 액션들을 위한 input 상태가 주어진 Q 함수를 나타낸다. 매 단계에서 그 네트워크는 어느 액션이 ϵ greedy(*탐욕) 정책을 사용해서 최고의 Q 값과 일치하는지를 구한다.

#### 훈련
처음에 나는 0.01의 표준 편차로 정규 분포를 사용해서 무작위로 모든 가중치 행렬을 초기화했다. 그리고 50000번의 실험들의 최대 크기로 재시작 매모리를 설정했다.

나는 네트워크 가중치의 갱신 없이 처음 10000번의 단계들로 무작위로 균등하게 액션들을 고름으로써 훈련을 시작했다. 이것은 그 시스템이 훈련이 시작되기 전에 그 재시작 매모리를 덧붙이는 것을 허락한다.

ϵ = 1로 초기화한 [1]번 논문과 달리 나는 선형적으로 다음 3백만 프레임에 거쳐서 0.1부터 0.0001까지 ϵ을 강화했다. 이렇게 내가 설정한 이유는 에이전트가 우리의 게임에서 매 0.03초 (30프레임)마다 액션을 선택할수 있게 하면, 높은 ϵ이 그것이 너무 많이 뛰게 만들고 그래서 게임 화면의 최상단에 그것이 유지되도록 만들고 결론적으로 어설픈 방법으로 그 파이프에 부딫치게 만들기 때문이다. 이 상태는 ϵ이 낮을 때 그것이 단지 다른 상태들을 본 이후로 Q 함수가 상대적으로 느리게 수렴하게 만든다. 하지만 다른 게임들에서는 ϵ을 1로 초기화하는 것이 더 합리적이다.

훈련 기간동안 각각의 단계에서 그 네트워크 예시들은 훈련할 재시작 매모리로부터 32크기의 미니 batch(*집단)들을 시험해본다. 그리고 0.000001의 학습률로 Adam 최적화 알고리즘으 사용해서 위에 묘사된 loss 함수(*비용함수) 에 점진적 하강 단계를 수행한다. 강화가 끝난 뒤로 그 네트워크는 ϵ이 0.001로 고정된채로 무기한으로 훈련을 계속한다.

## 질의응답

#### 체크포인트를 찾을 수 없다.
[`saved_networks/checkpoint`의 첫번째 줄을 ](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/saved_networks/checkpoint#L1)

`model_checkpoint_path: "saved_networks/bird-dqn-2920000"`로 바꾸어라.

#### 어떻게 다시 만들수 있는가?
1. [이 줄을](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/deep_q_network.py#L108-L112) 주석처리해라.

2. 아래애 나온 것 처럼 `deep_q_network.py`의 파라미터들을 수정해라:
```python
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
```

## 참고 문헌

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[3] Kevin Chen. **Deep Reinforcement Learning for Flappy Bird** [Report](http://cs229.stanford.edu/proj2015/362_report.pdf) | [Youtube result](https://youtu.be/9WKBzTUsPKc)

## 부인 설명
아래의 래포들을 바탕으로 하고 있다.:

1. [sourabhv/FlapPyBird] (https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)

