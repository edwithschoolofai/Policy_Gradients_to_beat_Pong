""" Policy Gradients를 이용해 퐁(Pong) 에이전트를 학습하며, OpenAI Gym 을 사용합니다. """
#행렬 연산
import numpy as np 
#모델을 저장하고 불러오기 위함. 피클링은 파이썬 객체의 계층을 바이트 스트림으로 변환하는 과정입니다.
import cPickle as pickle 
#RL 알고리즘을 테스트할 환경을 제공하는 OpenAI 라이브러리
import gym

#1 단계 - 퐁은 어떻게 작동할까요?
#1 - 이미지 프레임을 받습니다.
#2 - 패들을 위/아래로 움직입니다.
#3 - 동작을 통해 보상을 받습니다 (AI 를 제치면 +1, 공을 놓치면 -1, 그 외는 0).
#다른 세부사항들은 넘어가고, 알고리즘에 집중해봅시다.

#2 단계 - RL 적용
#1 RL 은 머신러닝 중에서도 순서대로 액션을 취하는 것과 관련된 분야입니다.
#주로 하나의 에이전트가 미지의 환경과의 상호작용 속에서
#리워드를 극대화하는 과정으라 이해할 수 있습니다.
#2 RL 는 다른 기술과 접목되면 매우 강력해집니다 (알파고, DQN).
#Policy Gradients > DQN (DQN 의 저자를 포함한 대부분이 그렇게 생각합니다) https://www.youtube.com/watch?v=M8RfOCYIL8k
#3 두 개의 완전연결층(Fully Connected Layer)을 빌드합니다.
#https://karpathy.github.io/assets/rl/policy.png
#이미지 픽셀을 받아서, 위쪽 이동의 확률을 출력합니다 (확률론적, stochasticity).

#3 단계 - Policy Gradients는 지도 학습과 3 가지 주된 차이점이 있습니다.
#1 정답으로 레이블된 데이타가 없기 때문에, 일종의 가짜 레이블로써 Policy로부터 샘플링한 액션을 대신 사용합니다.
#2 잠재적으로 발생 가능한 결과에 기반하여 각 상황들마다 손실값을 튜닝하게 되는데 
#  이는 실제 잘 동작한 액션의 로그 확률은 증가시키고, 그렇지 않았을 때는 감소시키기 위해서입니다.
#3 연속적으로 변화하는 데이터셋(the episodes)에서 실행하며, 각각 보상을 측정되면
#  데이터셋별로 얻어진 샘플에 기반하여, 한 번씩만(또는 아주 조금만) 업데이트합니다.
#4 노드가 확률론적(stochastic)일 때에도, 오차역전파를 사용할 수 있습니다 (이후에 더 알아볼 것입니다).

#4 단계 - PG vs 인간
#PG(Policy Gradients)에는 반드시 실제 플러스의 리워드를 경험해야하며, 또한 더 빈번하게 경험해야만 하는데요
#그래야만 높은 리워드 반복적으로 얻기 위한 액션을 위해, Policy 파라메터들이 서서히 튜닝될 수 있기 때문입니다.
#인간도 모델이 있다고 생각해본다면, 우리는 실제 리워드를 받거나 잃을 지 직접 경험하지 않고서도 알 수 있습니다.
#단순히 향후 그러지 않기 위해, 수백 번 차로 벽을 박을 필요는 없는거죠.
#(똥인지 된장인지 먹어 보지 않고도 알 수 있습니다.)

#5단계 - Credit Assignment problem
#Credit Assignment 문제 : +1 리워드를 얻었을 때. 어떤 것 덕분일까요? 수 백만 개의 변수와 픽셀들...어떤 프레임일까요? 
#레이블조차 없습니다.
#https://karpathy.github.io/assets/rl/rl.png
#forward pass > 아웃풋의 확률분포 > 거기서 나온 샘플에서 뽑은 액션 > 리워드 대기 > 리워드를 기울기로써 오차역전파로 네트워크를 갱신
#보상은 어떤 양수든 가능합니다. 신경망 덕분에 크기만 신경쓰면 됩니다.

#훈련 시간 - 맥북으로 3 일, AWS의 GPU 클러스터로 약 두 시간


# 하이퍼파라미터
H = 200 # 은닉층 뉴런의 개수
batch_size = 10 # 파라메터 갱신 주기
learning_rate = 1e-4 #수렴하도록 하기 위함 (너무 느림 - 수렴하는데 시간이 걸린다, 너무 높음 - 수렴하지 않는다)
gamma = 0.99 # 리워드에 대한 할인률 (i.e 나중에 받을 리워드는 훨씬 덜 중요합니다)
decay_rate = 0.99 # RMSProp에 대한 감소율
resume = False # 이전 체크포인트에서 시작합니다

# 모델 초기화
D = 80 * 80 # 입력 크기: 80x80
if resume:
  model = pickle.load(open('save.p', 'rb')) #저장된 체크포인트에서 로드
else:
  model = {} #모델을 초기화합니다
  #표준 정규 분포로부터 샘플을 반환합니다
  #Xavier 알고리즘이 입력과 출력 뉴런 개수에 따라 초기화 비율을 결정합니다.
  #가중치가 0 에 매우 가깝다고 상상해본다면, 각 층을 지날 때마다 신호가 줄어들어 사용하기에 너무 작게 됩니다.
  #만약 가중치가 너무 크다면, 각 층을 지날 때마다
  #신호의 크기가 증가해 사용하기에 너무 커집니다.
  #Xavier 초기화를 이용함으로써, 가중치가 너무 작거나 크지 않게 만들어 신호를 정확하게 전달할 수 있게 됩니다.
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" 초기화
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  #zeros_like 는 주어진 배열과 동일한 형태와 타입을 가진 0 으로 이루어진 배열을 반환합니다.
  #배치마다 기울기를 더하여 버퍼를 갱신할 것입니다.
  #모델에는 k,v 쌍, 가중치, 레이어 등이 있습니다.
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } 
## rmsprop (경사 하강법) 모델을 갱신하기 위해 사용된 메모리
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } 

#활성 함수
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid 로 함수의 구간을 [0,1] 로 압축시킵니다

#하나의 게임 프레임을 입력으로 받습니다.
#모델에 적용하기 전 예비과정
def prepro(I):
  """ 210x160x3 uint8 프레임을 6400 (80x80) 1차원 float 벡터로 전처리 """
  I = I[35:195] # 일부를 잘라냅니다.
  I = I[::2,::2,0] # 2 배 만큼 다운샘플합니다.
  I[I == 144] = 0 # 배경을 삭제합니다 (배경 타입 1)
  I[I == 109] = 0 # 배경을 삭제합니다 (배경 타입 2)
  I[I != 0] = 1 # 다른 모든 것들을 1 로 설정합니다.
  return I.astype(np.float).ravel() #평편화 

#실제로는 할인률이 결정권을 가진 자가
#다음 결정으로 게임이 끝날지 확실치 않음을 보여줍니다.
#만약 결정권자가 로봇이라면, 할인률은 다음 턴에 로봇이 종료될 확률이 될 수 있습니다. 
#그래서 로봇이 멀리 내다보지 못하고,
#할인된 보상의 합이 아닌 보상의 합을 최적화하지 못하는 것입니다.

#우리는 장기간이 아닌 단기간의 보상을 최적화합니다 (젤다의 전설 같이 말이죠).
def discount_rewards(r):
  """ 1차원 float 배열의 보상을 받아 할인된 보상 계산 """
  #할인 보상 행렬을 빈 행렬로 초기화합니다
  discounted_r = np.zeros_like(r)
  #보상의 합을 저장합니다
  running_add = 0
  #각 보상마다 반복
  for t in reversed(xrange(0, r.size)):
    #t 에서의 보상이 0 이 아니면, 게임의 경계이기 때문에 합을 재설정합니다.
    if r[t] != 0: running_add = 0 
    #총합에 더해줍니다
    #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
    running_add = running_add * gamma + r[t]
    #더 많은 값이 주어질 때 이전의 보상
    #계산한 총합을 할인 보상 행렬에 지정해줍니다.
    discounted_r[t] = running_add
  return discounted_r


#numpy woot 를 이용한 순전파!
def policy_forward(x):
  #은닉 상태를 얻기 위한 첫 가중치를 입력으로 받는 행렬의 곱셈으로
  #여러 게임 시나리오를 감지할 수 있습니다.
  h = np.dot(model['W1'], x)
  #활성 함수를 적용해줍니다
  #f(x)=max(0,x) 최대값을 취합니다. 만약 0 보다 작다면 0 을 사용합니다.
  h[h<0] = 0 # ReLU 비선형성
  #과정을 한 번 더 반복합니다.
  #각 경우에 위 혹은 아래로 움직일지 결정합니다.
  logp = np.dot(model['W2'], h)
  #활성화를 통해 압축합니다 (이 경우에는 sigmoid 를 이용해 확률을 출력합니다).
  p = sigmoid(logp)
  return p, h # 2 번 동작의 확률과 은닉 상태를 반환합니다.

def policy_backward(eph, epdlogp):
  """ 역순 전달. (eph는 중간 과정의 은닉 상태 배열) """
  #두 층의 파생 오류를 재귀적으로 계산하는 것을 연쇄법칙이라 합니다.
  #epdlopgp 가 경사를 조절합니다.
  #가중치 2 의 갱신된 파생값을 계산합니다. 매개변소는 은닉 상태의 전치행렬 * 경사 입니다 (그리고 ravel() 로 평편화 해줍니다).
  dW2 = np.dot(eph.T, epdlogp).ravel()
  #숨겨진 파생값을 계산합니다. 경사와 2x2 가중치 행렬의 외적입니다.
  dh = np.outer(epdlogp, model['W2'])
  #활성화를 적용합니다.
  dh[eph <= 0] = 0 # 역전파
  #은닉 상태의 전치 행렬과 입력 관찰을 이용해 1 가중치로 파생값 계산
  dW1 = np.dot(dh.T, epx)
  #두 파생값을 반환해 가중치를 갱신합니다.
  return {'W1':dW1, 'W2':dW2}

#환경
env = gym.make("Pong-v0")
#각 시간 단위마다 에이전트는 행동을 취하고, 관찰과 보상을 환경을 통해 전달 받습니다.
#초기 관찰을 반환하는 reset 을 불러낼 때 과정을 시작할 수 있습니다.
observation = env.reset()
prev_x = None # 차이를 계산하는데 사용됩니다.
#관찰, 은닉 상태, 경사, 보상
xs,hs,dlogps,drs = [],[],[],[]
#현재 보상
running_reward = None
#총 보상
reward_sum = 0
#어디까지 왔을까요?
episode_number = 0

#훈련을 시작합니다!
while True:

  #관찰을 시작하고 입력을 신경망에 설정해 다른 이미지가 되게 합니다.
  #정책 신경망이 행동을 감지해야 합니다.
  #격차 이미지 = 최후 프레임에 현재값은 뺀 것.
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
  #이것이 이미지 격차이고, 이것을 적용해줍니다.

  # 정책 네트워크를 실행하고 반환된 확률을 이용해 행동을 취합니다.
  aprob, h = policy_forward(x)
  #확률론적인 부분입니다
  #모델로부터 멀리 떨어져있지 않게 때문에, 모델을 구분할 수 있습니다.
  #만약 모델로부터 떨어져있다면, 다시 매개변수로 나타냅니다.
  action = 2 if np.random.uniform() < aprob else 3 # 주사위를 굴립니다!

  # 여러 중간값을 기록합니다 (이후 역전파에 사용됩니다).
  xs.append(x) # 관찰
  hs.append(h) # 은닉 상태
  y = 1 if action == 2 else 0 #"가짜 라벨"
  dlogps.append(y - aprob) # 이루어져야 할 동작을 수행하는 것 (헷갈린다면 http://cs231n.github.io/neural-networks-2/#losses 를 참고)

  # 환경의 단계와 새로운 값을 가져옵니다.
  env.render()
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # 보상을 기록합니다 (이전 동작에 대해 보상을 얻기 위해 step() 을 호출한 후에 실행해야 합니다).

  if done: # 에피소드 종료
    episode_number += 1

    # 이 에피소드의 모든 입력, 은닉 상태, 동작 경사, 그리고 보상을 쌓아놓습니다.
    #각 에피소드는 몇 십 번의 게임입니다.
    epx = np.vstack(xs) #관찰
    eph = np.vstack(hs) #은닉
    epdlogp = np.vstack(dlogps) #경사
    epr = np.vstack(drs) #보상
    xs,hs,dlogps,drs = [],[],[],[] # 배열 메모리를 재설정합니다.

    #샘플 동작의 강점은 모든 보상의 가중치에 따른 합입니다. 하지만 이후에는 보상이 훨씬 덜 중요하게 됩니다.
    # 역방향으로 할인 보상을 계산합니다.
    discounted_epr = discount_rewards(epr)
    # 단위 정규화되도록 보상을 표준화합니다 (경사 추정 분산을 조절하는데 도움을 줍니다).
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    #우위(advantage) - 어떤 동작이 평균적인 동작에 비해 얼마나 좋은지 알려주는 지표.
    epdlogp *= discounted_epr # 우위(advantage)를 가지고 경사를 조절합니다 (PG 의 마법이 이곳에서 나타납니다).
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # 배치의 경사를 누적시킵니다.

    # 각 batch_size 마다 rmsprop 변수 갱신을 수행합니다.
    #http://68.media.tumblr.com/2d50e380d8e943afdfd66554d70a84a1/tumblr_inline_o4gfjnL2xK1toi3ym_500.png
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # 지루한 부기
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # 퐁은 게임이 끝날 때 +1 혹은 -1 의 보상이 있습니다.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
