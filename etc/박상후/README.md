# 🙌 아이디어 회의 | 📅 10.14 (월)
### 1. 블럭코딩 기반 AI 모델 생성 서비스

1. 주제 적합성: 중
    - 사용자는 블럭을 배치만 하면 이를 AI 모델 형태로 제공받아 쉽게 테스트, validation, 공유할 수 있다.

2. 개발 난이도: 하
    - 단순히 블럭을 배치하고 파라미터를 설정하고 이를 기반으로 모델을 만들어주는 플로우로는 볼륨이 작음.

3. 창의성: 상
    - AI 개발 입문자들에게 교육용 솔루션이 될 것이라 예상.

<br/>

# 📌주제 구체화 | 📅 10.17 (목)
## 주제
🧱 블록 코딩을 기반으로 AI 모델 생성 및  학습 지원 도구

### ❓ 왜 필요한가?
---

#### 구조 변경의 어려움과 시각적 관리 필요

AI 모델을 만들 때 필요한 레이어(예: ReLU, Linear, Conv, Pool 등)는 이미 정해져 있지만, 이 레이어들을 어떻게 배치하는지가 모델 성능에 큰 영향을 미칩니다. 기존에는 코드로만 이러한 구조를 조작하다 보니, 전체적인 모델 구조가 제대로 구현됐는지 확인하기 어려웠습니다. 블록 코딩을 사용하면 시각적인 접근이 가능해, 원하는 모델 구조를 직관적으로 확인할 수 있습니다.

#### 실시간 유효성 검증의 필요성

AI 모델 개발에서는 레이어의 순서뿐 아니라 각 파라미터(예: 레이어 크기, 활성화 함수의 종류 등)도 중요합니다. 파라미터 값이 잘못되면 실행 중에 오류가 발생하는데, 기존에는 이러한 오류를 런타임에서만 알 수 있었습니다. 블록 코딩 기반 도구는 블록을 추가할 때마다 유효성을 즉시 검증해주어, 코드 작성 중에 발생할 수 있는 오류를 사전에 방지할 수 있습니다.

#### 드래그앤드롭 방식의 편리함

코드를 일일이 작성하는 대신, 드래그앤드롭으로 모델을 구성할 수 있다면 개발자들이 더욱 빠르고 효율적으로 모델을 만들 수 있습니다. 이 도구는 AI 모델 생성 과정을 시각적이면서도 직관적으로 만들어 주어, 복잡한 구조도 간편하게 설계할 수 있게 해줍니다.

#### 학습과정 시각화

> [Tensorflow - Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=7,2&seed=0.43549&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
> 

AI 모델 학습은 그 과정이 복잡하고 추상적이어서, 초심자들에게는 쉽게 다가오기 어려울 수 있습니다. AI 모델이 데이터를 어떻게 처리하고, 학습을 통해 성능을 개선해 나가는지를 직관적으로 이해하기 위해서는 시각화가 필요하다고 생각했습니다.