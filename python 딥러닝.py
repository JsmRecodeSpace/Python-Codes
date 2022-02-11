


# 목차
* DNN
* CNN
* RNN
* Import codes
* 딥러닝의 발전을 이끈 알고리즘들
* 혼합아이디어




---------- DNN ----------



---------- CNN ----------



---------- RNN ----------



---------- Import Codes ----------

# Torch
import torch
import torch.nn as nn # 신경망 모델들이 포함되어 있다.
import torch.optim as optim # 경사하강법 알고리즘들이 들어있다
import torch.nn.init as init # 텐서를 초기화 하는 함수들이 들어가 있습니다. (ex. uniform, normal, xavier 등등)
import torch.nn.functional as F # 'torch.nn' Module 중에서도 자주 이용되는 함수를 'F'로 지정
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# torchvision
import torchvision.datasets as dsets # 컴퓨터 비전 연구 분야에서 자주 사용되는 'torchvision' 모듈 내의 'transforms'와 'datasets' 함수를 임포트
import torchvision.transforms as transforms


# For reproducibility
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)




---------- 딥러닝의 발전을 이끈 알고리즘들 ----------


    # underfitting & overfitting
 - 수용력에 변화를 준다는 것은 무슨 의미일까? 인공신경망을 예를 들면,
   은닉층의 개수 또는 은닉층의 노드의 개수 등을 뜻한다고 할 수 있음
 - 인공신경망은 은닉층의 개수를 늘릴수록 더 복잡한 형태가 표현 가능해지는데
   이처럼 표현 가능한 형태가 많아진다는 것은 모델의 수용력이 커짐을 의미.
   다항식으로 예를 들면 몇 차까지 사용할 것인지 정하는 것이 수용력을 정하는 것에 해당.
 - 수용력과 언더피팅, 오버피팅의 관계를 살펴보았을 때
   언더피팅으로 판명이 되면 모델의 수용력을 늘려야 할 필요가 있고,
   오버피팅으로 판명이 되면 모델의 수용력을 줄임으로써 최적 수용력에 가까워지게 해야 함.



    # Regularization (정형화, 규제, 일반화)
 - 정형화 (定型化): 일정한 형식이나 틀로 고정됨
 - 어떤 제약조건을 추가로 걸어줌으로써 오버피팅을 해결하려는 기법
    -> 이 제약조건은 주로 손실함수(loss function)에 추가된다.
 - 모델이 과적합되게 학습하지 않고 일반성을 가질 수 있도록 규제 하는 것.
    -> 오버피팅될 때 오버피팅 구간을 벗어날 수 있도록 해준다.
 - 하늘색 선(고차원 방정식의 선)은 오버피팅 될 수 있으므로 빨간색 점선(직선)으로 모델이 설정될 수 있게 해주는 작업이다.
 - 데이터의 feature에는 손대지 않고 최대한 하늘색 선을 펴주려면 기울기(가중치, w)를 건드리면 된다.
 - 이때 이용하는 정형화의 대표적인 방법이 Lasso(L1정형화), Ridge(L2정형화)이다.
    -> 이 정형화 식을 L1 또는 L2 페널티라고도 한다.

    # Lasso
 - Lasso (Least absolute shrinkage and selection operator)
    = MSE + α*L1_norm
   -> (손실함수에 가중치 파라미터(w)의 절댓값을 모두 합한 형태)
   -> 학습의 방향이 단순히 손실함수를 줄여나가는 것 뿐만 아니라, 가중치(w)값들 또한 최소가 될 수 있도록 진행된다.
   -> 여기서 중요한 것은 α(알파)값이다. 사용자가 조정할 수 있는 하이퍼파라미터이기 때문이다.
   -> α는 정형화의 정도를 의미함. lnα수치가 클수록 정형화를 강하게 적용
 - α(알파)에 따른 결과
α 가중치결과
↑	↓	  underfitting: α 값을 높이면 (α=1) 어차피 전체 페널티의 값도 줄여야하니 l가중치l의 합도 줄이도록 학습한다.
                        계속해서 특정상수의 값을 빼나가는 것이므로 몇몇 가중치들은 0으로 수렴하고 이에 따라 feature의 수도 감소하게 된다.
                        즉, 구불구불한 그래프를 feature의 수를 감소시키면서 펴주고 일반화에 적합하도록 만드는 것이다.
                        가중치가 0인 중요하지 않은 특성들을 제외해줌으로써 모델에서 중요한 특성이 무엇인지 알 수 있게 된다.
↓	↑	  overfitting:  반대로 α값을 줄이면(α=0.0001) 상대적으로 많은 feature를 이용하게 되므로 과적합될 우려가 있다.

    # Ridge
 - Ridge
    = MSE + α*L2_norm (라쏘와 비슷하게 기존 손실함수에 L2항을 추가)
    -> 미분 값의 음수를 취하면 각 항의 부호가 바뀌기 때문에 가중치가 양수일 땐 L2 패널티가 음수,
       반대로 가중치가 음수일 땐 L2 패널티가 양수로 되어 가중치를 0의 방향으로 잡아당기는 역할을 한다.
       즉, 가중치의 절댓값을 가능한 작게 만들려는 작업을 하는 것이다.
       이를 weight decay라고 하며 weight decay는 특정 가중치가 비이상적으로 커지고 그것이 학습 효과에 큰 영향을 주는 것을 방지할 수 있다.
 - λ(람다)에 따른 결과
λ	              가중치    	             결과
↑	                ↓                  underfitting
         (기울기 감소, 특성의 영향 감소)
↓	                ↑                  overfitting
         (기울기 증가, 특성의 영향 증가)
    -> 람다 역시 다른 하이퍼파라미터처럼 적절한 값을 찾아야 합니다.

    # Lasso vs Ridge
      Lasso (L1)	                       Ridge (L2)
가중치를 0으로, 특성 무력화	가중치를 0에 가깝게, 특성들의 영향력 감소
  일부 특성이 중요하다면	        특성의 중요도가 전체적으로 비슷하다면
     sparse model	                  Non-sparse model
   feature selection


    # 파이토치에서는 정형화를 사용하는 방법은 크게 두가지가 있는데
 (1) 기존의 손실함수(loss function)에 정형화 식을 명시적으로 추가하는 방법
 (2) 최적화함수(optimizer)에 가중치 부식(weight_decay)인수를 주는 방법이 있다.
    -> 경사하강법의 가중치 업데이트 과정에서 가중치 부식을 주는 것
  - 모델이 오버피팅할 경우, 적절한 강도로 정형화를 걸어주면 이를 어느정도 극복할 수 있습니다.
    # weight decay Ex
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
















---------- 혼합아이디어 ----------









───────────────────────────────────────────────────────────────
