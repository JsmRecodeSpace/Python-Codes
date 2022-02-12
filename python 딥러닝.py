


# 목차
* DNN
* CNN
* RNN
* Import codes
* 딥러닝의 발전을 이끈 알고리즘들
* 공부하며 떠오른 것들
* 혼합아이디어




---------- DNN ----------



---------- CNN ----------



    # Data Augmentation 데이터 증강
 - 복잡한 모델을 만들기 위해서 다량의 데이터가 필요합니다. 하지만 우리가 갖고 있는 데이터는 한정적입니다.
   그래서 이를 보완하기 위한 방법이 Data Augmentation입니다. 데이터를 임의로 변형해 데이터의 수를 늘려
   다양한 Feature를 뽑는 방법을 'Data Augmentation'이라 합니다.
 - 이미지를 돌리거나 뒤집는 것만 해도 컴퓨터가 보기엔 전혀 다른 수치이기 때문에
   데이터 수를 늘리는 효과를 가져온다고 할 수 있습니다.
    - Flip: 반전을 의미합니다. 이미지를 랜덤하게 좌우 또는 상하 반전시키는 Random Flip 입니다.
    - Rotation: 이미지를 회전시키는 것입니다.
    - Crop: 이미지의 일정 부분을 잘라 사용하는 기법입니다.
    - Scaling: 이미지를 확대 또는 축소시키는 기법입니다.
    - Cutout: 이미지의 일부를 사각형 모양으로 검은색을 칠하는 기법입니다. 숫자로는 0을 채워 넣는 것이라 생각할 수 있습니다.
                일종의 Input 데이터에 대해 Dropout을 적용한 기법이라 이해하면 됩니다.
    - Cutmix: 두 이미지를 합쳐놓고 이미지의 Label를 학습시킬 대 각각의 이미지가 차지하는 비율만큼 학습시키는 방법입니다.
         ->Cutout과 Cutmix 모두 일반적인 이미지 분류에서 Data Augmentation보다 성능이 뛰어나다는 것이 논문을 통해 밝혀졌습니다.

    # Flip
transforms.RandomHorizontalFlip(p=0.5) # 좌우 반전
transforms.RandomVerticalFlip(p=0.5) # 상하 반전
    # Rotation
transforms.RandomRotation(degrees) # 이미지를 랜덤으로 degrees 각도로 회전
transforms.RandomAffine(degrees) # 랜덤으로 affine 변형을 함 (affine은 네이버말고 구글에 쳐야 나옴)
                                 # 기하학에서, 아핀 변환은 아핀 기하학적 성질들을 보존하는 두 아핀 공간 사이의 함수이다.
transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0)
                                # Performs a random perspective transformation of the given image with a given probability.
    # Crop
transforms.CenterCrop(size) # 가운데 부분을 size 크기로 자른다.
transforms.RandomCrop(size) # 이미지를 랜덤으로 아무데나 잘라 size 크기로 출력한다.
transforms.RandomSizedCrop(size) # RandomResizedCrop으로 인해 더이상 사용되지 x
transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2) # Crop the given image to random size and aspect ratio.
    # Size
transforms.Resize(size) # 이미지 사이즈를 size로 변경한다.
    # Scaling
transforms.Scale(*args, **kwargs) # Resize로 인해 잘 사용되지 않음.
transforms.Grayscale(num_output_channels=1) # grayscale로 변환한다.
                                            # 회색조 이미지는 1비트 투톤의 흑백 이미지와는 구분되며 컴퓨터 이미징에서 볼 때
                                            # 이미지는 검은색과 흰색의 두 색만을 가지고 있다(bilevel또는 이진 이미지라고도 불린다).
                                            # 회색조 이미지는 그 사이의 많은 회색 음영을 가지고 있다.
transforms.RandomGrayscale(p=0.1) # Randomly convert image to grayscale with a probability of p (default 0.1)
    # Erasing
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        # Randomly selects a rectangle region in an image and erases its pixels.
    # Random Choice
transforms.RandomChoice([transforms.RandomGrayscale(p=0.5),
                         transforms.RandomVerticalFlip(p=0.5)]) # 랜덤으로 다음 방법들 중 선택하여 적용
    # Color Jitter(brightness(밝기), contrast(대비), saturation(채도), hue(색조), all)
transforms.ColorJitter(brightness=(0.2, 3)) # 랜덤하게 brightness를 설정. float값으로 줄 수도 있지만 tuple로 (min, max)값을 설정할 수 있음.
transforms.ColorJitter(contrast=(0.2, 3))   # 랜덤하게 contrast를 설정. float 값으로 줄수도 있지만 tuple로 (min, max)값을 설정할수도 있음
transforms.ColorJitter(saturation=(0.2, 3)) # 랜덤하게 saturation을 설정. float 값으로 줄수도 있지만 tuple로 (min, max)값을 설정할수도 있음.
transforms.ColorJitter(hue=(-0.5, 0.5))     # 랜덤하게 hue을 설정. float 값으로 줄수도 있지만 tuple로 (min, max)값을 설정할수도 있음.
                                            # hue의 경우는 -0.5 ~ 0.5 사이의 값을 해야 함.
transforms.ColorJitter(brightness=(0.2, 2), # all
                       contrast=(0.3, 2),
                       saturation=(0.2, 2),
                       hue=(-0.3, 0.3))
   # Lambda
def random_rotate2d(img):
    rand = random.randrange(0, 360, 90)
    img = ndimage.interpolation.rotate(img, rand, reshape=False, order=0, mode='reflect')
    return img
transforms.Lambda(lambda x: random_rotate2d(x))

    #  Cutout
 - Cutout으로 사용할 정사각형의 크기는 실험적으로 정해야 함.
   각 데이터셋마다 최적의 정사각형의 길이가 다름.
   코드 구현은 Python 라이브러리인 Numpy를 사용하여 구현함. 본 논문과 동일하게 사각형의 mask를 사용하였으며 사각형의 길이는
   함수의 인자인 cut_length를 통해 받아옴. 그 뒤 입력 이미지에서 임의의 좌표 값을 정해주면 그 좌표를 중심으로 하는
   사각형 영역을 0으로 채워주는 방식임. 구현이 간단하며 쉽게 사용이 가능하리라 판단됨.
def cutout(images, cut_length):
    """
    Perform cutout augmentation from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param cut_length: int, the length of cut(box).
    :return: np.ndarray, shape: (N, h, w, C).
    """

    H, W, C = images.shape[1:4]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        image_mean = image.mean(keepdims=True)
        image -= image_mean

        mask = np.ones((H, W, C), np.float32)

        y = np.random.randint(H)
        x = np.random.randint(W)
        length = cut_length

        y1 = np.clip(y - (length // 2), 0, H)
        y2 = np.clip(y + (length // 2), 0, H)
        x1 = np.clip(x - (length // 2), 0, W)
        x2 = np.clip(x + (length // 2), 0, W)

        mask[y1: y2, x1: x2] = 0.
        image = image * mask

        image += image_mean
        augmented_images.append(image)

    return np.stack(augmented_images)    # shape: (N, h, w, C)


    # Cutmix
 # https://paperswithcode.com/paper/cutmix-regularization-strategy-to-train
 # 위에 사이트에서 원하는 cutmix 방법 찾아서 다운받아서 써야함!!!
pip install git+https://github.com/ildoonet/cutmix # 설치
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
...
dataset = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_train)
dataset = CutMix(dataset, num_class=100, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.
...

criterion = CutMixCrossEntropyLoss(True)
for _ in range(num_epoch):
    for input, target in loader:    # input is cutmixed image's normalized tensor and target is soft-label which made by mixing 2 or more labels.
        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Data Augmentation Ex - 1
mnist_train = dset.MNIST("./", train=True,
                         transform = transforms.Compose([
                             transforms.Resize(34),                             # 원래 28x28인 이미지를 34x34로 늘립니다.
                             transforms.CenterCrop(28),                         # 중앙 28x28를 뽑아냅니다.
                             transforms.RandomHorizontalFlip(),                 # 랜덤하게 좌우반전 합니다.
                             transforms.Lambda(lambda x: x.rotate(90)),         # 람다함수를 이용해 90도 회전해줍니다.
                             transforms.ToTensor(),                             # 이미지를 텐서로 변형합니다.
                         ]),
                         target_transform=None,
                         download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)







    # 기본 CNN Ex - 1
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),  # 28
            nn.ReLU(),
            #nn.Dropout2d(0.2), # 오버피팅하지 않는 상태에서 정형화나 드롭아웃을 넣으면 오히려 학습이 잘 안됨.
            nn.Conv2d(16,32,3,padding=1), # 28
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),            # 14
            nn.Conv2d(32,64,3,padding=1), # 14
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2)             # 7
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100,10)
        )

    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)
        return out

---------- RNN ----------



---------- Import Codes ----------

# Torch
import torch
import torch.nn as nn # 신경망 모델들이 포함되어 있다.
import torch.optim as optim # 경사하강법 알고리즘들이 들어있다
import torch.nn.init as init # 텐서를 초기화 하는 함수들이 들어가 있습니다. (ex. uniform, normal, xavier 등등)
                             # Weight, Bias 등 딥러닝 모델에서 초깃값으로 설정되는 요소에 대한 모듈인 init를 임포트합니다.
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
 - The weight_decay parameter adds a L2 penalty to the cost which can effectively lead to to smaller model weights.
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)




    # Dropout
 - 드롭아웃은 특정 뉴런의 확률 p를 0으로 바꾸는 것
 - 그렇게 되면 이전 층에서 계산된 값들이 전달되지 않게 됨.
 - 이를 모델 수용력의 관점으로 보면 p만큼 각 층의 수용력을 줄임으로써 전체 모델의 수용력을 줄이는 것을 볼 수 있음.
 - 전체 모델이 가지고 있는 수용력이 드롭아웃으로 인해 매번 감소하기 때문에 오버피팅 될 확률이 줄어들게 됨.
 - 드롭아웃은 수용력이 낮은 모델들의 앙상블 개념으로도 볼 수 있음. 학습 때 생기는 낮은 수용력의 모델들이 테스트할 때는 드롭하지 않음으로써 합쳐지기 때문.
 - Dropout layers
   º nn.Dropout
   º nn.Dropout2d
   º nn.Dropout3d
   º nn.AlphaDropout

 - Dropout은 신경망의 학습 과정 중 Layer의 노드를 랜덤하게 Drop함으로써 Generalization 효과를 가져오게 하는 테크닉입니다.
   Dropout를 적용한다는 것은 Weight Matrix에 랜덤하게 일부 Column에 0을 집어넣어 연산을 한다고 이해하면 됩니다.
   Input Data에 설정하여 Epoch마다 Column을 랜덤하게 Dropout할 수 있고, Hidden Layer에 적용하여 Weight Matrix의 Column에도 랜덤하게 Dropout를 적용할 수 있습니다.
   이러한 방식으로 연산하여 과적합을 어느 정도 방지할 수 있습니다. Dropout을 적용한 신경망이 적용하지 않은 신경망에 비해 Test Error가 낮다는 것을 알 수 있습니다.
   이는 RandomForest의 개념과 비교해 볼 수도 있습니다. 신경망의 한 Epoch을 하나의 모델로 보고 Dropout을 핸덤한 변수의 구성으로 본다면 Dropout을 적용한 신경망은 일종의 RandomForest와 비슷한 모델 구성이라 볼 수 있습니다.

 - Dropout은 학습 과정 속에서 랜덤으로 노드를 선택해 가중값이 업데이트되지 않도록 조정하지만,
   평가 과정 속에서는 모든 노드를 이용해 Output을 계산하기 떼문에 학습 상태와 검증 상태에서 다르게 적용돼야 함.




    # Initialization 초기화
 - 신경망은 처음에 Weight를 랜덤하게 초기화하고 Loss가 최소화되는 부분을 찾아갑니다.
   최적의 가중치가 존재한다고 가정하면 그 가중치 역시 어떠한 값이기 때문에 그 최적의 값과 가까운 지점에서 시작할수록 빠르게 수렴할 수 있습니다(학습 속도가 빨라짐)
   하지만 최적의 지점 자체가 우리가 모르는 어떤 목푯값이기 때문에 근처에서 시작한다는 말은 성립할 수 없습니다.
 - ★대신 모델이 학습되는 도중에 기울기 소실현상이나 기울기 과다와 같은 현상을 최소한 겪지 않게 하거나 손실함수 공간을 최적화가 쉬운 형태로 바꾸는 방법을 택합니다.
   이러한 방법 중 하나로 가중치의 '초기화'가 있고 그 중 대표적인 방법으로는 'Xavier Glorot 초기화'와 'Kaiming He 초기화'가 있습니다.

    # Xavier initialization
 - 기존의 무작위 수로 초기화와 다르게 Layer의 특성에 맞춰서 초기화하는 방법.
   가중치의 초깃값을 N(0, var=2/(n_in + n_out)) 에서 뽑는다는 게 핵심. (n_in과 n_out은 해당 레이어에 들어오는 특성의 수, 나가는 특성의 수를 의미)
 - ★ 데이터가 몇 개의 레이어를 통과하더라도 활성화 값이 너무 커지거나 너무 작아지지 않고 일정한 범위 안에 있도록 잡아주는 초기화 방법.
   Xavier의 경우 기존보다 좋은 성능을 보이지만 ReLU에서 출력 값이 0으로 수렴하는 문제가 종종 발생하였고 이를 해결하기 위해 나온 방법이 He initialization임.

    # He Initialization
 - 가중치를 N(0, var=2/((1+a^2) * n_in)에서 샘플링. a는 렐루 또는 리키 렐루의 음수 부분의 기울기에 해당.
   기본적으로 렐루를 사용한다는 가정하에 기본값은 0으로 지정되어 있음.

    # initialization reference
 - ★ 시그모이드나 하이퍼볼릭 탄젠트를 주로 사용하는 경우는 Xavier initialization를 사용하고,
      렐루를 주로 사용하면 He Initialization을 사용하는 것이 학습에 유리
 - 파이토치 내의 nn.linear는 Output으로 계산되는 벡터의 차원 수의 역수 값에 대한 +/- 범위 내 uniform distribution을 설정해 샘플링합니다.

    # initialization Ex - 1
 - nn.Module을 상속하여 모델을 만들고, self.modules() 함수를 사용하여 모델 내부의
   모듈들을 차례대로 돌면서 해당 모듈이 어떤 연산인지에 따라 초깃값을 nn.init 함수를
   사용하여 초기화할 수 있습니다.
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),  # 28 x 28
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2,2),            # 14 x 14
            nn.Conv2d(32,64,3,padding=1), # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2,2)             #  7 x 7
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

# 모델의 모듈을 차례대로 불러옵니다.
        for m in self.modules():
            # nn.Conv2d - Ex
            if isinstance(m, nn.Conv2d):
            # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)

            # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                init.xavier_normal_(m.weight.data)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)

            # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                init.kaiming_normal_(m.weight.data)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)

            # 만약 그 모듈이 nn.Linear인 경우
            elif isinstance(m, nn.Linear):
            # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)

            # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                init.xavier_normal_(m.weight.data)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)


            # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                init.kaiming_normal_(m.weight.data)
                # 편차를 0으로 초기화합니다.
                m.bias.data.fill_(0)

    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)
        return out

    # initialization Ex - 2
def weight_init(m):  # MLP 모델 내의 Weight를 초기화할 부분을 설정하기 위해 weight_init 함수를 정의합니다.
    if isinstance(m, nn.Linear): # MLP 모델을 구성하고 있는 파라미터 중 nn.Linear에 해당하는 파라미터 값에 대해서만 지정합니다.
        init.kaiming_uniform_(m.weight.data) # nn.Linear에 해당하는 파라미터 값에 대해 he_initailization을 이용해 파라미터 값을 초기화합니다.
      # init.xavier_normal_(m.weight.data)
      # init.normal_(m.weight.data, std=0.1)
model = Net().to(device)
model.apply(weight_init) # 정의한 weight_init 함수를 Net() 클래스의 인스턴스인 model에 적용합니다.
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()
print(model) # 초기화를 진행할 때, 정의된 weight_init 함수를 보면 모델 내 파라미터 값 중 nn.Linear 인스턴스에 대해서는 kaiming_uniform_을 이용해
             # 초기화하는 것으로 설정돼 있습니다. 여기서 kaiming_uniform_는 'He Initialization'을 의미합니다.

    # initialization Ex - 3
x1 = init.uniform_(torch.FloatTensor(3,4),a=0,b=9)  # uniform 분포를 따라 텐서를 초기화 할 수 있습니다.
x2 = init.normal_(torch.FloatTensor(3,4),std=0.2)   # 정규 분포를 따라 텐서를 초기화 할 수 있습니다.
x3 = init.constant_(torch.FloatTensor(3,4),3.1415)  # 지정한 값으로 텐서를 초기화 할 수 있습니다.
x1,x2,x3

    # initialization Ex - 4
linear1 = nn.Linear(784, 256, bias=True)
linear2 = nn.Linear(256, 256, bias=True)
linear3 = nn.Linear(256, 10, bias=True)
relu = nn.ReLU()
nn.init.xavier_uniform_(linear1.weight)
nn.init.xavier_uniform_(linear2.weight)
nn.init.xavier_uniform_(linear3.weight)




    # Learning Rate Scheduler
 - 학습률을 점차 떨어뜨리는 방법을 학습률 부식이라고 함.
   LR(Learning Rate) Scheduler는 미리 지정한 횟수의 epoch이 지날 때마다 lr을 감소(decay)시켜준다.
   이는 학습 초기에는 빠르게 학습을 진행시키다가 minimum 근처에 다다른 것 같으면 lr을 줄여서 더 최적점을 잘 찾아갈 수 있게 해주는 것이다.

   # lr Scheduler의 종류:
º optim.lr_scheduler.LambdaLR # lambda 함수를 하나 받아 그 함수의 결과를 lr로 설정한다.
º optim.lr_scheduler.StepLR   #특정 step마다 lr을 gamma 비율만큼 감소시킨다.
  scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma= 0.99)  # 지정한 스텝 단위로 학습률에 감마를 곱해 학습률을 감소시킵니다.
º optim.lr_scheduler.MultiStepLR # StepLR과 비슷한데 매 step마다가 아닌 지정된 epoch에만 gamma 비율로 감소시킨다,
                                 # step_size를 milestones 인수에 리스트로 받아서 원하는 지점마다 학습률을 감소시킴
  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,80], gamma= 0.1)  # 지정한 스텝 지점(예시에서는 10,30,80)마다 학습률에 감마를 곱해줍니다.
º optim.lr_scheduler.ExponentialLR # 매 에폭마다 학습률에 gamma를 곱해 lr을 지수함수적으로 감소시킨다.
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99) # 매 epoch마다 학습률에 감마를 곱해줍니다.
º optim.lr_scheduler.CosineAnnealingLR # lr을 cosine 함수의 형태처럼 변화시킨다. lr이 커졌다가 작아졌다가 한다.
º optim.lr_scheduler.ReduceLROnPlateau # 이 scheduler는 다른 것들과는 달리 학습이 잘 되고 있는지 아닌지에 따라 동적으로 lr을 변화시킬 수 있다.
                                       # 보통 validation set의 loss를 인자로 주어서 사전에 지정한 epoch동안 loss가 줄어들지 않으면 lr을 감소시키는 방식이다.
 - 각 scheduler는 공통적으로 last_epoch argument를 갖는다.
   Default value로 -1을 가지며, 이는 초기 lr을 optimizer에서 지정된 lr로 설정할 수 있도록 한다.

   # lr scheduler Ex - 1
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)

   # lr scheduler Ex - 2
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma= 0.99)  # 지정한 스텝 단위로 학습률에 감마를 곱해 학습률을 감소시킵니다.
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,80], gamma= 0.1)  # 지정한 스텝 지점(예시에서는 10,30,80)마다 학습률에 감마를 곱해줍니다.
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99) # 매 epoch마다 학습률에 감마를 곱해줍니다.
for i in range(num_epoch):
    scheduler.step()
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(i, loss, scheduler.get_lr())





    #  Normalization (입력 데이터의 정규화)
 - 정규화의 사전적 의미:
   정규화 또는 정상화(normalization)는 어떤 대상을 일정한 규칙이나 기준에 따르는 ‘정규적인’ 상태로 바꾸거나,
   비정상적인 대상을 정상적으로 되돌리는 과정을 뜻한다. 정규화 및 정상화라는 용어는 여러 분야에서 다음과 같이 사용된다.
 - 학습 데이터에서는 잘 동작하는데 테스트 데이터에서는 학습이 제데로 안 된다면, 단순히 오버피팅 문제가 아니라 두 데이터의 분포가 달라서인 경우도 있습니다.
    1. 다음 그림에서 왼쪽이 학습 데이터, 오른쪽이 테스트 데이터라고 하면, 학습 시에 결과가 잘 나오던 모델도 테스트 시에는 결과가 좋지 않게 나올 수 밖에 없을 것입니다.
    2. 학습 시에도 데이터 간의 분포가 다르다면 각 분포에 맞춰 변수가 업데이트될 테니 그 데이터를 그대로 쓰면 학습조차 제대로 안 될 것입니다.
 - 이럴 때 필요한 게 바로 정규화입니다.
    (1) 데이터를 정규화는 방법에는 여러가지가 있는데 대표적인 방법으로는 표준화(stardardization)가 있습니다.
    - 표준화의 과정: 정규화하기 전 -> x_ = x - μ -> x_ = (x - μ) / σ
    - 표준화는 데이터에서 평균을 빼고 표준편차로 나눠주는 과정을 거치는데 이렇게 되면 평균은 0, 분산은 1이 되어 데이터 분포가 표준정규분포화됩니다.
    - ★이렇게 되면 네트워크에 들어오는 입력값이 일정한 분포로 들어오기 때문에 학습에 유리합니다.
    - 데이터로 모델을 학습할 때도 전체 데이터셋의 평균과 분산을 구해 데이터를 표준화하고 그 값을 입력으로 전달합니다.
    (2) 표준화 이외에도 많이 사용되는 정규화 방법 중 최소극대화(minmax)정규화가 있습니다.
    - 최소극대화 정규화는 데이터를 주로 0에서 1사이로 압축하거나 늘리는 방법으로 데이터에서 최솟값을 빼준 값을 데이터의 최대값과 최솟값의 차이로 나눠줌으로써 변형합니다
    - x = (x - x.min()) / (x.max() - x.min())
    - 이렇게 되면 0에서 1사이 밖에 있는 값들은 0과 1사이로 압축되고, 전체 범위가 1이 안 되던 값들은 0과 1사이로 늘어나게 됩니다.
    - 하지만 평균적 범위를 넘어서는 너무 작거나 너무 큰 이상치가 있는 경우에는 오히려 학습에 방해가 되기도 합니다.

    # 정규화를 하면 학습이 더 잘되는 이유
 - 데이터가 정규화되지 않았을 때는 업데이트 과정에서 지그재그 모양으로 불필요한 업데이트가 많고 업데이트 횟수도 많이 필요합니다.
   ★하지만 정규화된 손실 그래프는 원형에 가까운 형태를 가지기 때문에 불필요한 업데이트가 적고 더 큰 학습률을 적용할 수 있습니다.
 - (자세한설명★)데이터가 정규화가 되지 않았다면 데이터의 각 요소별로 범위가 다를 것입니다. (feature 별로 범위가 다를 것)
   그렇게 되면 모델을 학습시킬 때, 이상적으로 어떤 값은 크게 업데이트하고 어떤 값은 비교적 작은 수치로 업데이트 해야 빠른 시간안에
   손실이 최소가 되는 지점에 도달할 것입니다. 하지만 각 변수마다 범위가 다르기 때문에 어떤 변수를 기준으로 학습률을 정하는지에 따라,
   어떤 변수는 손실 최소 지점을 중심에 두고 왔다 갔다 할 것입니다. 이에 비해 정규화된 데이터는 변수들의 범위가 일정하기 때문에
   비교적 높은 학습률을 적용시킬 수 있고 따라서 최소 지점에 더 빠르게 도달할 수 있게 됩니다.

    # Normalization - Ex 1
# 정규화는 transform을 통해 가능합니다.
# 여기서 mean, std는 미리 계산된 값입니다.
transforms.Normalize(mean, std, inplace=False) # 이미지를 정규화한다.

# Normalization in the cifar100 example
elif args.dataset == "cifar100":
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
#print(vars(train_dataset))
    print(train_dataset.data.shape)
    print(np.mean(train_dataset.data, axis=(0,1,2))/255)
    print(np.std(train_dataset.data, axis=(0,1,2))/255)
elif args.dataset == "mnist":
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
#print(vars(train_set))
    print(list(train_dataset.data.size()))
    print(train_dataset.data.float().mean()/255)
    print(train_dataset.data.float().std()/255)

# CIFAR10 standardization code
mean = [0.4913, 0.4821, 0.4465], std = [0.2470, 0.2434, 0.2615]
standardization = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
# CIFAR100 standarization code
mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
standardization = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# MINST standardization code
standardization = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])


    # Normalization - Ex 2
mnist_train = dset.MNIST("./", train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                         ]),
                         target_transform=None,
                         download=True)
mnist_test = dset.MNIST("./", train=False,
                        transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                        ]),
                        target_transform=None,
                        download=True)




    # Batch Normalization - 사전설명
 1. Internal Covariant Shift
 - 입력 데이터의 분포가 달랐던 것처럼 하나의 신경망에 대해서 입력의 범위가 바뀌는 것을 공변량 변화(covariate shift)라고 하는데
   딥러닝 모델 내부에서도 하나의 은닉층에 여러 범위의 입력이 들어오는 내부 공변량 변화(internal covariate shift)가 일어나게 됩니다.
    -> 내부 공변량 변화: 학습 과정에서 계층 별로 입력의 데이터 분포가 달라지는 현상
    -> 각 계층에서 입력으로 feature를 받게 되고 그 feature는 convolution이나 fully connected 연산을 거친 뒤 activation function을 적용하게 됩니다.
         그러면 연산 전/후에 데이터 간 분포가 달라질 수가 있습니다.
 2. Batch 단위로 학습을 하게 되면 Batch 단위간에 데이터 분포의 차이가 발생할 수 있습니다.
    즉, Batch 간의 데이터가 상이하다고 말할 수 있는데 위에서 말한 Internal Covariant Shift 문제입니다.
    -> 이 문제를 개선하기 위한 개념이 Batch Normalization 개념이 적용됩니다.

    # Batch Normalization - 정의설명
 - 배치 정규화는 말 그대로 한 번에 입력으로 들어오는 배치 단위로 정규화하는 것을 의미합니다.
   배치 정규화의 알고리즘을 살펴보면, 먼저 입력에 대해 평균과 분산을 구하고 정규화를 합니다.(각 배치 단위 별로 데이터가 다양한 분포를 가지더라도 각 배치별로 평균과 분산을 이용해 정규화하는 것)
   그리고 정규화된 데이터를 스케일 및 시프트(scale & shift)하여 다음 레이어에 일정한 범위의 값들만 전달되게 합니다.
    -> y_i = r*x^_i + β = BN_r,β(xi)
 - 베타와 감마가 없다고 가정하면 정규화하는 수식과 일치합니다. 베타와 감마는 backprpagation을 통하여 학습을 하게 됩니다.
   배치 정규화는 학습 시 배치 단위의 평균과 분산들을 차례대로 받아 이동평균과 이동분산을 저장해놓았다가 테스트할 때는
   해당 배치의 평균과 분산을 구하지 않고 구해놓았던 평균과 분산으로 정규화합니다. -> eval()함수 이용.
   (자세한 정보: https://gaussian37.github.io/dl-concept-batchnorm/)
 -  batch normalization은 activation function 앞에 적용됩니다.
    batch normalization을 적용하면 weight의 값이 평균이 0, 분산이 1인 상태로 분포가 되어지는데,
    이 상태에서 ReLU가 activation으로 적용되면 전체 분포에서 음수에 해당하는 (1/2 비율) 부분이 0이 되어버립니다.
    기껏 정규화를 했는데 의미가 없어져 버리게 됩니다.
    따라서 γ,β가 정규화 값에 곱해지고 더해져서 ReLU가 적용되더라도 기존의 음수 부분이 모두 0으로 되지 않도록 방지해 주고 있습니다.
    물론 이 값은 학습을 통해서 효율적인 결과를 내기 위한 값으로 찾아갑니다.
 - 최근에는 데이터 하나당 정규화를 하는 인스턴스 정규화나 은닉층 가중치 정규화 등의 방법이 소개 되기도 했습니다.
 - Batch Normalization은 1-Dimension, 2-Dimension, 3-Dimension 등 다양한 차원에 따라 적용되는 함수명이 다르기 때문에 유의해서 사용해야 합니다.
 - nn.BatchNorm() 함수를 이용해 적용하는 부분은 논문이나 코드에 따라 activation function 이전에 적용하는지, 이후에 적용하는지 연구자들의 선호도에 따라 다르게 이용됩니다.
    -> 아직도 어디에 적용하는 것이 나은지 논쟁중

    # Batch Normalization 장점
 - 은닉층 단위마다 배치 정규화를 넣어주면 내부 공변량의 변화를 방지하는 동시에 정규화의 장점인 더 큰 학습률의 사용도 가능해집니다.
 - 학습 단계에서 모든 Feature에 정규화를 해주게 되면 정규화로 인하여 Feature가 동일한 Scale이 되어 learning rate 결정에 유리해집니다.
   왜냐하면 Feature의 Scale이 다르면 gradient descent를 하였을 때, gradient가 다르게 되고 같은 learning rate에 대하여 weight마다 반응하는 정도가 달라지게 됩니다.
   gradient의 편차가 크면 gradient가 큰 weight에서는 gradient exploding이, 작으면 vanishing 문제가 발생하곤 합니다.
   하지만 정규화를 해주면 gradient descent에 따른 weight의 반응이 같아지기 때문에 학습에 유리해집니다.
 - Input 분포를 정규화해 학습 속도를 빠르게 해줍니다.
 - 학습 속도를 향상시켜주고 Gradient Vanishing / Exploding 문제도 완화해줍니다.
 - 분포를 정규화해 비선형 활성 함수의 의미를 살리는 개념이라 볼 수 있습니다. (Batch Normalization을 사용하지 않는다면 Hidden Layer를 쌓으면서 비선형 활성 함수를 사용하는 의미가 없어질 가능성도 있습니다.)
논문에서 주장하는 Batch Normalization의 장점은 다음과 같다.
 - 기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다.
   이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다.
   따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.
 - Batch Normalization의 경우 자체적인 regularization 효과가 있다.
   이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며,
   나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.)
   Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다.


   # Batch Normalization을 CNN에 적용시키고 싶을 경우 지금까지 설명한 방법과는 다소 다른 방법을 이용해야만 한다.
 - 먼저, convolution layer에서 보통 activation function에 값을 넣기 전 Wx+b 형태로 weight를 적용시키는데,
   ★Batch Normalization을 사용하고 싶을 경우 normalize 할 때 beta 값이 b의 역할을 대체할 수 있기 때문에 b를 없애준다.
 - 또한, CNN의 경우 convolution의 성질을 유지시키고 싶기 때문에, 각 channel을 기준으로 각각의 Batch Normalization 변수들을 만든다.
   예를 들어 m의 mini-batch-size, n의 channel size 를 가진 Convolution Layer에서 Batch Normalization을 적용시킨다고 해보자.
   convolution을 적용한 후의 feature map의 사이즈가 p x q 일 경우, 각 채널에 대해 m x p x q 개의 각각의 스칼라 값에 대해
   mean과 variance를 구하는 것이다. 최종적으로 gamma와 beta는 각 채널에 대해 한개씩 해서 총 n개의 독립적인 Batch Normalization 변수들이 생기게 된다.


   # CNN Batch Normalization Ex - 1
# 입력 데이터를 정규화하는것처럼 연산을 통과한 결과값을 정규화할 수 있습니다.
# 그 다양한 방법중에 대표적인것이 바로 Batch Normalization이고 이는 컨볼루션 연산처럼 모델에 한 층으로 구현할 수 있습니다.
# https://pytorch.org/docs/stable/nn.html?highlight=batchnorm#torch.nn.BatchNorm2d
# nn.BatchNorm2d(x)에서 x는 입력으로 들어오는 채널의 개수입니다.
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),  # 28 x 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), # 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),            # 14 x 14
            nn.Conv2d(32,64,3,padding=1), # 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)             #  7 x 7
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)
        return out















---------- 공부하며 떠오른 것들 ----------

학습시 언더피팅나면
 - 모델 수용력 늘리기
학습시 오버피팅나면
 - 모델 수용력 줄이기
    -> 수용력: 모델 노드 수, 모델 깊이


오버피팅 시
 - 1. 적정한 강도로 정형화, 규제(Regularzation)을 손실함수에 걸어주기
 - 2. 최적화함수에 가중치부식 인수를 주기
 - 3. 드랍아웃 적용
 - 4. 모델의 수용력 줄이기


기울시 소실이나 과다가 일어날 경우
 - 모델에 Initialization 적용 (데이터가 몇 개의 레이어를 통과하더라도 활성화 값이 너무 커지거나 너무 작아지지 않고 일정한 범위 안에 있도록 잡아줌)
 - Batch Normalization 적용 (grdient descent를 하였을 때, gradient의 편차가 갑자기 커지거나 작아지는 것 방지)


높은 학습률과 빠른 학습을 위해
 - 입력 데이터 정규화
 - Initialization 적용
 - Batch Normalization 적용


학습은 잘 되었는데 테스트 데이터에 대해서 성능이 잘 안나올 때
 - 분포가 데이터간의 분포가 달라서 그럴 수 있으니 그럴때는 데이터에 대한 정규화 진행


스케일링은 정규화와 다르다.
 - 스케일링은 분포의 모양이 변하진 않지만 정규화는 분포의 모양이 변하며 예시로는 ML에서 자주 사용했던 log1p화가 있다
   그래서 안정적인 분포로 피처들의 왜도를 정규분포처럼 바꾸어 주어서 log1p를 사용하는 것





    # Optimizer 정의
 - 옵티마이저란 최적화 함수(optimization function)라고도 하며,
   경사하강법을 적용하여 오차를 줄이고 최적의 가중치와 편차를 근사할 수 있게 하는 역할을 합니다.
 - ★GPU CUDA를 사용할 계획이라면 optimizer를 정의하기 전에 model.cuda()를 미리 해놓아야 한다.
   공식 홈페이지에 따르면,
   If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
   Parameters of a model after .cuda() will be different objects with those before the call.
   In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used.
   이유를 설명하자면
   optimizer는 argument로 model의 parameter를 입력받는다.
   .cuda()를 쓰면 모델의 parameter가 cpu 대신 gpu에 올라가는 것이므로 다른 object가 된다.
   따라서 optimizer에 model parameter의 위치를 전달한 후 model.cuda()를 실행하면, 학습시켜야 할 parameter는 GPU에 올라가 있는데
   optimizer는 cpu에 올라간 엉뚱한 parameter 위치를 참조하고 있는 것이 된다. 그러니 순서를 지키자.

    # optimizer에 대해 알아 두어야 할 것들
 1. optimizer는 step() method를 통해 argument로 전달받은 parameter를 업데이트한다.
 2. 모델의 parameter별로(per-parameter) 다른 기준(learning rate 등)을 적용시킬 수 있다.
 3. torch.optim.Optimizer(params, defaults)는 모든 optimizer의 base class이다.
 4. nn.Module과 같이 state_dict()와 load_state_dict()를 지원하여 optimizer의 상태를 저장하고 불러올 수 있다.
 5. zero_grad() method는 optimizer에 연결된 parameter들의 gradient를 0으로 만든다.
 6. torch.optim.lr_scheduler는 epoch에 따라 learning rate를 조절할 수 있다.

    # Optimizer의 종류:
 º optim.Adadelta,
 º optim.Adagrad,
 º optim.Adam, optim.SparseAdam, optim.Adamax,
 º optim.ASGD
 º optim.LBFGS(LBFGS는 per-parameter 옵션이 지원되지 않는다. 또한 memory를 다른 optimizer에 비해 많이 잡아먹는다고 한다.),
 º optim.RMSprop, optim.Rprop,
 º optim.SGD
 º ★ Leaky ReLU, ELU, prametric ReLU, SELU, SERLU


    # Optimizer 종류 설명 - Ex 1
    # SGD
SGD: Batch 단위로 Back Propagation을 함

    # Momentum
 - momemtum은 미분을 통한 gradient 방향으로 가되, 일종의 관성을 추가하는 개념입니다.
   일반적인 SGD는 조금씩 최저 해(Global Optinum)를 찾아갑니다. 전체 데이터에 대해 Back Propagation을 하는 것이 아니라
   Batch 단위로 Back Propagation하기 때문에 일직선으로 찾아 가지 않습니다.
   momemtum을 사용하면 최적의 장소로 더 빠르게 수렴할 수 있습니다. 걸어가는 보폭을 크게 하는 개념이라 이해하면 될 것 같습니다.
   또한 최적 해가 아닌 지역해(Local Minimum)를 지나칠 수도 있다는 장점이 있습니다.

    # NAG
NAG(Nesterov Accelerated Gradient)
 - NAG는 Momemtum을 약간 변형한 방법으로, Momemtum으로 이동한 후 Gradient를 구해 이동하는 방식입니다.

    # Adagrad
Adagrad(Adaptive Gradient)
 - Adagrad의 개념은 '가보지 않은 곳은 많이 움직이고 가본 곳은 조금씩 움직이자' 입니다.
   학습을 통해 크게 변동이 있었던 가중치에 대해서는 학습률을 감소시키고 학습을 통해 아직 가중치의 변동이 별로 없었던 가중치는 학습률을 증가시켜서 학습이 되게끔 한다.

    # RMSProp
 - RMSProp는 Adagrad의 단점을 보완한 방법입니다. Adagrad의 단점은 학습이 오래 진행될수록 부분이 계속 증가해 Step Size가 작아진다는 것인데,
   RMSPorp는 G가 무한히 커지지 않도록 지수 평균을 내 계산합니다.

    # Adadelta
Adadelta(Adaptive Delta)
 - Adadelta 또한 Adagrad의 단점을 보완한 방법입니다. Gradient를 구해 움직이는데,
   Gradient의 양이 너무 적어지면 움직임이 멈출 수 있습니다. 이를 방지하기 위한 방법이 Adadelta입니다.

    # Adam
Adam(Adaptive Moment Estimation)
 - Adam은 딥러닝 모델을 디자인할 때, 기본적으로 가장 많이 사용하는 optimizer로,
   RMSProp와 Momentum 방식의 특징을 잘 결합한 방법입니다.
   2020년을 기준으로 많은 딥러닝 모델에서 기본적으로 Adam을 많이 사용하고 있습니다.

    # RAdam
RAam(Rectified Adam optimizier)
 - Adam뿐 아니라 대부분의 Optimizer는 학습 초기에 Bad Local Optimum에 수렴해 버릴 수 있는 단점이 있습니다.
   학습 초기에 Gradient가 매우 작아져서 학습이 더 이상 일어나지 않는 현상이 발생하는 것입니다.
   RAdam은 이러한 Adaptive Learning Rate Term의 분산을 교정(Recify)하는 Optimizer로
   논문의 저자는 실험 결과를 통해 Learning Rate를 어떻게 조절하든 성능이 비슷하다는 것을 밝혔습니다.
(출처 : https://www.slideshare.net/yongho/ss-79607172)


    # Optimizer 종류 설명 - Ex 2
GD: 모든 자료를 다 검토해서 내 위치의 산기울기를 계산해서 갈 방향 찾겠다.
SGD: 전부 다봐야 한 걸음은 너무 오래 걸리니까 조금만 보고 빨리 판단한다. 같은 시간에 더 많이 간다.

 - 스텝방향
Momemtum: 스텝 계산해서 움직인 후, 아까 내려오던 관성 방향으로 또 가자
Nag: 일단 관성 방향으로 움직이고, 움직인 자리에 스텝을 계산하니 더 빠르더라

 - 스텝사이즈
Adagrad: 안가본 곳은 성큼 빠르게 걸어 훑고 많이 가본 곳은 잘 아니까 갈수록 보폭을 줄여 세밀히 탐색
AdaDelta: 종종걸음 너무 작아져서 정지하는 걸 막아보자
RMSProp: 보폭을 줄이는 건 좋은데 이전 맥락 상황봐가며 하자.

 - 스텝 방향, 스텝사이즈
Adam: RMSProp + Momemtum, 방향도 스텝사이즈도 적절하게!
Nadam: Adam에 Momemtum 대신 NAG를 붙이자.





---------- 혼합아이디어 ----------









───────────────────────────────────────────────────────────────
