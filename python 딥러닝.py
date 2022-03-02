


# 목차
* Pytorch Basic
* DNN
* CNN
* RNN
* AE
* GAN
* Object Detection
* Style Transfer
* Import codes
* 딥러닝의 발전을 이끈 알고리즘들
* 공부하며 떠오른 것들
* 혼합아이디어
* 기타 설명







---------- Pytorch Basic ----------


    # Random Numbers
# 0에서 1사이의 랜덤한 숫자
torch.rand()
x = torch.rand(2,3)


# 정규분포에서 샘플링한 값
torch.randn()
randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
x = torch.randn(2,3)

# 시작과 끝 사이의 랜덤한 자연수(시작 포함, 끝 불포함)
torch.randint()
x = torch.randint(2,5, size=(2,3))



    # Zeros & Ones
torch.ones(): 1으로 채워진 텐서
x = torch.ones(2,3)

torch.zeros(): 0으로 채워진 텐서

ones_like(x): x와 같은 shape이고 1로만 가득찬 텐서를 반환
zeros_like(x): x와 같은 shape이고 0로만 가득찬 텐서를 반환
    # Zeros & Ones -Ex 1
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))
    # Zeros & Ones -Ex 2
x = torch.ones(2, 2, requires_grad=True) # 2 x 2 tensor를 생성하고 requires_grad=True를 설정하여 연산을 기록함.
print(x)



    # Tensor Data Type
tensor.type(): 해당 텐서의 타입을 리턴하고 type(tensor)는 토치의 텐서 클래스라는 것을 리턴함
x = torch.rand(2,3)
print(x.type())
print(type(x))

# tensor.type()을 dtype과 함께 사용하면 텐서의 데이터 타입을 dtype에 넣어준 데이터 타입으로 바꿔줍니다.
double_x = x.type(dtype=torch.DoubleTensor)
print(double_x.type())

# type_as
# tensor.type_as(): type_as라는 함수를 사용해 데이터타입을 바꿀 수 있습니다.
int_x = x.type_as(torch.IntTensor())
print(int_x.type())



    # Numpy to Tensor, Tensor to Numpy
# torch.from_numpy(): 넘파이 배열을 토치텐서로 바꿀 수 있습니다.
x1 = np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6]))
x2 = torch.from_numpy(x1)
x2,x2.type()

# tensor.numpy(): 토치 텐서를 .numpy()를 통해 넘파이 배열로 바꿀 수 있습니다.
x3 = x2.numpy()
x3



    # Tensor on CPU & GPU
 - 파이토치는 내부적으로 CUDA, cuDNN이라는 API를 통해 GPU를 연산에 사용할 수 있고,
   이로 인해 생기는 연산 속도의 차이는 엄청납니다.
   CUDA는 엔비디아가 GPU를 통한 연산을 가능하게 만든 API모델이며,
   cuDNN은 CUDA를 이용해 딥러닝 연산을 가속해주는 라이브러리입니다.
# CUDA: use GPU
torch.cuda.is_available(): 학습을 시킬 때는 GPU를 많이 사용한다. GPU가 사용가능한지 알 수 있다.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.device(device): 어느 device(GPU나 CPU)를 쓸 지 선택한다.
torch.cuda.device_count(): 현재 선택된 device의 수를 반환한다.
torch.cuda.init(): C API를 쓰는 경우 명시적으로 호출해야 한다.
torch.cuda.set_device(device): 현재 device를 설정한다.
torch.cuda.manual_seed(seed): 랜덤 숫자를 생성할 시드를 정한다. multi-gpu 환경에서는 manual_seed_all 함수를 사용한다.
torch.cuda.empty_cache(): 사용되지 않는 cache를 release하나, 가용 메모리를 늘려 주지는 않는다.



    # Tensor Size
# size함수를 이용해 텐서의 형태를 알 수 있음
x = torch.FloatTensor(10,12,3,3)
x.size(), x.size()[1:2]




    # Indexing
# torch.index_select(): 지정한 차원 기준으로 원하는 값들을 뽑아낼 수 있습니다.
x = torch.randn(4,3)
# dim으로 차원 설정, index로 몇번째 행 or 몇번째 열 설정
selected = torch.index_select(x, dim=1, index=torch.LongTensor([0,2]))

# torch.masked_select(): 뽑고자 하는 값들을 마스킹해서 선택할 수 있습니다.
x = torch.randn(2,3)
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
out = torch.masked_select(x,mask)
print(x, mask, out, sep="\n\n")



    # Concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0)) # 행으로 붙임 - 아래로 붙임
print(torch.cat([x, y], dim=1)) # 열로 붙임 - 옆으로 붙임



    # Stacking
# Concatenate의 기능을 좀 더 편리하게 단축해 놓은 것
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z])) # default로 dim=0으로 쌓음. 행으로 쌓음
print(torch.stack([x, y, z], dim=1)) # 열로 쌓음
## 위의 dim=0 한 코드는 아래의 코드와 같음
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
## 위의 dim=1 한 코드는 아래의 코드와 같음
print(torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1))



    # Slicing
# torch.chunk(): 텐서를 원하는 chunk 개수만큼으로 분리할 수 있습니다.
x = torch.FloatTensor([[1,2,3],[4,5,6]])
y = torch.FloatTensor([[-1,-2,-3],[-4,-5,-6]])
z1 = torch.cat([x,y],dim=0)
x_1, x_2 = torch.chunk(z1,2,dim=0)
y_1, y_2, y_3 = torch.chunk(z1,3,dim=1)
print(z1,x_1,x_2,z1,y_1,y_2,y_3,sep="\n")

# torch.split() : 원하는 사이즈로 텐서를 자를 수 있습니다.
x1,x2 = torch.split(z1,2,dim=0)
y1 = torch.split(z1,2,dim=1) # y1에 z1의 두번째 열에서 잘라서 두 개의 텐서가 저장됨 (열 2개짜리, 열 1개짜리)
print(z1,x1,x2,sep="\n")
print("\nThis is y1:")
for i in y1:
      print(i)



      # squeezing
 - 길이가 1인 차원들을 압축시킬 수 있습니다.
 - 쥐어 짜는 것
 - 자동으로 디멘션이 1인 경우 해당 디멘션을 없애줌.
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape) # output: torch.Size([3, 1])
print(ft.squeeze())
print(ft.squeeze().shape) # output: torch.Size([3])
# squeeze(dim=?)하고 ?값을 주었을 경우 해당 디멘션이 1일 경우 없애줌.
x1 = torch.FloatTensor(10,1,3,1,4)
x2 = torch.squeeze(x1)
print(x1.size(),x2.size(),sep="\n")

    # Unsqueeze
 - squeeze와 반대로 unsqueeze를 통해 차원을 늘릴수 있습니다.
 - 내가 원하는 디멘션에 1을 넣어줌.
 - 따라서 원하는 디멘션을 꼭 명시해 주어야함
 - view함수를 쓴 것과 비슷함
x1 = torch.FloatTensor(10,3,4)
x2 = torch.unsqueeze(x1, dim=0)
x3 = torch.unsqueeze(x1, dim=1)
print(x1.size(),x2.size(), x3.size(), sep="\n")
# torch.Size([10, 3, 4])
# torch.Size([1, 10, 3, 4])
# torch.Size([10, 1, 3, 4])

# -1 은 ft가 가지고 있는 마지막 dimension을 말함
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

    # Unsqueeze - Ex 1
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
 # 밑의 view과 같은 결과
print(ft.view([1, -1]))
print(ft.view([1, -1]).shape)

    # Unsqueeze - Ex 2
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

    # Unsqueeze - Ex 3
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()



    # 사칙연산
print(torch.add(scalar1, scalar2))
print(torch.sub(scalar1, scalar2))
print(torch.mul(scalar1, scalar2))
print(torch.div(scalar1, scalar2))

print(torch.add(vector1, vector2))
print(torch.sub(vector1, vector2))
print(torch.mul(vector1, vector2))
print(torch.div(vector1, vector2))
print(torch.dot(vector1, vector2))

torch.add(matrix1, matrix2)
torch.sub(matrix1, matrix2)
torch.mul(matrix1, matrix2)
torch.div(matrix1, matrix2)
# 행렬 곱 Matrix Multiplication -> matmul
torch.matmul(matrix1, matrix2)

torch.add(tensor1, tensor2)
torch.sub(tensor1, tensor2)
torch.mul(tensor1, tensor2)
torch.div(tensor1, tensor2)
torch.matmul(tensor1, tensor2)



    # Broadcasting
 - 우리가 벡터나, 행렬 등을 계산할때 계산이 가능하도록 항상 사이즈가 같은 것끼리 계산하곤 한다.
   파이토치에서 제공하는 Broadcasting기능은 사이즈가 다른 것끼리 계산할 때 사이즈를 자동적으로 맞춰서 계산하도록 한다.
 - 기존의 일반적인 곱셈의 경우에는 크기(shape)가 같아야 하지만 shape가 다른경우 BoardCasting 기능으로 계산이 될 수 있다.
# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)
output: tensor([[4., 5.]])

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
output: tensor([[4., 5.], # 1과 2에 3이 더해짐
        	    [5., 6.]])# 1과 2에 4가 더해짐

# 행렬의 곱이 연산되게 됨.
print(m1.matmul(m2)) # 2 x 1
# Broadingcasting 기능과 행렬내에서 각 자리가 같은 원소끼리 곱해져서 출력되게 됨
print(m1.mul(m2))



    # 파워 연산(x의 n승)
# torch.pow(input,exponent)
x1 = torch.FloatTensor(3,4)
torch.pow(x1,2),x1**2


    # exponential 연산
# torch.exp(tensor,out=None)
x1 = torch.FloatTensor(3,4)
torch.exp(x1)


    # 로그 연산
# torch.log(input, out=None) -> natural logarithm
x1 = torch.FloatTensor(3,4)
torch.log(x1)


    # 행렬곱 연산
# torch.mm(mat1, mat2) -> matrix multiplication
x1 = torch.FloatTensor(3,4)
x2 = torch.FloatTensor(4,5)
torch.mm(x1,x2)

    # 배치 행렬곱 연산. 맨 앞에 batch 차원은 무시하고 뒤에 요소들로 행렬곱을 합니다.
# torch.bmm(batch1, batch2) -> batch matrix multiplication
x1 = torch.FloatTensor(10,3,4)
x2 = torch.FloatTensor(10,4,5)
torch.bmm(x1,x2).size()


    # 두 텐서간의 프로덕트 연산
# torch.dot(tensor1,tensor2) -> dot product of two tensor
x1 = torch.tensor([2, 3])
x2 = torch.tensor([2, 1])
torch.dot(x1,x2)


    # 행렬의 전치
# torch.t(matrix) -> transposed matrix
x1 = torch.tensor([[1,2],[3,4]])
print(x1,x1.t(),sep="\n")


    # 차원을 지정할 수 있는 행렬의 전치 연산
# torch.transpose(input,dim0,dim1) -> transposed matrix
x1 = torch.FloatTensor(10,3,4)
print(x1.size(), torch.transpose(x1,1,2).size(), x1.transpose(1,2).size(),sep="\n")




    # 1D Array with numpy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print('Rank of t: ', t.ndim) # 몇개의 차원인지
print('Shape of t:' , t.shape) # shpae은 어떤 모양인지: 하나의 차원에 7개의 원소가 들어있어
print(t.size()) # shape
    # 2D Array with numpy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t[:, 1]) # Element
print(t[:, 1].size())
print(t[:, :-1]) # Slicing


    # Mean
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim=0)) # 행 방향으로 mean을 계산
print(t.mean(dim=1)) # 열 방향으로 mean을 계산
print(t.mean(dim=-1)) # 해당 텐서의 마지막 dimension의 방향으로 mean을 계산

    # Sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum())
print(t.sum(dim=0)) # 행 방향으로 sum을 계산
print(t.sum(dim=1)) # 열 방향으로 sum을 계산
print(t.sum(dim=-1))



    # Max and Argmax
# max: 가능 큰값 반환
# argmax: 가장 큰값의 인덱스값 반환
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max(dim=0)) # Returns two values: max and argmax
                    # dim=0 이므로, 전체 행 중에 가장 큰 값
print('Max: ', t.max(dim=0)[0]) # 행 방향으로 중에서 가장 큰 값
print('Argmax: ', t.max(dim=0)[1]) # 해당 max값이 어느 디멘션에 있는지 인덱스 반환



    # View
 - Numpy의 reshape 메소드와 동일
   Tensor의 모양을 자유적으로 바꿀 수 있도록 view함수를 익히도록 하자
t = np.array([[[0, 1, 2],
             [3, 4, 5]],
             [[6, 7,8 ],
             [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape) # output: torch.Size([4, 3])
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape) # output: torch.Size([4, 1, 3])



    # Type Casting
# 텐서의 타입을 바꾸어 준다.
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
# 조건문을 사용하여 넣어주면 True -> 1, False -> 0 으로 바꾸어줌
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())



    # In-place Operation
 - 언더바(_)가 붙은 연산으로
   판다스 메소드에서 inplace = True와 같은 역할을 함
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x) # x에 할당을 안했으므로 x 값 출력시 2를 곱하지 않은 상태
print(x.mul_(2.))
print(x) # inplace operation으로 인해서 x 값 출력시 2를 곱한 상태를 출력




    # torch.utils.data.DataLoader
DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
)
# DataLoader - 설명 1
 - DataLoader 객체는 학습에 쓰일 데이터 전체를 보관했다가, train 함수가 batch 하나를 요구하면
   batch size 개수만큼 데이터를 꺼내서 준다고 보면 된다.
   실제로 [batch size, num]처럼 미리 잘라놓는 것은 아니고, 내부적으로 Iterator에 포함된 Index가 존재한다.

# DataLoader - 설명 2
 - train() 함수가 데이터를 요구하면 사전에 저장된 batch size만큼 return하는 형태이다.
   사용할 torch.utils.data.Dataset에 따라 반환하는 데이터(자연어, 이미지, 정답 label 등)는 조금씩 다르지만,
   일반적으로 실제 DataLoader를 쓸 때는 다음과 같이 쓰기만 하면 된다.
for idx, (data, label) in enumerate(data_loader):
    ...

# DataLoader - Ex 1
 - DataLoader 안에 데이터가 어떻게 들어있는지 확인하기 위해, MNIST 데이터를 가져와 보자.
   DataLoader는 torchvision.datasets 및 torchvision.transforms와 함께 자주 쓰이는데,
   각각 Pytorch가 공식적으로 지원하는 dataset, 데이터 transformation 및 augmentation 함수들(주로 이미지 데이터에 사용)를 포함한다.
   각각의 사용법은 아래 절을 참조한다.

input_size = 28
batch_size = 64
transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                transforms.ToTensor()])
data_loader = DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
print('type:', type(data_loader), '\n')
first_batch = data_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(data_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))
# 총 데이터의 개수는 938 * 28 ~= 60000(마지막 batch는 32)이다.




    # Custom Dataset 만들기
 - nn.Module을 상속하는 Custom Model처럼, Custom DataSet은 torch.utils.data.Dataset를 상속해야 한다.
   또한 override해야 하는 것은 다음 두 가지다. python dunder를 모른다면 먼저 구글링해보도록 한다.
 1. __len__(self): dataset의 전체 개수를 알려준다.
 2. __getitem__(self, idx): parameter로 idx를 넘겨주면 idx번째의 데이터를 반환한다.

 - 위의 두 가지만 기억하면 된다. 전체 데이터 개수와, i번째 데이터를 반환하는 함수만 구현하면 Custom DataSet이 완성된다.
   다음에는 완성된 DataSet을 torch.utils.data.DataLoader에 인자로 전달해주면 끝이다.

 - 완전 필수는 아니지만 __init__()도 구현하는 것이 좋다.
   1차함수 선형회귀(Linear Regression)의 예를 들면 다음과 같다.
   데이터는 여기(https://drive.google.com/file/d/1gVxV5eD5NfyEO4aHSyAGmsDgUco8FQPb/view)에서 받을 수 있다.

# Custom Dataset - Ex1
class LinearRegressionDataset(Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
        self.y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y

dataset = LinearRegressionDataset('02_Linear_Regression_Model_Data.csv')



    # Pytorch Model 구조 설명1
# PyTorch의 모든 모델은 기본적으로 다음 구조를 갖는다.
# PyTorch 내장 모델뿐 아니라 사용자 정의 모델도 반드시 이 정의를 따라야 한다.
class Model_Name(nn.Module):
    def __init__(self):
        super(Model_Name, self).__init__()
        self.module1 = ...
        self.module2 = ...
        """
        ex)
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        """

    def forward(self, x):
        x = some_function1(x)
        x = some_function2(x)
        """
        ex)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        """
        return x

# PyTorch 모델로 쓰기 위해서는 다음 조건을 따라야 한다.
# 내장된 모델들(nn.Linear 등)은 당연히 이 조건들을 만족한다.
  1. torch.nn.Module을 상속해야 한다.
  2. __init__()과 forward()를 override해야 한다.
     사용자 정의 모델의 경우 init과 forward의 인자는 자유롭게 바꿀 수 있다. 이름이 x일 필요도 없으며, 인자의 개수 또한 달라질 수 있다.
     이 두 가지 조건은 PyTorch의 기능들을 이용하기 위해 필수적이다.

# 따르지 않는다고 해서 에러를 내뱉진 않지만, 다음 규칙들은 따르는 것이 좋다:
  1. __init__()에서는 모델에서 사용될 module을 정의한다. module만 정의할 수도, activation function 등을 전부 정의할 수도 있다.
     아래에서 설명하겠지만 module은 nn.Linear, nn.Conv2d 등을 포함한다.
     activation function은 nn.functional.relu, nn.functional.sigmoid 등을 포함한다.
  2. forward()에서는 모델에서 행해져야 하는 계산을 정의한다(대개 train할 때). 모델에서 forward 계산과 backward gradient 계산이 있는데,
     그 중 forward 부분을 정의한다. input을 네트워크에 통과시켜 어떤 output이 나오는지를 정의한다고 보면 된다.
    º __init__()에서 정의한 module들을 그대로 갖다 쓴다.
    º 위의 예시에서는 __init__()에서 정의한 self.conv1과 self.conv2를 가져다 썼고, activation은 미리 정의한 것을 쓰지 않고 즉석에서 불러와 사용했다.
    º backward 계산은 PyTorch가 알아서 해 준다. backward() 함수를 호출하기만 한다면.


    # Pytorch Model 구조 설명2
# nn.Module
 - nn.Module은 모든 PyTorch 모델의 base class이다.
   모든 Neural Network Model(흔히 Net이라고 쓴다)은 nn.Module의 subclass이다.
 - nn.Module을 상속한 어떤 subclass가 Nerual Network Model로 사용되려면 다음 두 메서드를 override해야 한다.
  º __init__(self):
# Initialize. 여러분이 사용하고 싶은, Model에 사용될 구성 요소들을 정의 및 초기화한다.
     self.conv1 = nn.Conv2d(1, 20, 5)
     self.conv2 = nn.Conv2d(20, 20, 5)
     self.linear1 = nn.Linear(1, 20, bias=True)
     # __init__에서 정의된 요소들을 잘 연결하여 모델을 구성한다. Nested Tree Structure가 될 수도 있다.

  º forward(self, x):
     x = F.relu(self.conv1(x))
     return F.relu(self.conv2(x))
# 다른 말로는 위의 두 메서드를 override하기만 하면 손쉽게 Custom net을 구현할 수 있다는 뜻이기도 하다.
# 본문: https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
# 참고: https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/#import



    # nn.Module 내장 함수
 - nn.Module에 내장된 method들은 모델을 추가 구성/설정하거나,
   train/eval(test) 모드 변경,
   cpu/gpu 변경,
   포함된 module 목록을 얻는 등의 활동에 초점이 맞춰져 있다.

모델을 추가로 구성하려면,
 - add_module(name, module): 현재 module에 새로운 module을 추가한다.
 - apply(fn): 현재 module의 모든 submodule에 해당 함수(fn)을 적용한다.
               주로 model parameter를 초기화할 때 자주 쓴다.
모델이 어떻게 생겼는지 보려면,
 - children(), modules(): 자식 또는 모델 전체의 모든 module에 대한 iterator를 반환한다.,
 - named_buffers(), named_children(), named_modules(), named_parameters(): 위 함수와 비슷하지만 이름도 같이 반환한다.
모델을 통째로 저장 혹은 불러오려면,
 - state_dict(destination=None, prefix='', keep_vars=False)
   모델의 모든 상태(parameter, running averages 등 buffer)를 딕셔너리 형태로 반환한다. ,
 - load_state_dict(state_dict, strict=True)
   parameter와 buffer 등 모델의 상태를 현 모델로 복사한다. strict=True이면 모든 module의 이름이 정확히 같아야 한다.
학습 시에 필요한 함수들을 살펴보면,
 - cuda(device=None): 모든 model parameter를 GPU 버퍼에 옮기는 것으로 GPU를 쓰고 싶다면 이를 활성화해주어야 한다.
   GPU를 쓰려면 두 가지에 대해서만 .cuda()를 call하면 된다. 그 두 개는 모든 input batch 또는 tensor, 그리고 모델이다.
 - .cuda()는 optimizer를 설정하기 전에 실행되어야 한다.
   잊어버리지 않으려면 모델을 생성하자마자 쓰는 것이 좋다.
 - eval(), train(): 모델을 train mode 또는 eval(test) mode로 변경한다.
   Dropout이나 BatchNormalization을 쓰는 모델은 학습시킬 때와 평가할 때
   구조/역할이 다르기 때문에 반드시 이를 명시하도록 한다.
 - parameters(recurse=True): module parameter에 대한 iterator를 반환한다.
   보통 optimizer에 넘겨줄 때 말고는 쓰지 않는다.
 - zero_grad(): 모든 model parameter의 gradient를 0으로 설정한다.

    # nn.Module 내장 함수 - Ex
def user_defined_initialize_function(m):
    pass
model = torchvision.models.vgg16(pretrained=True)
last_module = nn.Linear(1000, 32, bias=True)
model.add_module('last_module', last_module)
last_module.apply(user_defined_initialize_function)
model.cuda()
# set optimizer. model.parameter를 넘겨준다.
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
# train
model.train()
for idx, (data, label) in dataloader['train']:
    ...
# test
model.eval()
for idx, (data, label) in dataloader['test']:
    ...



    # 모델 구성 방법
 - 크게 6가지 정도의 방법이 있다.
   nn 라이브러리를 잘 써서 직접 만들거나,
   함수 또는 클래스로 정의,
   cfg파일 정의,
   또는 torchvision.models에 미리 정의된 모델을 쓰는 방법이 있다.

# 단순한 방법
 - 매우 단순한 모델을 만들 때는 굳이 nn.Module을 상속하는 클래스를 만들 필요 없이 바로 사용 가능하며, 단순하다는 장점이 있다.
model = nn.Linear(in_features=1, out_features=1, bias=True)

# nn.Sequential을 사용하는 방법
sequential_model = nn.Sequential(
    nn.Linear(in_features=1, out_features=20, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=1, bias=True),
)

# nn layers
 - 여러 Layer와 Activation function들을 조합하여 하나의 sequential model을 만들 수 있다.
   역시 상대적으로 복잡하지 않은 모델 중 모델의 구조가 sequential한 모델에만 사용할 수 있다.
linear1 = nn.Linear(2, 2, bias=True)
linear2 = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

# 함수로 정의하는 방법
 - 바로 위의 모델과 완전히 동일한 모델이다. 함수로 선언할 경우 변수에 저장해 놓은 layer들을 재사용하거나, skip-connection을 구현할 수도 있다.
   하지만 그 정도로 복잡한 모델은 아래 방법을 쓰는 것이 낫다.
def TwoLayerNet(in_features=1, hidden_features=20, out_features=1):
    hidden = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
    activation = nn.ReLU()
    output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
    net = nn.Sequential(hidden, activation, output)
    return net
model = TwoLayerNet(1, 20, 1)

# nn.Module을 상속한 클래스를 정의하는 방법
 - 가장 정석이 되는 방법이다. 또한, 복잡한 모델을 구현하는 데 적합하다.
   __init__ 함수에 있는 super 클래스는 부모 클래스인 nn.Module을 초기화하는 역할을 한다.
class TwoLinearLayerNet(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super(TwoLinearLayerNet, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

model = TwoLinearLayerNet(1, 20, 1)

 - 역시 동일한 모델을 구현하였다. 여러분의 코딩 스타일에 따라, ReLU 등의 Activation function을 forward()에서 바로 정의해서 쓰거나,
   __init__()에 정의한 후 forward에서 갖다 쓰는 방법을 선택할 수 있다. 후자의 방법은 아래와 같다.
  물론 변수명은 전적으로 여러분의 선택이지만, activation1, relu1 등의 이름을 보통 쓰는 것 같다.

class TwoLinearLayerNet(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super(TwoLinearLayerNet, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)

    def forward(self, x):
        x = self.activation1(self.linear1(x))
        return self.linear2(x)

model = TwoLinearLayerNet(1, 20, 1)

 - 두 코딩 스타일의 차이점 중 하나는 import하는 것이 다르다(F.relu와 nn.ReLU는 사실 거의 같다).
   Activation function 부분에서 torch.nn.functional은 torch.nn의 Module에 거의 포함되는데,
   forward()에서 정의해서 쓰느냐 마느냐에 따라 다르게 선택하면 되는 정도이다.


# cfg(config)를 정의한 후 모델을 생성하는 방법
 - 처음 보면 알아보기 까다로운 방법이지만, 매우 복잡한 모델의 경우 .cfg 파일을 따로 만들어 모델의 구조를 정의하는 방법이 존재한다.
   많이 쓰이는 방법은 대략 두 가지 정도인 것 같다.
   먼저 PyTorch documentation에서 찾을 수 있는 방법이 있다. 예로는 VGG를 가져왔다. 코드는 여기에서 찾을 수 있다. (https://pytorch.org/docs/0.4.0/_modules/torchvision/models/vgg.html)

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(...)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):...

    def _initialize_weights(self):...

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

 - 여기서는 .cfg 파일이 사용되지는 않았으나, cfg라는 변수가 configuration을 담당하고 있다.
   VGG16 모델을 구성하기 위해 cfg 변수의 해당하는 부분을 읽어 make_layer 함수를 통해 모델을 구성한다.
 - 더 복잡한 모델은 아예 따로 .cfg 파일을 빼놓는다. YOLO의 경우 수백 라인이 넘기도 한다.

 - .cfg 파일은 대략 다음과 같이 생겼다.
[net]
# Testing
bsatch=1
subdivisions=1
# Training
# batch=64
# subdivisions=8
...

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2
...

 - 이를 파싱하는 코드도 있어야 한다.
def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks
 - 이 방법의 경우 대개 depth가 수십~수백에 이르는 아주 거대한 모델을 구성할 때 사용되는 방법이다.
   많은 수의 github 코드들이 이런 방식을 사용하고 있는데, 그러면 그 모델은 굉장히 복잡하게 생겼다는 뜻이 된다.



    # Containers
 - 여러 layer들을 하나로 묶는 데 쓰인다.
   종류는 다음과 같은 것들이 있는데, Module 설계 시 자주 쓰는 것으로 nn.Sequential이 있다.
    º nn.Module
    º nn.Sequential
    º nn.ModuleList
    º nn.ModuleDict
    º nn.ParameterList
    º nn.ParameterDict
# 본문: https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
# 참조: https://pytorch.org/docs/stable/nn.html#containers

    # nn.Sequential
 - 이름에서 알 수 있듯 여러 module들을 연속적으로 연결하는 모델이다.
model = nn.Sequential(
         nn.Conv2d(1,20,5),
         nn.ReLU(),
         nn.Conv2d(20,64,5),
         nn.ReLU()
         )
# model(x)는 nn.ReLU(nn.Conv2d(20,64,5)(nn.ReLU(nn.Conv2d(1,20,5)(x))))와 같음.

model = nn.Sequential(
          nn.Linear(1,6),
          nn.ReLU(),
          nn.Linear(6,10),
          nn.ReLU(),
          nn.Linear(10,6),
          nn.ReLU(),
          nn.Linear(6,1),
      )

    # Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
 - 조금 다르지만 비슷한 역할을 할 수 있는 것으로는 nn.ModuleList, nn.ModuleDict가 있다.




    # Set Loss function(criterion)
 - Loss function은 모델이 추측한 결과(prediction 또는 output)과 실제 정답(label 또는 y 등)의 loss를 계산한다.
   이는 loss function을 어떤 것을 쓰느냐에 따라 달라진다. 예를 들어 regression model에서 MSE(Mean Squared Error)를 쓸 경우 평균 제곱오차를 계산한다.
   사용법은 다른 함수들도 아래와 똑같다.
 - 여러 코드들을 살펴보면, loss function을 정의할 때는 보통 criterion, loss_fn, loss_function등의 이름을 사용하니 참고하자.

    # loss function Ex - 1
criterion  = nn.MSELoss()
prediction = torch.Tensor([12, 21, 30, 41, 52]) # 예측값
target     = torch.Tensor([10, 20, 30, 40, 50]) # 정답
loss       = criterion(prediction, target)
print(loss)
# tensor(2.)
# loss = (2^2 + 1^2 + 0^2 + 1^2 + 2^2) / 5 = 2

    # loss function Ex - 2
criterion_reduction_none = nn.MSELoss(reduction='none')
loss = criterion_reduction_none(prediction, target)
print(loss)
# tensor([4., 1., 0., 1., 4.])


    # PyTorch Loss function의 종류
    # L1, L2
º nn.L1Loss: 각 원소별 차이의 절댓값을 계산한다.L1
º nn.MSELoss: Mean Squared Error(평균제곱오차) 또는 squared L2 norm을 계산한다.MSE
    # CrossEntropyLoss
º nn.CrossEntropyLoss: Cross Entropy Loss를 계산한다.
# CrossEntropyLoss 설명 - Ex 1
 - nn.LogSoftmax() and nn.NLLLoss()를 포함한다. weight argument를 지정할 수 있다.
# CrossEntropyLoss 설명 - Ex 2
 - http://www.gisdeveloper.co.kr/?p=8668
   PyTorch에서는 다양한 손실함수를 제공하는데, 그 중 torch.nn.CrossEntropyLoss는 다중 분류에 사용됩니다.
   torch.nn.CrossEntropyLoss는 nn.LogSoftmax와 nn.NLLLoss의 연산의 조합입니다.
   nn.LogSoftmax는 신경망 말단의 결과 값들을 확률개념으로 해석하기 위한 Softmax 함수의 결과에 log 값을 취한 연산이고,
   nn.NLLLoss는 nn.LogSoftmax의 log 결과값에 대한 교차 엔트로피 손실 연산(Cross Entropy Loss|Error)입니다.
# CrossEntropyLoss 설명 - Ex 3
 - https://wingnim.tistory.com/34
   Cross Entropy Loss는 다음과 같습니다.
   정답 Y와 log (우리가 예측한 Y)값을 곱해서 뺀 것입니다.
   이전 binary classification에서 본 것과 조금 다른데요, multi label classification에서는 CELoss를 이렇게 정답 label에 대해서만 (정답 label만 1이므로...) 적용시킵니다.
   이렇게 되면 모든 label을 1로 예측하면 loss가 0이 아니냐! 라고 생각할 수도 있는데요, softmax를 거치기 떄문에 모든 label의 총 합이 1이라서 모든 label을 1로 예측할 수는 없습니다 !

º nn.CTCLoss: Connectionist Temporal Classification loss를 계산한다.
º nn.NLLLoss: Negative log likelihood loss를 계산한다.NLL
º nn.PoissonNLLLoss: target이 poission 분포를 가진 경우 Negative log likelihood loss를 계산한다.PNLL
º nn.KLDivLoss: Kullback-Leibler divergence Loss를 계산한다.KLDiv
º nn.BCELoss: Binary Cross Entropy를 계산한다.BCE
  - 이진분류에서 사용, MSE나 MAE보다 패널티가 높아서 이진분류할 때 손실함수로써 더 자주 사용하는 편
º nn.BCEWithLogitsLoss: Sigmoid 레이어와 BCELoss를 하나로 합친 것인데, 홈페이지의 설명에 따르면 두 개를 따로 쓰는 것보다 이 함수를 쓰는 것이 조금 더 수치 안정성을 가진다고 한다.
   BCE이외에 MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss를 계산하는 함수들이 있다. 필요하면 찾아보자.
# 본문: https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
# 참조: https://pytorch.org/docs/stable/nn.html#loss-functions


    # softmax
 - 신경망의 결괏값을 확률로 바꿔줘야 하는데 이때 사용하는 방법이 소프트맥스 함수입니다.
   softmax(yi) = exp(yi) / ∑jexp(yj)
   결괏값 벡터에 있는 값들이 지수 함수를 통과하면 모든 값은 양수가 됩니다.
   결괏값은 지수 함수를 거쳐 변환되고 전체 합 중 각각의 비중이 소프트맥스 함수의 결과가 됩니다.

 - 교차 엔트로피는 목표로 하는 최적의 확률분포 p와 이를 근사하려는 확률분포 q가 얼마나 다른지를 측정하는 방법입니다.
   즉 교차 엔트로피는 원래 p였던 분포를 q로 표현했을 때 얼마만큼의 비용이 드는지를 측정한다고 할 수 있습니다.

 - 교차 엔트로피 값은 예측이 잘못될수록 L1 손실보다 더 크게 증가하는 것을 확인할 수 있습니다.
   그만큼 더 페널티가 크고 손실 값이 크기 때문에 학습 면에서도 교차 엔트로피 손실을 사용하는 것이 장점이 있다는 뜻입니다.
   따라서 분류 문제에서는 교차 엔트로피 손실을 많이 사용합니다.



    # model.train() 과 model.eval()
 - 모델을 학습 상태와 테스트 상태로 조정하는 기능은 torch.nn.Module에 Module.train() 및 Module.eval()이라는 이름으로 구현되어 있음.
   학습할 때는 기본 모드인 train()으로 해놓고 학습하고,
   테스트할 때는 eval()을 사용함.
 - 테스트할 때 eval(): 드롭아웃 모드를 바꿔줄 수 있음,
	배치정규화시 학습때 배치 단위의 평균과 분산들을 차례대로 받아 저장해놓았다가
    테스트할 때는 해당 배치의 평균과 분산을 구하지 않고 구해놓았던 평균과 분산으로 정규화를 함


    # Train Model
 - Pytorch의 학습 방법은 다음과 같다.
  1. model structure, loss function, optimizer 등을 정한다.
  2. optimizer.zero_grad(): 이전 epoch에서 계산되어 있는 parameter의 gradient를 0으로 초기화한다.
  3. output = model(input): input을 모델에 통과시켜 output을 계산한다.
  4. loss = loss_fn(output, target): output과 target 간 loss를 계산한다.
  5. loss.backward(): loss와 chain rule을 활용하여 모델의 각 레이어에서 gradient(Δw)를 계산한다.
  6. optimizer.step(): w←w−αΔw 식에 의해 모델의 parameter를 update한다.

    # 간단한 학습 과정은 다음 구조를 따른다.
# 변수명으로 input을 사용하는 것은 비추천. python 내장 함수 이름이다.
for data, target in datalodaer:
    optimizer.zero_grad(): # RNN에서는 생략될 수 있음
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

optimizer.zero_grad():
 - Pytorch는 gradient를 loss.backward()를 통해 계산하지만,
   이 함수는 이전 gradient를 덮어쓴 뒤 새로 계산하는 것이 아니라, 이전 gradient에 누적하여 계산한다.
   귀찮은데? 라고 생각할 수는 있다. 그러나 이러한 누적 계산 방식은 RNN 모델을 구현할 때는 오히려 훨씬 편하게 코드를 작성할 수 있도록 도와준다.
   그러니 gradient가 누적될 필요 없는 모델에서는 model에 input를 통과시키기 전 optimizer.zero_grad()를 한번 호출해 주기만 하면 된다고 생각하면 끝이다.
 - Pytorch가 대체 어떻게 loss.backward() 단 한번에 gradient를 자동 계산하는지에 대한 설명하면,
   모든 Pytorch Tensor는 requires_grad argument를 가진다. 일반적으로 생성하는 Tensor는 기본적으로 해당 argument 값이 False이며,
   따로 True로 설정해 주면 gradient를 계산해 주어야 한다. nn.Linear 등의 module은 생성할 때 기본적으로 requires_grad=True이기 때문에,
   일반적으로 모델의 parameter는 gradient를 계산하게 된다.
 - 마지막 레이어만 원하는 것으로 바꿔서 그 레이어만 학습을 수행하는 형태의 transfer learning을 requires_grad를 이용해 손쉽게 구현할 수 있다.
   이외에도 특정 레이어만 gradient를 계산하지 않게 하는 데에도 쓸 수 있다.
   아래 예시는 512개의 class 대신 100개의 class를 구별하고자 할 때 resnet18을 기반으로 transfer learning을 수행하는 방식이다.
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)
# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
 º requires_grad=True인 Tensor로부터 연산을 통해 생성된 Tensor도 requires_grad=True이다.



    # train 설계 - Ex 1: Linear Regression
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train * W
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(f'Epoch {epoch}/{nb_epochs} W: {W.item():.4f}, Cost: {cost.item():.4f}')


    # train 설계 - Ex 2: Multivariable Linear regression
# 1. 데이터 정의
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 2. 모델 정의
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 3. optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # 4. Hypothesis 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @
    # 5. Cost 계산 (MSE)
    cost = torch.mean((hypothesis - y_train) ** 2)
    # 6. Gradient descent: cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(f'Epoch {epoch:4d}/{nb_epochs} hypothesis: {hypothesis.squeeze().detach()} Cost: {cost.item()}')


    # train 설계 - Ex 3: nn.Module을 상속하여 모델을 생성
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# 1. 데이터 정의
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 2. 모델 정의
# W = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
model = MultivariateLinearRegressionModel()
# 3. optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # 4. Hypothesis 계산
    #hypothesis = x_train.matmul(W) + b # or .mm or @
    prediction = model(x_train)
    # 5. Cost 계산 (MSE)
    #cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)
    # 6. Gradient descent: cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item()}')


    # train 설계 - Ex 4: Logistic regression
# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')


    # train 설계 - Ex 5: Logistic regression nn.Module를 상속하여 클래스 생성
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = model(x_train)
    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} Accuracy {accuracy * 100:2.2f}')



    # train 설계 - Ex 6, 7: CrossEntropy로 계산
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # Cost 계산
    z = x_train.matmul(W) + b # or .mm or @
    cost = F.cross_entropy(z, y_train)
    # Cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

## nn.Module를 상속하여 클래스 생성
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3 !
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    prediction = model(x_train)
    # Cost 계산
    cost = F.cross_entropy(prediction, y_train)
    # Cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')



    # train 설계 - Ex 8: train, test 함수화
# define train
def train(model, optimizer, x_train, y_train):
    nb_epochs = 30
    for epoch in range(nb_epochs):
        # H(x) 계산
        prediction = model(x_train)
        # cost 계산
        cost = F.cross_entropy(prediction, y_train)
        # cost = F.mse_loss(prediction, y_train)
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost {cost.item():.6f}')

train(model, optimizer, x_train, y_train)

# define test
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)
    print(f'Accuracy: {correct_count/len(y_test) * 100}% Cost: {cost.item():.6f}')

test(model, optimizer, x_test, y_test)





    # with torch.no_grad()
 - with torch.no_grad(): 범위 안에서는 gradient 계산을 하지 않는다.
   with torch.no_grad() 안에서 선언된 with torch.enable_grad():
   범위 안에서는 다시 gradient 계산을 한다.
   이 두 가지 기능을 통해 국지적으로 gradient 계산을 수행하거나 수행하지 않을 수 있다.



    # 딥러닝 모델을 설계할 때 활용하는 장비 확인
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using Pytorch version: {torch.__version__}, Device: {DEVICE}')



    # 하이퍼파라미터 설명
BATCH_SIZE: 모델에서 파라미터를 업데이트할 때 계산되는 데이터의 개수,
            수만큼 출력된 결괏값에 대한 오찻값을 계산,
            계산된 오찻값을 평균하여 Back Propagation을 적용, 이를 바탕으로 파라미터 업데이트
INPUT_SIZE: Input의 크기이자
            입력층의 노드 수를 의미
HIDDEN_SIZE: Input을 다수의 파라미터를 이용해 계산한 결과에 한번 더 계산되는 파라미터 수,
             은닉층의 노드 수를 의미
OUTPUT_SIZE: 최종으로 출력되는 값의 벡터의 크기를 의미,
             보통 Output의 크기는 최종으로 비교하고자 하는 레이블의 크기와 동일하게 설정

    # 파라미터 설명 - Ex 1
 - if you have 1000 trainning examples, and your batch size is 500,
   then it will take 2 iterations to complete 1 epoch.



    # 모델 작은 데이터로 동작하는지 테스트
batch_train_images, batch_train_labels = next(iter(train_loader))
batch_test_images, batch_test_labels = next(iter(test_loader))

temp_images = batch_train_images.to(device)
predictions = model(temp_images)
print(temp_images.shape)
print(predictions.shape)



    # 학습 후 모델 변수 값 확인
# 현재 모델은 weight와 bias을 변수로 가지고 있는데 그 값들이 학습 후 실제 몇인지 수치적으로 확인해봅니다.
param_list = list(model.parameters())
print("Weight:",param_list[0].item(),"\nBias:  ",param_list[1].item())

    # 학습된 모델의 결과값과 실제 목표값의 비교
plt.figure(figsize=(10,10))
plt.scatter(x.detach().numpy(),y_noise,label="Original Data")
plt.scatter(x.detach().numpy(),output.detach().numpy(),label="Model Output")
plt.legend()
plt.show()
# 중간에 꺾인 부분은 렐루 함수의 영향입니다.
# 은닉층은 해당 층의 입력값에 가중치를 곱해줌으로써 선형변환이 일어나도록 하고
# 렐루 활성화 함수는 이 중 0보다 작은 값들을 모두 0으로 만들기 때문에
# 여러 은닉층을 통과하면서 여러 지점에서 꺾인 모양이 나타나게 됩니다.



    # 헷갈리는 max, argmax 코드
_, predicted = torch.max(val_output.data, 1) == val_output.argmax(dim=1)
predicted



    # torchvision.models의 모델을 사용하는 방법
# torchvision.models에서는 미리 정의되어 있는 모델들을 사용할 수 있다.
# torchvision.models 참조: https://pytorch.org/docs/stable/torchvision/models.html
 - 이 모델들은 그 구조뿐 아니라 pretrained=True 인자를 넘김으로써 pretrained weights를 가져올 수도 있다.
    º AlexNet
    º VGG-11, VGG-13, VGG-16, VGG-19
    º VGG-11, VGG-13, VGG-16, VGG-19 (with batch normalization)
    º ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    º SqueezeNet 1.0, SqueezeNet 1.1
    º Densenet-121, Densenet-169, Densenet-201, Densenet-161
    º Inception v3

 - 모델에 따라 train mode와 eval mode가 정해진 경우가 있으므로 이는 주의해서 사용하도록 한다.
   모든 pretrained model을 쓸 때 이미지 데이터는 [3, W, H] 형식이어야 하고, W, H는 224 이상이어야 한다.
   또 아래 코드처럼 정규화된 이미지 데이터로 학습된 것이기 때문에, 이 모델들을 사용할 때에는 데이터셋을 이와 같이 정규화시켜주어야 한다.
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   사용법은 대략 다음과 같다. 사실 이게 거의 끝이고, 나머지는 다른 일반 모델처럼 사용하면 된다.

# torchvision.models - Ex 1
import torchvision.models as models
# model load
alexnet = models.alexnet()
vgg16 = models.vgg16()
vgg16_bn = models.vgg16_bn()
resnet18 = models.resnet18()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()

    # pretrained model - Ex 1
resnet18 = models.resnet18(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
...

    # pretrained model - Ex 2
model = models.resnet34(pretrained = False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.cuda()



    # transfer learning 전이학습
- 전이학습은 특정 조건에서 얻어진 어떤 지식을 다른 상황에 맞게 말 그래도 '전이'해서 활용하는 학습 방법입니다.
- 이때 전이되는 것은 범용적인 형태를 구분할 수 있는 지식, 즉 학습한 필터가 될 것입니다.
장점
 1. 데이터 부족을 어느 정도 해결
 2. 학습에 걸리는 시간이 줄어듬
 3. 시뮬레이션에서 학습된 모델을 현실에 적용할 수 있게 해줌

- pretrained = False로 하면 모델 변수는 학습된 변수 대신 무작위 값으로 초기화됨.

    # transfer learning - Ex 1 Resnet
 - 이미지넷으로 이미 학습된 모델의 앞부분을 사용합니다 (Pretrained ResNet-50)
 - 또한 해당 모델을 다른 데이터셋에 적용합니다.
 - 다른 데이터셋에 적용하기 위해 모델의 뒷단을 새롭게 만듭니다. (Add fully connected layer )
import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
for name,module in resnet.named_children():
    print(name)

 - 커스텀 레즈넷을 새로 정의하되 layer0는 이미 학습된 모델의 파라미터를 가져오고
 - layer1는 새롭게 만들어서 이 부분을 학습합니다.
class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet50.children())[0:-1])
        self.layer1 = nn.Sequential(
            nn.Linear(2048,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,num_category),
            nn.ReLU()
        )

    def forward(self,x):
        out = self.layer0(x)
        out = out.view(batch_size,-1)
        out= self.layer1(out)
        return out

# Module on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Resnet().to(device)
# 모델의 layer0의 파라미터들은 학습이 되지 않도록 기울기 계산을 꺼둡니다.
for params in model.layer0.parameters():
    params.require_grad = False
# layer1의 파라미터들은 학습되도록 기울기 계산을 켜둡니다.
for params in model.layer1.parameters():
    params.requires_grad = True
# 모델을 한번 확인합니다
for m in model.children():
    print(m)

# summary
import torchsummary
from torchsummary import summary
summary(resnet50, input_size=(3, 224, 224))




    # Save & Load model
 - 모델을 저장하는 방법은 여러 가지가 있지만, pytorch를 사용할 때는 다음 방법이 가장 권장된다.
   아주 유연하고 또 간단하기 때문이다.
 - 모델을 통째로 저장 혹은 불러오려면,
    - state_dict(destination=None, prefix='', keep_vars=False)
      모델의 모든 상태(parameter, running averages 등 buffer)를 딕셔너리 형태로 반환한다. ,
    - load_state_dict(state_dict, strict=True)
      parameter와 buffer 등 모델의 상태를 현 모델로 복사한다. strict=True이면 모든 module의 이름이 정확히 같아야 한다.
 - torch.save & torch.load
   내부적으로 pickle을 사용하며, 따라서 모델뿐 아니라 일반 tensor, 기타 다른 모든 python 객체를 저장할 수 있다.

    # Save Ex - 1:
torch.save(model.state_dict(), PATH)
 ex: torch.save(net.state_dict(), '../model/model.pth')
    # Load Ex - 1:
model = TheModelClass(*args, **kwargs)
 ex: new_net = CNN().to(device)
model.load_state_dict(torch.load(PATH))
 ex: new_net.load_state_dict(torch.load('../model/model.pth'))


    # Save Ex - 2:
 - epoch별로 checkpoint를 쓰면서 저장할 때는 다음과 같이 혹은 비슷하게 쓰면 좋다.
   checkpoint를 쓸 때는 단순히 모델의 parameter뿐만 아니라 epoch, loss, optimizer 등을 저장할 필요가 있다.
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
    # Load Ex - 2
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# model.train() or model.eval()


    # Save - Ex 3
 - 모델을 저장하려면 torch.save 함수를 이용한다. 저장할 모델은 대개 .pt 확장자를 사용한다
 torch.save(obj=model, f='02_Linear_Regression_Model.pt')
   참고: .pt 파일로 저장한 PyTorch 모델을 load해서 사용하려면 다음과 같이 한다. 이는 나중에 Transfer Learning과 함께 자세히 다루도록 하겠다.
    # Load - Ex 3
loaded_model = torch.load(f='02_Linear_Regression_Model.pt')
display_results(loaded_model, x, y)
# 전체 코드: https://github.com/greeksharifa/Tutorial.code/blob/master/Python/PyTorch_Usage/02_Linear_Regression_Model/main.py








---------- DNN ----------


    # DNN Ex - 1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    # DNN Ex - 2
class NNModel(torch.nn.Module):

    def __init__(self):
        super(NNModel,self).__init__()
        self.l1 = nn.Linear(784,520)
        self.l2 = nn.Linear(520,320)
        self.l3 = nn.Linear(320,240)
        self.l4 = nn.Linear(240,120)
        self.l5 = nn.Linear(120,10)

    def forward(self, x):
        # input data : ( n , 1 , 28 , 28 )
        x = x.view(-1,784) # Flatten : ( n , 784 )
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)





    # DNN train, valid - Ex 1
# 마지막 loss 프린트 부분은 사용시 수정해서 사용
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1, last_epoch=-1)

for epoch in range(num_epoch):
    lr_scheduler.step()

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        show_period = 100
        if i % show_period == show_period-1:    # print every "show_period" mini-batches
            print(f'[{epoch + 1}, {(i + 1)*batch_size:5d}/50000] loss: {running_loss / show_period:.7f}')
            running_loss = 0.0

    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'[{epoch + 1} epoch] Accuracy of the network on the validation images: {100 * correct / total}')
print('Finished Training')


    # DNN train - Ex 2
def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        if batch_idx%50==0:
            print(f'Train Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(train_loader.dataset)} \
            ({100. * batch_idx / len(train_loader):.0f}%)]\t
            Loss: {loss.data[0]:.6f}')




    # DNN test - Ex 1
def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        test_loss += criterion(output,target).data[0]

        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))








---------- CNN ----------


     # Cnn Basic
# Convolution?
 - 이미지 위에서 stride 값 만큼 filter(kernel)을 이동시키면서
   겹쳐지는 부분의 각 원소의 값을 곱해서 모두 더한 값으로 출력으로 하는 연산
# Stride and Padding
 - stride: filter를 한번에 얼마나 이동 할 것인가
 - padding: zero-padding
# 입력의 형태
- input type: torch.Tensor
- input shape: (N x C x H x W)
               (batch_size, channel, height, width)

# Convolution의 output 크기
 - output size = (input size - filter size + (2 * padding)) / Stride + 1

# Conv2d 구조
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',)
- https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
- in_channels: 입력의 채널 수
- out_channels: 출력의 채널 수
- kernel_size: 필터 혹은 커널의 크기
- stride: 필터 적용의 간격 (stride: 걸음걸이)
- padding: 입력 데이터를 추가적으로 둘러싸는 층의 두께
- dilation: 책에서 다루지 않고 넘어간 내용이라 링크로 대체합니다. (https://laonple.blog.me/220991967450)
- groups: 입력을 채널 단위로 몇개의 분리된 그룹으로 볼 것인가
- bias: 편차의 사용여부
- padding_mode: 패딩 적용 방식 (ex. zero padding은 0으로 채우는 경우)

# Pooling 설명, 구조
 - 풀링은 다운샘플링(downsamping) 또는 서브샘플링(Subsamping) 의 일종으로, 합성곱 신경망에서는 크게 두 종류가 사용됨.
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,)
 - 일정 크기의 구간 내에서 가장 큰 값만을 전달하고 다른 정보는 버리는 방법
 - MaxPool2d에는 stride가 default로 None값이 들어가서 따로 설정을 안해줄 경우 Maxpool2d를 거치면 크기가 반으로 줄게 된다
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
 - 일정 크기의 구간 내의 값들의 평균을 전달하는 방법




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
                         transforms.RandomVerticalFlip(p=0.5)]) # transforms 리스트에 포함된 변환 함수 중 랜덤으로 1개 적용
    transforms.RandomApply(transforms, p=0.5) # transforms 리스트에 포함된 변환 함수들을 p의 확률로 적용한다.
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


# 위의 모든 변환 함수들을 하나로 조합하는 함수는 다음과 같다.
# 이 함수를 dataloader에 넘기면 이미지 변환 작업이 간단하게 완료된다.
    transforms.Compose(transforms)
        transforms.Compose([
            transforms.CenterCrop(14),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
# 변환 순서는 보통 resize/crop, toTensor, Normalize 순서를 거친다. Normalize는 tensor에만 사용 가능하므로 이 부분은 순서를 지켜야 한다.


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





    # 데이터 확인하기 (1). 구조
for (X_train, y_train) in train_loader:
    print(f'X_train: {X_train.size()}, type: {X_train.type()}')
    print(f'y_train: {y_train.size()}, type: {y_train.type()}')
    break

for i in range(3):
    img= mnist_train[i][0].numpy()
    plt.imshow(img[0],cmap='gray')
    plt.show()

print('train:', train_loader.dataset.data.shape)
print('test:', test_loader.dataset.data.shape)

    # 데이터 확인하기 (2). 그림
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1) # 1 by 10의 행렬에다가 그림을 순서대로 그릴것, i+1은 그 중 어느 열인지를 나타냄.
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap='gray_r')
    # plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    plt.title(f'Class: {str(y_train[i].item())}')


    # 데이터 확인하기 (2) - Ex 1. VGG 만들때 사용하였던 것
np.random.seed(42)
random_train_pictures = [np.random.randint(1, 50000) for i in range(10)]

plt.figure(figsize=(20, 2))
for i in range(len(random_train_pictures)):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_loader.dataset.data[random_train_pictures[i]])
    plt.axis('off')
    plt.title(f'Class: {str(train_loader.dataset.targets[random_train_pictures[i]])}')






    # CNN Ex - 1
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


    # CNN Ex - 2
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, 8 * 8 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    # CNN Ex - 3
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = out.view(out.shape[0], -1)
        print(out.shape)
        out = self.layer3(out)
        return out


    # CNN Ex - 4
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2), # 224 -> 111
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) # 111 -> 55
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2), # 55 -> 27
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 27 -> 13
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2), # 13 -> 6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 6 -> 3
        )
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x) # (32, 16, 55, 55)
        out = self.layer_2(out) # (32, 13, 13)
        out = self.layer_3(out) # -> (32, 64, 3, 3)
        out = out.view(batch_size, -1) # -> (32, 576)
                                        # -> (32, 64)
        out = self.relu(self.fc1(out)) # (576, 10)
        out = self.fc2(out) # (10, 2)
        return out


    # CNN Ex - 5
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(3 * 3 * 128, 625)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 1- self.keep_prob)
        self.fc2 = nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


    # CNN Ex - 6
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout2d(0.2), # 오버피팅하지 않는 상태에서 정형화나 드롭아웃을 넣으면 오히려 학습이 잘 안됨.
            nn.Conv2d(16,32,3,padding=1), # 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),            # 14
            nn.Conv2d(32,64,3,padding=1), # 14
            nn.BatchNorm2d(64),
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

        # 초기화 하는 방법
        # 모델의 모듈을 차례대로 불러옵니다.
        for m in self.modules():
            # 만약 그 모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):

                # 작은 숫자로 초기화하는 방법
            # 가중치를 평균 0, 편차 0.02로 초기화합니다.
            # 편차를 0으로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
            # 모듈의 가중치를 xavier normal로 초기화합니다.
            # 편차를 0으로 초기화합니다.
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

                # Kaming Initialization
            # 모듈의 가중치를 kaming he normal로 초기화합니다.
            # 편차를 0으로 초기화합니다.
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)


            # 만약 그 모듈이 nn.Linear인 경우
            elif isinstance(m, nn.Linear):

                # 작은 숫자로 초기화하는 방법
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)
        return out


    # CNN Ex - 7
class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_block1, self.shape = self.conv_block(224, 3, 8, 3, padding=1)
    self.conv_block2, self.shape = self.conv_block(self.shape, 8, 16, 3, stride=2)
    self.conv_block3, self.shape = self.conv_block(self.shape, 16, 16, 3, padding=1)
    self.conv_block4, self.shape = self.conv_block(self.shape, 16, 32, 3, stride=2)
    self.conv_block5, self.shape = self.conv_block(self.shape, 32, 32, 3, padding=1)
    self.conv_block6, self.shape = self.conv_block(self.shape, 32, 64, 3, stride=2)
    self.conv_block7, self.shape = self.conv_block(self.shape, 64, 64, 3, padding=1)
    self.conv_block8, self.shape = self.conv_block(self.shape, 64, 128, 3, stride=2)

    self.fc_block1 = self.fc_block(128 * self.shape**2, 256)
    self.fc_block2 = self.fc_block(256, 128)
    self.fc_block3 = self.fc_block(128, 32)

    self.output = torch.nn.Linear(32, 1)

  def conv_block(self, shape, in_, out_, kernel, stride= 1, padding=0):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_, out_, kernel, stride=stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_),
        torch.nn.ReLU()
    )

    shape = int(np.floor((shape - kernel + 2*padding) / stride) + 1 )

    return block, shape

  def fc_block(self, in_, out_):
    block = torch.nn.Sequential(
        torch.nn.Linear(in_, out_, bias=False),
        torch.nn.BatchNorm1d(out_),
        torch.nn.ReLU()
    )
    return block


  def forward(self, x):
    # (1, 224, 224)
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.conv_block6(x)
    x = self.conv_block7(x)
    x = self.conv_block8(x)

    x = torch.flatten(x, 1)
    # x = x.view(-1, 64 * self.shape**2)

    x = self.fc_block1(x)
    x = self.fc_block2(x)
    x = self.fc_block3(x)

    x = self.output(x)
    x = torch.sigmoid(x)

    return x


    # VGG Ex - 1
class VGG(nn.Modules):
    def __init__(self, features, num_classes=1000, init_weights=True):
       super(VGG, self).__init__()
       self.features = features # convolution layer
       self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
       self.classifier = nn.Sequential(
           nn.Linear(512 * 7 * 7, 4096),
           nn.ReLU(True),
           nn.Linear(4096, 4096),
           nn.ReLU(True),
           nn.Linear(4096, num_classes),
       ) # FC layer
       if init_weights:
           self.initialize_weights()

   def forward(self, x):
       x = self.features(x)
       x = self.avgpool(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       return x

   def _initialize_weights(self):
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weights, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

   def make_layers(cfg, batch_norm=False):
      layers = []
      in_channels = 3
      for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                con2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [con2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)



    cfgs = {
       'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
       'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M']
    }
    # VGG Ex - 1 사용예시
vgg_custom = VGG(make_layers(cfg['custom']), num_classes=10, init_weights=True)

cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M',
        256, 256, 256, 512, 512, 512, 'M'] # 13 + 3 = vgg16
conv = make_layers(cfg['custom'], batch_norm=True)
vgg16 = VGG(vgg.make_layers(cfg), 10, init_weights=True).to(device)




    # VGG Ex - 2
class VGG(nn.Module):
    def __init__(self, conv_layers, num_classes=10, init_weights=True):
        super(VGG, self).__init__()

        self.conv_layers = conv_layers
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 2048), # 32->(16, 8, 4): 3번 pooling
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
         ) # FC layer

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
vgg16 = VGG(conv, num_classes=10, init_weights=True)



# VGG test Ex - 1
correct = 0
total = 0

with torch.no_grad():
    for image, label in testloader:
        image = image.to(device)
        label = label.to(device)
        outputs = vgg16(image)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == label).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%%')



    # 1x1 합성곱
 - 1x1 합성곱은 이미지의 가로 세로 기준으로 하나의 픽셀만을 입력으로 받지만
   채널 관점에서 봤을 때는 채널의 개수만큼의 입력을 받아서 하나의 결과를 생성하고 이를 목표 채널 수만큼 반복합니다.
 - 1x1 합성곱은 입력 채널과 결과 채널간의 완전연결 네트워크라고 볼 수 있습니다.

 - 1x1 합성곱이 어떻게 메모리를 적게 쓰도록 만들까요?
   채널의 수가 증가하도록 할 수 있지만 반대로 줄어들게 설계할 수도 있습니다
   예를 들어 128개에서 32개로 줄인다면 이를 통해 입력 텐서를 채널 방향으로 압축할 수 있는 것입니다.
   기초적인 인셉션 모듈 (a)가 128개 채널에서 256개 채널로 늘어나는 연산이었다면
   (b)는 128개의 채널을 1x1 합성곱을 통해 32개로 줄이고 다시 합성곱 연산은 256개의 채널로 늘립니다.
   이렇게 되면 기존 연산보다 메모리 사용을 줄일 수 있습니다.
   1x1 합성곱 연산이 완전연결 네트워크이므로 연산이 더 늘어날 것 같지만, 입력갑의 가로 세로 어느 위치에나
   동일하게 적용되는 합성곱 연산의 특성 때문에 기존 연산보다 적은 메모리를 사용합니다.



    # 잔차 학습 블록
 - 잔차 학습 블록은 이전 단계에서 뽑았던 특성들을 변형시키지 않고 그대로 더해서 전달하기 때문에
   입력 단에 가까운 곳에서 뽑은 단순한 특성과 뒷부분에서 뽑은 복잡한 특성 모두를 사용한다는 장점이 있음.
   또한 더하기 연산은 역전파 계산을 할 때 기울기가 1이기 때문에 손실이 줄어들거나 하지 않고
   모델의 앞부분까지 잘 전파되기 때문에, 학습 면에서도 GoogleNet처럼 보조분류기가 필요하지 않다는 장점이 있음.

    # ResNet 설명
 - ResNet은 Residual Network의 약자로, 마이크로소프트에서 제안한 모델입니다.
   지금까지도 이미지 분류의 기본 모델로 널리 쓰이고 있습니다.
   Residual Block이라는 개념을 도입했으며 이전 Layer의 Feature Map을 다음 Layer의 Feature Map에 더해주는 개념입니다.
   이를 'Skip Connection' 이라 합니다. 네트워크가 깊어짐에 따라 앞 단의 Layer에 대한 정보는 뒤의 Layer에서는 희석될 수 밖에 없습니다.
   이러한 단점을 보완하기 위해 이전의 정보를 뒤에서도 함께 활용하는 개념이라 이해할 수 있습니다.

    # ResNet - Ex 1
# ----- BasicBlock 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 =  nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        # ----- shortcut 정의(이전 feature map 그대로 가져오는 층)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ------ ResNet 정의
class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride = 1)
        self.layer2 = self._make_layer(32, 2, stride = 2)
        self.layer3 = self._make_layer(64, 2, stride = 2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



    # ResNet - Ex 2
def conv3x3(in_planes, out_planes, stride=1):
    ''' 3x3 convolution with padding '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stirde, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 3x3, stride = stride
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 . stride = 1
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # relu까지 거치고 원래의 값을 더하는게 아니라, 본래의 값까지 더한 후에 relu를 거침
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def  __init__(self, in_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        slef.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNormwd(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 1x1, stride = 1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3, stride = stride
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1, stride = 1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0]) # '''3'''
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # '''4'''
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # '''6'''
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # '''3'''

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1):

            downsample = None

            if stride != 1 or self.inplanes != planes * block.expansion:

                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride), # conv1x1(256, 512, 2)
                    nn.BatchNorm2d(planes * block.expansion), # batchnorm2d(512)
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))

            self.inplanes = planes * block.expasion # self.inplanes = 128 * 4

            for _ in range(1, block):
                layers.append(block(self, inplanes, planes)) # * 3

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

def resnet18(pretained=False, **kwargs):
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs) # => 2 * (2+2+2+2) + 1(conv1) + 1(fc) = 16 + 2 = resnet18
    return model
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #=> 3*(3+4+6+3) +(conv1) +1(fc) = 48 +2 = 50
    return model
def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs) # 3*(3+8+36+3) +2 = 150+2 = resnet152
    return model



    # GoogleNet - Ex 1
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        slef.branch3x3db1_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3db1_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3db1_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]
        return torch.cat(outputs, 1)

class googleNet(nn.Module):

    def __init__(self):
        super(googleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88,20, kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


    # GoogleNet의 보조 분류기(auxiliary classifier)
 - 보조 분류기는 모델이 깊어지면서 마지막 단의 분류 네트워크에서 발생한 손실이 모델의 입력 부분까지
   전달이 안되는 현상(Gradient Vanishing problem)을 극복하기 위해 사용되었습니다.
   즉 학습을 보조하는 역할입니다.
   물론 학습 이후 테스트 시에는 사용되지 않습니다.




    # CNN train - Ex 1
# 내가 짠 것
loss_arrs = []

def fit(model, train_loader, epochs, optimizer, loss_func):
    for epoch in range(epochs):
        start = time.time()
        avg_loss = 0
        total_batch = len(train_dataset) // batch_size

        for num, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss / total_batch
            loss_arrs.append(avg_loss)
        print(f'[Epoch: {epoch+1}/{epochs}]  loss={avg_loss:.4f},  time={time.time()-start:.2f}')
    print('Learning Finished !')
fit(model, train_loader, epochs, optimizer, loss_func)

# 손실 그래프로 손실이 어떻게 줄어가는지 확인합니다.
plt.plot(loss_arr)
plt.show()



    # CNN train, valid - Ex 2
# 내 스타일에 맞는 코드
valid_loss_arr = []
train_loss_arr = []

for epoch in range(epochs):
    start = time.time()
    train_avg_loss = 0
    train_acc = 0
    model.train()
    for image, label in train_loader:
        # ------- assign train data
        image = image.to(device)
        label = label.to(device)
        # ------- forward prop
        optimizer.zero_grad()
        output = model(image)
        # ------- backward prop
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        # ------- get train performance
        batch_acc = ((output.argmax(dim=1) == label).float().mean())
        train_acc += batch_acc / len(train_loader)
        train_avg_loss += loss / len(train_loader)
    train_loss_arr.append(train_avg_loss)
    print(f'Epoch : {epoch+1}/{epochs}, train_acc : {train_acc:.4f}, train_loss : {train_avg_loss:.4f}', end=' / ')

    model.eval()
    with torch.no_grad():
        valid_acc=0
        valid_avg_loss =0
        for image, label in valid_loader:
            # ------- assign valid data
            image = image.to(device)
            label = label.to(device)
            # ------- forward prop
            val_output = model(image)
            val_loss = loss_func(val_output,label)
            # ------- get valid performance
            val_batch_acc = ((val_output.argmax(dim=1) == label).float().mean()) # acc = 맞춘 개수 / 배치사이즈
            valid_acc += val_batch_acc / len(valid_loader) # acc / total_Iteration
            valid_avg_loss += val_loss / len(valid_loader) # val_loss / total_Iteration
        valid_loss_arr.append(valid_avg_loss)
        print(f'valid_acc : {valid_acc:.4f}, val_loss : {valid_avg_loss:.4f}, takes {time.time() - start}secs')

plt.plot(train_loss_arr, label='train')
plt.plot(valid_loss_arr, label='valid')
plt.legend()
plt.show()



    # CNN train - Ex 3
loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())



    # CNN train - Ex 4
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        # ------------- forward
        optimizer.zero_grad()
        output = model(image)
        # ------------- backward
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {Epoch} \
                # 전체 데이터셋 중에 몇개 인지 센 것,
                [{batch_idx * len(image)} / {len(train_loader.dataset)}] \
                # 전체 배치 갯수 중에 몇 번째 배치인지 센 것
                ({batch_idx / len(train_loader) * 100:.0f}%) \
                # Loss가 몇인지 출력, 줄어드는지 확인
                 Train Loss: {loss.item():.6f}')


    # CNN train - Ex 5
# loss만 보는 깔끔 코드
total_batch = len(train_loader)

Epochs = 5
for epoch in range(Epochs):
    avg_cost = 0
    for num, (image, label) in enumerate(train_loader):
        # ---- assign data
        image = image.to(device)
        label = label.to(device)
        # ---- forward
        optimizer.zero_grad()
        output = net(image)
        # ---- backward
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print(f'[Epoch: {epoch}] cost = {avg_cost}')
print('Learning Finished !')



    # CNN train - Ex 6
# lr_scheduler 사용
print(len(trainloader))
epochs = 50

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    lr_sche.step()
    for batch_idx, (image, label) in enumerate(trainloader, 0):
        # get the inputs
        image = image.to(device)
        label = label.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = vgg16(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 30 == 29: # print every 30 mini-batches
            print(f'[{epoch}, {batch_idx}, loss: {running_loss / 30:.3f}]')
            running_loss = 0
print('Finished Training')



    # CNN train, valid(test) - Ex 7
total_batch = len(train_loader)
loss_arr = []
epochs = 20

for epoch in range(epochs):  # vgg16 velog
    avg_loss = 0
    lr_sche.step()
    for num, (image, label) in enumerate(train_loader):
        x = image.to(DEVICE)
        y_ = label.to(DEVICE)

        optimizer.zero_grad()
        output = model.forward(x)

        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    loss_arr.append(avg_loss)
    print(f'[Epoch: {epoch+1} loss = {avg_loss}]')
print('Training Finished !')

correct = 0
total = 0

with torch.no_grad():
    for num, (image, label) in enumerate(test_loader):
        x = image.to(DEVICE)
        y_ = label.to(DEVICE)
        outputs = model.forward(x)

        _, predicted = torch.max(outputs.data, 1)

        total += y_.size(0)
        correct += (predicted == y_).sum().item()

print(f'Accuracy: {correct / total * 100}%')



    # CNN test - Ex 1
with torch.no_grad():
    start = time.time()
    test_acc=0
    for image, label in test_loader:
        # ------- assign valid data
        image = image.to(device)
        label = label.to(device)
        # ------- forward prop
        test_output = model(image)
        # ------- get valid performance
        test_batch_acc = ((test_output.argmax(dim=1) == label).float().mean()) # acc = 맞춘 개수 / 배치사이즈
        test_acc += test_batch_acc / len(test_loader) # acc / total_Iteration
    print(f'test_acc : {test_acc:.4f}, takes {time.time() - start}secs')


    # CNN test 함수화 - Ex 2
def test(model, test_loader, epochs):
    start = time.time()
    test_acc = 0
    for num, (images, labels) in enumerate(test_loader):
        test_images = images.to(device)
        test_labels = labels.to(device)

        test_outputs = model(test_images)

        test_batch_acc = ((test_outputs.argmax(dim=1) == test_labels).float().mean()) # acc = 맞춘 개수 / 배치사이즈
        test_acc += test_batch_acc / len(test_loader) # acc / total_Iteration
    print(f'test_acc: {test_acc}, takes {time.time()-start:.2f} secs')
test(model, test_loader, epochs)



    # CNN test - Ex 3
with torch.no_grad():
    for num, (image, label) in enumerate(test_loader):
        image = image.to(device)
        label = label.to(device)

        prediction = new_net(image)

        correct_prediction = torch.argmax(prediction, 1) == label

        accuracy = correct_prediction.float().mean()
        print(f'Accuracy: {accuracy}')


    # CNN test - Ex 4
with torch.no_grad(): # 개인적으로 맘에드는 testing 코드 !
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)

    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f'Accuracy: {accuracy}')



    # CNN test - Ex 5
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for image,label in test_loader:
        x = image.to(device)
        y_= label.to(device)

        output = model.forward(x)
        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == y_).sum().float()

    print("Accuracy of Test Data: {}".format(100*correct/total))


    # CNN test - Ex 6
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            test_loss += criterion(output, label).item()

            prediction = output.max(1, keepdim=True)[1]
	# max()하면 해당 tensor에서 가장 큰 값을 반환
	# max(1)하면 해당 tensor에서 열쪽으로 쭉 본것들 중에 가장 큰 값들 반환
	# max(1, keepdim=True) False가 default인데, keepdim값을 켜주면, max로 반환된 값의 열쪽으로의 index값을 반환
	# max(1, keepdim=True)[1] keepdim=True일때 값 반환시, 가장 큰 값들의 텐서와 index값들의 텐서 두개를 반환해서 그 중에 인덱스 반환 텐서를 선택
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            # eq로 prediction의 값들과 일치하는지 T/F로 이루어진 텐서 반환
	# view_as 메소드로 label의 [16]텐서를 [1, 16]텐서로 prediction과 동일하게 변환
	# 비교한 값들의 sum()으로 해당 batch에서 몇개를 맞추었는지 갯수를 item()으로 반환
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, test_accuracy



---------- RNN ----------

    # RNN
 - rnn은 sequential 데이터를 잘 다루기 위해서 도입되었음
   sequential 데이터는 데이터의 값 뿐만 아니라
   데이터의 순서도 중요한 의미를 갖는 데이터를 말함.
   데이터의 순서도 데이터의 일부인 것
   ex: hello

 - rnn은 hidden state를 통해 이전의 정보를 받아들이고 학습하여 순서를 기억함
 - 모든 셀은 파라미터를 공유함: 모든 셀이 A 하나임
 - h_t = f(h_t-1, x_t)
   셀 A에서 함수 연산이 일어나는데
   전단계의 히든 스테이트h_t-1와 지금 단계에서의 입력 값(x_t)을 가지고
   함수 연산을 가지고 h_t를 만듦
   EX)
   h_t = tanh(W_h*h_t-1 + W_x * x_t)

 - 특이한 점이 input값인 x뿐만 아니라 h_t-1에도 weight값이 붙어서 학습한다는 점임
   유명한 설계 방법들은 LSTM과 GRU가 있음



    # torchtext
# 자연어처리(NLP)를 다룰 때 쓸 수 있는 좋은 라이브러리
# 자연어처리 데이터셋을 다루는 데 있어서 매우 편리한 기능을 제공
 º 데이터셋 로드
 º 토큰화(Tokenization)
 º 단어장(Vocabulary) 생성
 º Index mapping: 각 단어를 해당하는 인덱스로 매핑
 º 단어 벡터(Word Vector): word embedding을 만들어준다. 0이나 랜덤 값 및 사전학습된 값으로 초기화할 수 있다.
 º Batch 생성 및 (자동) padding 수행



    # Usages of RNN
one to one: 일반적인 neural network: 하나의 입력에 하나의 출력이 나옴

one to many: 이미지 데이터 하나가 나오고 출력값으로는 문장이 나온다.
 - ex: image captioning: image -> sequence of words

many to one: 문장이 입력되고 하나의 값이 나옴
 - ex: Sentiment Classification: sequence of words ->  sentiment

many to many: 문장이 들어오고 문장이 출력되는 형태
 -  Machine Translation: sequence of words -> sequence of words

many to many: 여러개의 input이 있고 들어올때마다 새로 output들이 나오는 다른 버전
 - Video classification on frame level


    # RNN applications
- Language Modeling
- Speech Recognition
- Machine Traslation
- Conversation Modeling / Question Answering
- Image / Video Captioning
- Image / Music / Dance Generation



    # 순환 신경망 연산이 이루어지는 방식
 - t=1일 때 은닉층 각 노드의 값은, 이전 시간(t=0)의 은닉층 값과 현재 시간(t=1)의 입력값의 조합으로 값이 계산된다고 할 수 있음.
   pytorch라는 단어를 예로 들면, 전에 p라는 단어가 들어왔었다는 것을 기억하고 있는 상태로 이번에 y가 들어왔을 때
   다음에 나와야 할 알파벳이 무엇인지 예측하는 방식.

    # 시간에 따라 풀어서 본 순환 신경망
 - 은닉층의 노드들은 어떠한 초깃값을 가지고 계산이 시작되고
   첫 번째 입력값이 들어온 t=0시점에서 입력값과 초깃값을 조합으로 은닉층의 값들이 계산되게 됨.
   이 시점에서 결괏값이 도출되면. t=1시점에서는 새로 들어온 입력값과 t=0 시점에서 계산된 은닉층의 값과의 조합으로
   t=1일 때 은닉층의 값과 결괏값이 다시 계산되게 됨. 이러한 과정이 지정한 시간만큼 반복됨.

 - 순환 신경망은 계산에 사용된 시점의 수에 영향을 받음.
   예를 들어 t=0에서 t=2까지 계산에 사용됐다면 그 시간 전체에 대해 역전파를 해야 하는 것임.
   이를 시간에 따른 역전파(backprpagation through time) BPTT라고 부름.




    # RNN hello - Ex
input_size = 4  # 4개의 차원을 받는다: h, e, l, o
hidden_size = 2 # hidden state의 벡터 디멘션을 정의
                # = 몇 차원의 출력(output)을 원하는지
                # 즉, output size = hidden size

# 1-hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
input_data_np = np.array([[h, e, l, l, o],
                          [e, o, l, l, l],
                          [l, l, e, e, l]
                         ], dtype = np.float32)

input_data = torch.Tensor(input_data_np)
rnn = torch.nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)

    # RNN hello - Ex 설명
input_data.shape  -> (-, -, 4)
output_data.shape -> (-, -, 2)
        # Sequence Length
hello를 예를 들면 sequence length = 5이다.
이러한 sequence length는 PyTorch에서 자동적으로 계산된다.
input_data.shape -> (-, 5, 4)  가운데 5가 sequence length이다.
output_data.shpae -> (-, 5, 2)
# Batch Size
여러개의 데이터를 하나의 batch로 묶어서 모델에게 학습시킴
batch size역시 PyTorch에서 자동으로 파악함
input_data.shape -> (3, 5, 4)
output_data.shape -> (3, 5, 2)
# ----- 따라서 input_data와 hidden_size만 잘 정의하여 주면 된다!!!!



    # RNN hihello Ex
# Random seed to make results deterministic and reproducible
torch.manual_seed(0)
# declare dictionary
char_set = ['h', 'i', 'e', 'l', 'o']
# hyper parameters
input_size = len(char_set) # 몇 차원의 input을 받을지 -> char_set의 유니크 값의 개수
hidden_size = len(char_set)
learning_rate = 0.1
# data setting
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]
# transform as torch tensor variable
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
# declare RNN
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)  # batch_first guarantees the order of output = (B, S, F): batch_size, Sequence_length, Feature
# loss & optimizer setting
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)
# start training
for i in range(100):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)



    # RNN charseq Ex
# Random seed to make results deterministic and reproducible
torch.manual_seed(0)
sample = " if you want you"
# make dictionary
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}
print(char_dic)
# hyper parameters
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1
# data setting
sample_idx = [char_dic[c] for c in sample]  # sample의 글자를 인덱스로 숫자화시킴
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]	 # np.eye(size)으로 size에 해당하는 만큼의 항등벡터를 만듦
					 # 그 중 1에 해당하는 값을 뒤에 인덱스로 알려주면 해당 인덱스값이 1이고 나머지는 0으로 채워지는 벡터값을 반환해줌
y_data = [sample_idx[1:]]
# transform as torch tensor variable
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
# declare RNN
rnn = torch.nn.RNN(dic_size, hidden_size, batch_first=True)
# loss & optimizer setting
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)
# start training
for i in range(50):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)





    # LSTM(Long short-term memory)
 - 기존 순환 신경망 모델에 장기기억(long-term memory)을 담당하는 부분을 추가한 것.
   기존에는 은닉 상태(hidden state)만 있었다면 셀 상태(cell state)라는 이름을 가지는 전달 부분을 추가한 것.
   기본적인 순환 신경망에 비해 하나의 상태가 더 생긴 것을 확인 할 수 있음(input 값이 하나 더 증가)

# 셀 상태
 - 셀 상태는 장기기억을 담당하는 부분으로, 곱하기(x) 부분은 기존의 정보를 얼마나 남길 것인지에 따라
   비중을 곱하는 부분이고, 더하기(+) 부분은 현재 들어온 데이터와 기존의 은닉 상태를 통해 정보를 추가하는 부분임.

# 망각 게이트
 - 기존의 정보들로 구성되어 있는 셀 상태의 값을 얼마나 잊어버릴 것인지 정하는 부분.
   현재 시점의 입력값과 직전 시점의 은닉 상태 값을 입력으로 받는 한층의 인공 신명아이라고 보면 간단히 이해할 수 있음.
   가중치를 곱해주고 바이어스를 더한 값을 시그모이드 함수에 넣어주면 0에서 1값이 나오는데 이 값으로
   기존의 정보를 얼마나 전달할지 비중을 정하는 것이라고 할 수 있음

# 입력 게이트
 - 어떤 정보를 얼마큼 셀 상태에 새롭게 저장할 것인지 정하는 부분.
   기존의 방식과 비슷하게 새로운 입력값과 직전 시점의 은닉 상태를 받아서 한 번은 시그모이드 활성화 함수를 통과시키고
   또 한 번은 하이퍼볼릭 탄젠트 활성화 함수를 통과시킴.
   하이퍼볼릭 탄젠트를 통해 나온 값은 -1에서 1사이의 값을 가지고 새롭게 셀 상태에 추가할 정보가 됨.
   시그모이드 함수를 통해 나온 값은 0에서 1사이의 비중으로 새롭게 추가할 정보를
   얼마큼의 비중으로 셀 상태에 더해줄지 정하게 됨.

# 셀 상태의 업데이트
 - 현재 시점의 새로운 입력값과 직전 시점의 은닉 상태 값의 조합으로 기존의 셀 상태의 정보를
    얼마큼 전달할지도 정하고 어떤 정보를 얼마큼의 비중으로 더할지로 정하는 것이라 할 수 있음.

# 은닉 상태의 업데이트
 - 새로운 은닉 상태는 업데이트된 셀 상태 값을 하이퍼볼릭 탄젠트 함수를 통과시킨 -1에서 1사이의 비중을 곱한 값으로 생성됨.



 # GRU(gated recurrent unit)
  - GRU는 LSTM과 달리 셀 상태와 은닉 상태를 분리하지 않고 은닉 상태 하나로 합쳤음.


---------- AE, AutoEncoder ----------


    # AE, AutoEncoder, 오토인코더
 - 오토인코더는 데이터에 대한 효율적인 압축을 신경망을 통해 자동으로 학습하는 모델입니다.
   오토인코더는 입력 데이터 자체가 라벨로 사용되기 때문에 비지도학습에 속합니다.

 - 보통 입력 데이터의 차원보다 낮은 차원으로 압축하기 때문에,
   효율적인 인코딩(efficient data encoding), 특성학습(feature learning),
   표현 학습(representation learning)의 범주에 속하기도 하고 차원 축소(dimensionality reduction)의 한 방법이기도 합니다.

 - 왼쪽에서 입력 X가 들어와 신경망을 통해 잠재 변수(latent variable) z가 됩니다.
   압축된 z는 다시 신경망을 통과해 출력 X'가 됩니다.

 - 손실은 X와 X'에서 같은 위치에 있는 픽셀 간의 차이를 더한 값이라 할 수 있습니다.
   이 출력과 입력 간의 차이로 모델의 가중치를 업데이트 합니다.



# AE model - Ex 1
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),)

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# AE model - Ex 2
# 인공신경망으로 이루어진 오토엔코더를 생성합니다.
# 단순하게 하기 위해 활성화 함수는 생략했습니다.
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)

    def forward(self,x):
        x = x.view(batch_size,-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size,1,28,28)
        return out



# AE train - Ex 1
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.view(-1, 28 * 28).to(DEVICE)
        target = image.view(-1, 28 * 28).to(DEVICE)

        optimizer.zero_grad()

        encoded, decoded = model(image)
        loss = criterion(decoded, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {Epoch}, \
                     [{batch_idx * len(image)}/{len(train_loader.dataset)} \
                     ({100 * batch_idx / len(train_loader):.0f}%)], \
                     Train Loss {loss.item():.6f}')

# AE train - Ex 2
loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,x)
        loss.backward()
        optimizer.step()

        if (j+1) % 50 == 0:
            print((j+1), loss)
            loss_arr.append(loss.item())

# AE test - Ex 1
def evaluate(model, test_loader):
    model.eval()
    avg_loss = 0
    real_image = []
    gen_image = []

    with torch.no_grad():
        for image, _ in test_loader:
            image = image.view(-1, 28 * 28).to(DEVICE)
            target = image.view(-1, 28 * 28).to(DEVICE)
            encoded, decoded = model(image)

            avg_loss += criterion(decoded, target).item()
            real_image.append(image.to("cpu"))
            gen_image.append(decoded.to("cpu"))
    avg_loss /= len(test_loader.dataset)

    return avg_loss, real_image, gen_image



# AE 함수 - Ex 1
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    avg_loss, real_image, gen_image = evaluate(model, test_loader)
    print(f'\n[EPOCH: {Epoch}], \tAvg_Loss: {avg_loss:.4f}')

    f, a = plt.subplots(2, 10, figsize=(10, 4))
    for i in range(10):
        img = np.reshape(real_image[0][i], (28, 28))
        a[0][i].imshow(img, cmap='gray_r')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(10):
        img = np.reshape(gen_image[0][i], (28, 28))
        a[1][i].imshow(img, cmap='gray_r')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
    plt.show()



# AE 이미지 확인 - MNIST
out_img = torch.squeeze(outputs.cpu().data)
print(out_img.size())

for i in range(10):
    fig, axes = plt.subplots(1, 2, figsize=(3, 2))
    axes[0].imshow(torch.squeeze(images[i]).cpu().numpy(), cmap='gray')
    axes[1].imshow(out_img[i].cpu().numpy(), cmap='gray')
    plt.show()




    # CAE, 합성곱 오토인코더
 - 합성곱 연산을 오토인코더에 적용

 - 전치합성곱: 하나의 입력값을 받아 여기에 서로 다른 가중치를 곱해 필터의 크기만큼 입력값을 '퍼뜨리는' 역할을 합니다.
   하나의 입력에 대해서 커널 사이즈 만큼의 결과가 생성됩니다.
   파이토치에서는 이미지 데이터에 대해 nn.ConvTranspose2d 함수를 사용해 전치 합성곱 연산을 합니다.

 - 전치 합성곱에서는 패딩은 결괏값에서 제일 바깥 둘레를 빼주는 역할을 합니다.

 - 아웃풋패딩(output_padding)은 결과로 나오는 텐서의 크기를 맞추기 위해 있는 인수입니다.
   padding인수로 잘리는 부분을 줄여주는 역할을 합니다

 - 패딩없이 아웃풋패딩을 하게 되면 테두리를 0으로 채누는 결과가 됩니다.

 - 전치 컨볼루션 연산으로 이미지 크기를 2배로 늘리는 방법 2가지 둘중에 kernel_size=4,stride=2,padding=1 세팅이 체커보드 아티팩트가 덜합니다.

# CAE model(convolutional AE) - Ex 1
# (Encoder, Decoder class 따로 정의)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,16,3,padding=1),                            # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16,32,3,padding=1),                           # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),                           # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)                                       # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),                          # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),                         # batch x 64 x 7 x 7
                        nn.ReLU()
        )


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),                    # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),                       # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,16,3,1,1),                        # batch x 16 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.ConvTranspose2d(16,1,3,2,1,1),                       # batch x 1 x 28 x 28
                        nn.ReLU()
        )

    def forward(self,x):
        out = x.view(batch_size,256,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

# 인코더 디코더의 파라미터를 동시에 학습시키기 위해 이를 묶는 방법입니다.
parameters = list(encoder.parameters())+ list(decoder.parameters())

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)


# CAE train - Ex 1
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        optimizer.zero_grad()
        image = image.to(device)

        output = encoder(image)
        output = decoder(output)

        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()

    if j % 10 == 0:
        # 모델 저장하는 방법
        # 이 역시 크게 두가지 방법이 있는데 여기 사용된 방법은 좀 단순한 방법입니다.
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save([encoder,decoder],'./model/conv_autoencoder.pkl')
        print(loss)


out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

for i in range(5):
    plt.subplot(1,2,1)
    plt.imshow(torch.squeeze(image[i]).cpu().numpy(),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()



# CAE test - Ex 1
with torch.no_grad():
    for j,[image,label] in enumerate(test_loader):

        image = image.to(device)
        output = encoder(image)
        output = decoder(output)

    if j % 10 == 0:
        print(loss)

out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

for i in range(5):
    plt.subplot(1,2,1)
    plt.imshow(torch.squeeze(image[i]).cpu().numpy(),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()



    # Denoising AE, 디노이징 오토인코더
 - 입력으로 들어오는 이미지에 특정 노이즈를 추가한 손상된 데이터를 넣어서,
   노이즈를 제거하는 것도 가능합니다.
   예를 들어 가우시안 노이즈를 데이터에 추가하고 모델을 통과한 결괏값이
   노이즈 없는 깨끗한 데이터로 복원될 수 있다면, 그냥 압축하는 모델이 아니라
   노이즈도 제거하는 모델 또한 만들 수 있습니다.

noise = init.normal_(torch.FloatTensor(batch_size, 1, 28, 28), 0, 0.1) # MNIST 예제
noise_image = image + noise




    # U-Net
 - 전체적으로 합성곱 오토인코더의 형태를 따르고 있는 걸 알 수 있으나
   기존의 오토인코더 모델과 다른 점은 회색 화살표로 표시된 copy and crop 연산이라고 할 수 있습니다.
 - 이런 방식을 스킵 커넥션(skip connection)이라고 하는데 앞에서 배운 ResNet 모델에서는 이를 텐서 간의 합으로 사용했고,
   U-Net에서는 합 대신 텐서 간의 연결(concatenation)로 사용했습니다. -> DenseNet 느낌
 - 오토인코더에서는 입력 이미지가 압축되다 보면 위치 정보가 어느정도 손실되게 됩니다.
   그렇게 되면 다시 원본 이미지 크기로 복원하는 과정에서 정보의 부족 때문에
   원래 물체가 있었던 위치에서 어느 정도 이동이 일어나게 됩니다.
   이런 복원 과정에 스킵 커넥션을 사용하게 되면 원본 이미지의 위치 정보를 추가적으로 전달받는 셈이 되므로
   비교적으로 정확한 위치를 복원할 수 있게 되고 따라서 분할 결과도 좋아지게 됩니다.
 - U-Net은 세그멘테이션 모델이나 이미지 간 이전 모델에서 가장 기본이 되는 형태이고,
   합성곱 연산에 ResNet의 스킵 커넥션을 추가한 FusionNet같은 모델도 있습니다.

# U-Net Architecture
![대체 텍스트](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

# 자주 쓰는 연산들과 항상 세트로 쓰는 연산들은 편의를 위해 함수로 정의해 놓습니다.
def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class UnetGenerator(nn.Module):
    def __init__(self,in_dim,out_dim,num_filter):
        super(UnetGenerator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn)

        self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
        self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
        self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
        self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
        self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
        self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
        self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
        self.up_4 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter,self.out_dim,3,1,1),
            nn.Tanh(),  #필수는 아님
        )

    def forward(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1,down_4],dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_3],dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_2],dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4,down_1],dim=1)
        up_4 = self.up_4(concat_4)
        out = self.out(up_4)
        return out

batch_size = 16
img_size = 256
in_dim = 1
out_dim = 3
num_filters = 16

sample_input = torch.ones(size=(batch_size,1,img_size,img_size))

model = UnetGenerator(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters)
output = model(sample_input)
print(output.size())





---------- GAN ----------


    # GAN (Generative Adversarial Network) 생성적 적대 신경망
 - Generative(생성적): 생성 모델은 데이터 자체를 만들어내기 때문에 특성을 뽑아내는
   모델보다 더 어려운 작업을 수행하는 것이고 따라서 학습도 좀 더 어렵습니다.
   이러한 생성 모델의 대표적인 사례가 바로 GAN과 변분 오토인코더입니다.
 - Adversarial(적대적): GAN에서는 생성 네트워크와 구분 네트워크 간의
   상반되는 목적 함수(objective function)로 인해 적대성이 생기게 됩니다.
   많이 드는 예시로 위조지폐를 만드는 사람(생성 네트워크)과 위조지폐 감별사(구분 네트워크)가 있습니다.
 - Network(네트워크): 네트워크는 우리가 딥러닝에서 흔히 사용하는 의미대로,
   신경망의 형태를 가진 다양한 네트워크를 의미합니다.
   GAN에서 생성자와 구분자의 구조가 인공 신경망의 형태를 이룹니다.

 - 생성자: 생성자는 어떠한 입력 z를 받아서 가짜fake 데이터를 생성합니다.
 - 구분자: 구분자는 실제 데이터와 가짜 데이터를 받아서 각각 실제인지 아닌지 구분하게 됩니다.


    # GAN FLOW (MNIST Example)
 - 먼저 생성자는 노이즈를 z로 받습니다. 특별한 조건 없이 이미지를 생성해야 하기 때문에
   랜덤한 노이즈를 사용한다고 보면 되며, 표준정규분포(N(0,1))를 따르는 데이터면 충분합니다.
   z 벡터의 길이는 생성하는 데이터의 종류마다 다르지만 MNIST의 경우는 50정도면 충분하다고 알려져 있습니다.
   이러한 z가 입력으로 들어오면 생성자는 신경망이나 합성곱 신경망을 통해서 MNIST 데이터와 같은 형태의 데이터를 생성해냅니다.
 - 구분자는 MNIST 형태의 데이터를 입력으로 받아서 하나의 결괏값을 내는 네트워크입니다.
   구분자에는 실제 데이터가 들어가기도 하고 생성자에서 만들어진 데이터가 들어가기도 합니다.
   당연히 구분자는 실제 데이터가 들어오면 실제라고 구분해야 하고 가짜 데이터가 들어오면 가짜라고 구분해야 합니다.
 - <구분자 입장>에서 실제 데이터는 1에, 가짜 데이터는 0에 가깝게 나오도록 학습되지만
   <생성자 입장>에서는 생성한 가짜 데이터가 1에 가깝게 나오는 것이 목표이기 때문에
   가짜 데이터의 구분에 대해 서로 경쟁을 하게 됩니다.


    # GAN 목적함수
목적 함수: minGmaxD V(D,G)= {log(D(x))+log(1−D(G(z)))}
 - <구분자 입장>에서는 이 값을 최대화해야 하는데,
   D(x)는 1, D(G(z))는 0이 되어야 합니다.
   실제 데이터는 구분자를 통과했을 때 1, 생성된 가짜 데이터는 0으로 판단되어야 한다는 것과 일치합니다.
 - <생성자 입장>에서는 이 값을 최소화해야 하는데,
   그러려면 생성자가 관여하는 log(1−D(G(z)))에서
   1−D(G(z))가 0이 되어야 하고 이는 D(G(z))가 1이어야 한다는 것과 같습니다.
   이 부분은 구분자의 목적과 정확히 반대이고 여기서 적대성이 발생하게 됩니다.
 - 구분자, 생성자 모두 손실함수로 BCELoss()를,
   라벨은 실제 데이터에 대한 라벨1로 계산합니다.
   처음 GAN은 이진 교차 엔트로피 손실 함수를 사용했지만 LSGAN(least squares GAN) 등에서
   L2함수를 사용하여 더 안정적인 학습을 달성했으므로, 이 책에서도 L2 손실 함수를 사용하겠습니다.


    # Generator receives random noise z and create 1x28x28 image
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([ # orderdDict 자료형은 순서를 기억하지 못하는 일반 딕셔너리와 다르게
                                                  # 순서가 지정되는 딕셔너리 입니다. 이를 이용해 레이어 순서에 따라 이름을 지정했습니다.
                        ('fc1',nn.Linear(z_size,middle_size)),
                        ('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.ReLU()),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc2', nn.Linear(middle_size,784)),
                        #('bn2', nn.BatchNorm1d(784)),
                        ('tanh', nn.Tanh()),
        ]))
    def forward(self,z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = out.view(batch_size,1,28,28)
        return out

    # Discriminator receives 1x28x28 image and returns a float number 0~1
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(784,middle_size)),
                        #('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.LeakyReLU()),

        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc2', nn.Linear(middle_size,1)),
                        ('bn2', nn.BatchNorm1d(1)),
                        ('act2', nn.Sigmoid()), # 0에서 1 사이의 값으로 만들기 위해 마지막 층에는 시그모이드 함수가 들어가 있습니다.
        ]))

    def forward(self,x):
        out = x.view(batch_size, -1)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

# Put class objects on Multiple GPUs using
# torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# device_ids: default all devices / output_device: default device 0
# along with .cuda()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

generator = nn.DataParallel(Generator()).to(device)
discriminator = nn.DataParallel(Discriminator()).to(device)

# Get parameter list by using class.state_dict().keys()
gen_params = generator.state_dict().keys()
dis_params = discriminator.state_dict().keys()
for i in gen_params:
    print(i)

# loss function, optimizers, and labels for training
loss_func = nn.MSELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

ones_label = torch.ones(batch_size,1).to(device)
zeros_label = torch.zeros(batch_size,1).to(device)


# train
for i in range(epoch):
    for j,(image,label) in enumerate(train_loader):
        image = image.to(device)
        # ----------- 구분자 학습
        dis_optim.zero_grad()
        # Fake Data
        # 랜덤한 z를 샘플링해줍니다.
        z = init.normal_(torch.Tensor(batch_size,z_size),mean=0,std=0.1).to(device)
        gen_fake = generator.forward(z) # 가짜 데이터 생성
        dis_fake = discriminator.forward(gen_fake)
        # Real Data
        dis_real = discriminator.forward(image)
        # 두 손실을 더해 최종손실에 대해 기울기 게산을 합니다.
        dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
        dis_loss.backward(retain_graph=True)
        dis_optim.step()

        # ----------- 생성자 학습
        gen_optim.zero_grad()
        # Fake Data
        z = init.normal_(torch.Tensor(batch_size,z_size),mean=0,std=0.1).to(device)
        gen_fake = generator.forward(z)
        dis_fake = discriminator.forward(gen_fake)

        gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real
        gen_loss.backward()
        gen_optim.step()

        # model save
        if j % 100 == 0:
            print(gen_loss,dis_loss)
            torch.save([generator,discriminator],'./model/vanilla_gan.pkl')
            v_utils.save_image(gen_fake.cpu().data[0:25],"./result/gen_{}_{}.png".format(i,j), nrow=5)
            print("{}th epoch gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))



    # DCGAN (deep convolutional GAN)
 - DCGAN은 GAN에 합성곱 연산을 적용한 것입니다.
   합성곱 신경망과 비지도학습의 한 종류인 GAN을 결합한 논문이자,
   모델이 그냥 결과를 생성하는 것이 아니라 어떤 의미를 가지는 특성(또는 표현)을 학습하여
   생성할 수 있다는 것을 보여주었습니다. DCGAN의 생성자 네트워크는 전치 합성곱 연산을 통해
   랜덤 노이즈로부터 데이터를 생성해냅니다. 구분자 네트워크 역시 합성곱 연산으로 이루어져 있습니다.
   모델이 데이터를 외운것이 아니라 어떤 특성들을 학습했다는 점이 중요합니다.

    # 어떻게 학습시키면 DCGAN 학습이 잘되는지
  - 풀링 연산을 합성곱 연산으로 대체하고 생성자 네트워크는 전치 합성곱 연산을 사용한다.
  - 생성자와 구분자에 배치 정규화를 사용한다.
  - 완전연결 네트워크를 사용하지 않는다.
  - 생성자 네트워크에는 마지막에 사용되는 하이퍼볼릭 탄젠트 함수 외에는 모든 활성화 함수에 렐루를 사용한다.
  - 구분자 네트워크의 모든 활성화 함수로는 리키렐루를 사용한다.



---------- Object Detection ----------


    # 20년도 까지 나온 OD기법들 나열
- https://github.com/hoya012/deep_learning_object_detection




---------- Style Transfer ----------


    # 스타일 트랜스퍼 style transfer
 - 스타일 트랜스퍼는 전이학습의 단적인 예라고 할 수 있습니다.
 - 학습된 필터들의 특성은 바로 범용성(여러 분야나 용도로 널리 쓰일 수 있는 특성)에 있습니다.
   가로세로 대각선 필터들은 사실 작업 종류와는 관계없이 물체를 인식하는 데 모두 적용될 수 있기 때문에
   학습된 모델에서 얻은 지식을 다른 작업에 전이할 수 있는 것입니다.

    # 스타일
 - 논문에서 스타일은 다른 필터 응답들 간의 연관성(correlations between the different filter response)라 서술
 - 필터 활성도의 그람 행렬로 나타냄.
 - 그람 행렬은 내적이 정의된 공간에서 벡터 v1, v2, vn,... 이 있을 때 가능한 모든 경우의 내적을 행렬로 나타낸 것

    # 콘텐츠
 - 스타일과 대비되는 형태를 의미
 - 더 높은 레이어 내의 특성 응답(feature responses in higher layers of the network)이라 정의.
 - 특성 응답은 활성화 지도를 의미하고, 더 높은 레이어라 한 것은 모델에서 어느 정도 깊이가 있는 지점을 의미

총 손실은 콘텐츠 손실과 스타일 손실에 각각 가중치 α, β를 곱해서 합한 값이 됨. 왼쪽과 가운데를 비교해서 스타일 손실을 계산하고,
가운데와 오른쪽을 비교해서 콘텐츠 손실을 계산함.
스타일 손실은 모든 위치에서 발생하지만 콘텐츠 손실은 conv_4에서만(특정 깊이의 위치) 발생하는 것을 알 수 있습니다.
스타일 손실을 모든 위치에서 계산하는 이유는 모델의 위치에 따라 수용 영역(receptive field)가 달라지기 때문입니다.
 -> 좁은 영역의 스타일부터 넓은 영역의 스타일까지 다양하게 보겠다는 의미.
     좁은 수용 영역에서 뽑은 스타일은 세밀한데 비해, 넓은 수용 영역에서 뽑아낸 스타일은 전체적인 스타일과 가까움.
논문에서 con_v에서 콘텐츠 손실을 계산한 것은 형태를 보존하면서도 스타일을 잘 입힐 수 있도록 실험을 통해 적절한 위치를 찾은 것으로 보임.

스타일 트랜스퍼를 구현할 때는 2차 미분 값까지 이용한 L-BFGS(limited-memory BFGS) 알고리즘을 사용하는 편


## Image Style Transfer ##
< 목차 >
1. Settings
 1) Import required libraries
 2) Hyperparameter
2. Data
 1) Directory
 2) Preprocessing Function
 3) Postprocessing Function
3. Model & Loss Function
 1) Resnet
 2) Delete Fully Connected Layer
 3) Gram Matrix Function
 4) Model on GPU
 5) Gram Matrix Loss
4. Train
 1) Prepare Images
 2) Set Targets & Style Weights
 3) Train
5. Check Results


# 컨텐츠 손실을 어느 지점에서 맞출것인지 지정해놓습니다.
content_layer_num = 1
image_size = 512
epoch = 5000

content_dir = "./images/content/Tuebingen_Neckarfront.jpg"
style_dir = "./images/style/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

# 이미 학습된 ResNet 모델이 이미지넷으로 학습된 모델이기 때문에 이에 따라 정규화해줍니다.
def image_preprocess(img_dir):
    img = Image.open(img_dir)
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                         std=[1,1,1]),
                ])
    img = transform(img).view((-1,3,image_size,image_size))
    return img


# 정규화 된 상태로 연산을 진행하고 다시 이미지화 해서 보기위해 뺐던 값들을 다시 더해줍니다.
# 또한 이미지가 0에서 1사이의 값을 가지게 해줍니다.

def image_postprocess(tensor):
    transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                     std=[1,1,1])
    img = transform(tensor.clone())
    img = img.clamp(0,1)
    img = torch.transpose(img,0,1)
    img = torch.transpose(img,1,2)
    return img

# 미리 학습된 resnet50를 사용합니다.
resnet = models.resnet50(pretrained=True)
for name,module in resnet.named_children():
    print(name)

# 레이어마다 결과값을 가져올 수 있게 forward를 정의합니다.

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self,x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

# 그람 행렬을 생성하는 클래스 및 함수를 정의합니다.
# [batch,channel,height,width] -> [b,c,h*w]
# [b,c,h*w] x [b,h*w,c] = [b,c,c]

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        return G

# 모델을 학습의 대상이 아니기 때문에 requires_grad를 False로 설정합니다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

resnet = Resnet().to(device)
for param in resnet.parameters():
    param.requires_grad = False

# 그람행렬간의 손실을 계산하는 클래스 및 함수를 정의합니다.

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out

# 컨텐츠 이미지, 스타일 이미지, 학습의 대상이 되는 이미지를 정의합니다.

content = image_preprocess(content_dir).to(device)
style = image_preprocess(style_dir).to(device)
generated = content.clone().requires_grad_().to(device)

print(content.requires_grad,style.requires_grad,generated.requires_grad)

# 각각을 시각화 합니다.

plt.imshow(image_postprocess(content[0].cpu()))
plt.show()

plt.imshow(image_postprocess(style[0].cpu()))
plt.show()

gen_img = image_postprocess(generated[0].cpu()).data.numpy()
plt.imshow(gen_img)
plt.show()


# 목표값을 설정하고 행렬의 크기에 따른 가중치도 함께 정의해놓습니다

style_target = list(GramMatrix().to(device)(i) for i in resnet(style))
content_target = resnet(content)[content_layer_num]
style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

# LBFGS 최적화 함수를 사용합니다.
# 이때 학습의 대상은 모델의 가중치가 아닌 이미지 자체입니다.
# for more info about LBFGS -> http://pytorch.org/docs/optim.html?highlight=lbfgs#torch.optim.LBFGS

optimizer = optim.LBFGS([generated])

iteration = [0]
while iteration[0] < epoch:
    def closure():
        optimizer.zero_grad()
        out = resnet(generated)

        # 스타일 손실을 각각의 목표값에 따라 계산하고 이를 리스트로 저장합니다.
        style_loss = [GramMSELoss().to(device)(out[i],style_target[i])*style_weight[i] for i in range(len(style_target))]

        # 컨텐츠 손실은 지정한 위치에서만 계산되므로 하나의 수치로 저장됩니다.
        content_loss = nn.MSELoss().to(device)(out[content_layer_num],content_target)

        # 스타일:컨텐츠 = 1000:1의 비중으로 총 손실을 계산합니다.
        total_loss = 1000 * sum(style_loss) + torch.sum(content_loss)
        total_loss.backward()

        if iteration[0] % 100 == 0:
            print(total_loss)
        iteration[0] += 1
        return total_loss

    optimizer.step(closure)


# 학습된 결과 이미지를 확인합니다.

gen_img = image_postprocess(generated[0].cpu()).data.numpy()

plt.figure(figsize=(10,10))
plt.imshow(gen_img)
plt.show()



---------- Import Codes ----------

# Data Handling
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')


# visualization
import matplotlib.pyplot as plt
%matplotlib inline


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
import torchvision.models as models
import torchvision.utils as v_utils
import torchvision.transforms as transforms



# For reproducibility
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'




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



    # Activation Function 설명
 - 활성함수는 어떤 신호를 입력받아 이를 적절히 처리해 출력해주는 함수를 의미합니다.

 - 시그모이드를 activation 함수로써 쓰게 되면 Back Propagation 과정 중에 미분한 값을 계속 곱해주면서 Gradient값이 앞 단의 Layer로 올수록 0으로 수렴하는 현상이 발생합니다.
   이를 'Gradient Vanishing'이라 하며 이는 Hidden Layer가 싶어질수록 심해지기 때문에 Hidden Layer를 깊게 쌓아 복잡한 모델을 만들 수 있다는 장점이 없게 됩니다.

 - ReLU(Rectified Linear Unit)함수는 기존의 시그모이드 함수와 같은 비선형 활성 함수가 지니고 있는 문제점을 어느 정도 해결한 활성 함수 입니다.
   활성 함수 ReLU는 f(x) = max(0, x)와 같이 정의어서 입력 값이 0 이상이면 이 값을 그대로 출력하고, 0 이하이면 0으로 출력하는 함수입니다.
   이 활성 함수가 시그모이드 함수에 비해 좋은 이유는 이 활성 함수를 미분할 때 입력 값이 0 이상인 부분은 기울기가 1, 입력 값이 0 이하인 부분은 0이 되기 때문입니다.
   즉, Back Propagation 과정 중 곱해지는 Activation 미분값이 0 또는 1이 되기 때문에 아예 없애거나 완전히 살리는 것으로 해석할 수 있습니다.

 - ReLU의 변형 함수인 다양한 함수가 나오기 시작했습니다. Leaky ReLU, ELU, parametric ReLU, SELU, SERLU 등..

 - Leaky ReLU는 수식을 f(x) = max(ax, x)로 변형시키고 상수 a에 작은 값을 설정함으로써 0 이하의 자극이 들어왔을 때도 활성화 값이 전달 되게 합니다
   예를 들어 a가 0.2라고 하면 0이하의 자극에는 0.2를 곱해 전달하고 0보다 큰 자극은 그대로 전달하는 활성화 함수가 됩니다.
   이렇게 되면 역전파가 일어날 때, 0 이하인 부분에서는 가중치가 양의 방향으로 업데이트되고 0보다 큰 부분에서는 음의 방향으로 업데이트되므로 다잉 뉴런 현상을 방지할 수 있습니다.

 - 랜덤 리키 렐루의 경우는 a의 값을 랜덤하게 지정하는 활성화 함수입니다.

 - 활성함수는 딥러닝을 적용하는 분야에 따라 조금씩 성능의 차이가 있습니다.

    # Pytorch Activation function의 종류
1. Non-linear activations
    º nn.ELU
    º nn.SELU
    º nn.Hardshrink, nn.Hardtanh
    º nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU
    º nn.Sigmoid, nn.LogSigmoid
    º nn.Softplus, nn.Softshrink, nn.Softsign
    º nn.Tanh, nn.Tanhshrink
    º nn.Threshold
2. Non-linear activations(other)
    º nn.Softmin
    º nn.Softmax, nn.Softmax2d, nn.LogSoftmax
    º nn.AdaptiveLogSoftmaxWithLoss
# 본문: https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
# 참조: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity


    # Mish activation function
class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)



    # 연쇄법칙 Chain rule
 - 연쇄법칙을 간단히 설명하면, z가 y에 대한 종속변수이고, y는 x에 대한 종속변수일 때,
   z를 x에 대해 미분한 값은 z를 y에 대해 미분한 값과 y를 x에 대해 미분한 값의 곱과 같다는 의미입니다.
   z가 y에 비례해서 변하고 y는 x에 비례하기 때문에 두 쌍의 관계를 구해놓으면
   z와 x의 관계는 이 두 관계의 곱으로 구할 수 있다고 보면 됩니다.
 - 전파가 입력값이 여러 은닉층을 통과해 결과로 나오는 과정이었다고 하면,
   역전파는 결과와 정답의 차이로 계산된 손실을 연쇄법칙을 이용하여 입력 단까지 다시 전달하는 과정을 의미합니다.



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

    # 어떤 경우에 데이터를 정규분포화 하는가? 왜 하는가?
 y_train = (m, 2)라 하자.
 [1000, 0.1]   이때 뉴럴네트워크는 0.1쪽들의 값들은 작아서
 [999,  0.2],  앞의 값에 더욱 치중해서 계산하라 것이다. 정규분포화 하여
 [1010, -0.3]  둘다 비슷한 값들도 만들어서 피처의 중요성을 동등하게 만들어준다 !!
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma
print(norm_x_train)

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
 - 5. 데이터 늘리기 (노이즈 증식, 또는 아에 늘리기)
 - 6. 정규화 (Data normalization, BatchNormalization)
 - 7. Early Stopping


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








---------- 혼합아이디어 ----------


 - PCA와 유사한 역할을 하는 AE,
   데이터를 학습한 AE의 Encoder를 통과한 z를 feature로써 사용하여 ML/DL에 넣어주어 학습한다









---------- 기타 설명 ----------

* 데이터셋
* Pytorch Layer의 종류
* MNIST Dataset
* CIFAR10 Dataset
* CIFAR100 Dataset
* FashionMNIST Dataset
* Hymenoptera_data
* Python PIL, Pillow
* glob


    # torchvision.datasets
# 참조: https://pytorch.org/docs/stable/torchvision/datasets.html
# Pytorch가 공식적으로 다운로드 및 사용을 지원하는 datasets이다. 2020.02.04 기준 dataset 목록은 다음과 같다.
# MNIST
MNIST(숫자 0~9에 해당하는 손글씨 이미지 6만(train) + 1만(test))
Fashion-MNIST(간소화된 의류 이미지),
KMNIST(일본어=히라가나, 간지 손글씨),
EMNIST(영문자 손글씨),
QMNIST(MNIST를 재구성한 것)
# MS COCO
Captions(이미지 한 장과 이를 설명하는 한 영문장),
Detection(이미지 한 장과 여기에 있는 object들을 segmantation한 정보)
# LSUN(https://www.yf.io/p/lsun)
# ImageFolder, DatasetFolder
# Image
ImageNet 2012,
CIFAR10 & CIFAR100,
STL10, SVHN, PhotoTour, SBU
# Flickr8k & Flickr30k, VOC Segmantation & Detection,
# Cityscapes, SBD, USPS, Kinetics-400, HMDB51, UCF101
# 각각의 dataset마다 필요한 parameter가 조금씩 다르기 때문에, MNIST만 간단히 설명하도록 하겠다. 사실 공식 홈페이지를 참조하면 어렵지 않게 사용 가능하다.



    # Pytorch Layer의 종류
1. Linear layers
    º nn.Linear
    º nn.Bilinear
2. Convolution layers
    º nn.Conv1d, nn.Conv2d, nn.Conv3d
    º nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
    º nn.Unfold, nn.Fold
3. Pooling layers
    º nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
    º nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d
    º nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d
    º nn.FractionalMaxPool2d
    º nn.LPPool1d, nn.LPPool2d
    º nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d
    º nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d
4. Padding layers
    º nn.ReflectionPad1d, nn.ReflectionPad2d
    º nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d
    º nn.ZeroPad2d
    º nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d
5. Normalization layers
    º nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
    º nn.GroupNorm
    º nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
    º nn.LayerNorm
    º nn.LocalResponseNorm
6. Recurrent layers
    º nn.RNN, nn.RNNCell
    º nn.LSTM, nn.LSTMCell
    º nn.GRU, nn.GRUCell
8. Sparse layers
    º nn.Embedding
    º nn.EmbeddingBag
# 본문: https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
# 참조: https://pytorch.org/docs/stable/nn.html#module



    # GlabalAveragePoolin
 - AdaptiveAvgPool2d을 Tensorflow의 GlobalAveragePooling2D처럼 활용하기 위해서는
   output_size 인자로 1을 넣으면 된다.
    # AdaptiveAvgPool2d - Ex 1
torch.nn.AdaptiveAvgPool2d(1)
 - 이때 Global Average Pooling Layer는 각 Feature Map 상의 노드값들의
   평균을 뽑아낸다. 이런 방식으로 Global Average Pooling Layer는
   레이어 집합을 인풋으로 하여 벡터를 아웃풋으로 낸다.



    # MNIST Dataset Example - 1
# Load MNIST data
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)
# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True) # batch_size에 안맞게 마지막에 남은 것: 사용할지 않할지, True시 드랍하여 사용안함.)

# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias=True).to(device)
# define cost/Loss & optimizer
criterion = nn.CrossEntropyLoss().to(device) # Softmax is internally computed
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # Label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)

        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch

    print(f'Epoch: {epoch+1:4d}, cost ={avg_cost:.9f}')
print('Learning finished')


# Test the model using test sets
with torch.no_grad(): # 해당 범위코드내에서는 grad를 계산하지 않겠다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f'Accuracy: {accuracy.item()}')
# 정리 잘 되어 있는 블로그 : https://wingnim.tistory.com/34


    # MNIST Dataset Example - 2
train_dataset = datasets.MNIST(root='MNIST_data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='MNIST_data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)



    # CIFAR10 Dataset Example - 1
# standardization
standardization = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4913, 0.4821, 0.4465], std = [0.2470, 0.2434, 0.2615])
])
train_dataset = datasets.CIFAR10(root = '../data/CIFAR_10', train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = '../data/CIFAR_10', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # CIFAR10 Dataset Example - 2
train_dataset = datasets.CIFAR10(root = '../data/CIFAR_10', train = True, download = True,
 			      transform = transforms.Compose([
                                          	transforms.RandomHorizontalFlip(),
                                          	transforms.ToTensor(),
                                          	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ]))
test_dataset = datasets.CIFAR10(root = '../data/CIFAR_10', train = False, download = True,
                                      transform = transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]))
 - transforms.Compose(): 불러오는 이미지 데이터에 전처리 및 Augmentation을 다양하게 적용할 때 이용하는 메서드. Compose의 괄호 안에 있는 처리 과정을 거친 데이터를 불러오는 것을 의미
 - transforms.RandomHorizontalFlip(): 50% 확률로 이미지를 좌우 반전하는 것을 의미
 - transforms.Normalize(): ToTensor() 형태로 전환된 이미지에 대해 또 다른 정규화를 진행하는 것을 의미. 정규화를 진행할 때는 평균과 표준편차가 필요한데 red, greed, blue 순으로 평균을 '0.5'씩, 표준편차를 '0.5'씩 적용하는 것을 의미.




    # CIFAR100 Dataset Example - 1
# standardization
standardization = transforms.Compose([
    transforms.Resize(256), # 그냥 하면 구글넷 모델에 비해 사진의 크기가 작다는 오류가 발생
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])
train_dataset = datasets.CIFAR100(root = '../data/CIFAR_100', train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR100(root = '../data/CIFAR_100', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)




    # FashionMNIST Dataset Example - 1
train_dataset = datasets.FashionMNIST(root = '../data/FashionMNIST', train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root = '../data/FashionMNIST', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)



    # Hymenoptera_data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224), # 이미지의 중앙 부분을 크롭하여 [size, size] 크기로 만든다
        transforms.Resize(256), # 이미지를 지정한 크기로 변환한다. 직사각형으로 자를 수 있다.
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}
# ImageFolder라는 함수를 이용해 따로 이미지-라벨 쌍을 만들지 않고 폴더에 저장하는것만으로 쉽게 이미지-라벨 쌍을 만들 수 있습니다
ex)
root/dog/xxx.png
root/dog/xxy.png
root/cat/123.png
root/cat/nsdf3.png

image_datasets = {x: datasets.ImageFolder(f'data/hymenoptera_data/{x}', data_transforms[x]) for x in ['train', 'val']}
# 이미지 데이터를 불러오는 것을 의미합니다.
#'../data/hymenoptera_data' 위치에 접근해 train 폴더와 val폴더에 접근해 데이터를 불러옵니다.
# 해당 코드는 dictionary comprehension을 사용한 것.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=0, shuffle=True) for x in ['train', 'val']}


     # 구글 colab 코랩 런타임 끊김 방지
# 방법 1
function ClickConnect(){
    console.log("코랩 연결 끊김 방지");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60 * 1000)

# 방법 2
 - 구글 코랩에서 F12로 개발자 도구창을 열고 Console 선택 후 아래의 코드를 입력한 뒤 엔터를 누르면됩니다.
function ClickConnect() {var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); buttons.forEach(function(btn) { btn.click(); }); console.log("1분마다 자동 재연결"); document.querySelector("colab-toolbar-button#connect").click(); } setInterval(ClickConnect,1000*60);



    # Python PIL, Pillow 라이브러리
 - Python에서 사용가능한 패키지로 PIL 또는 Pillow가 존재합니다.
   PIL은 Python Image Library의 약자로 다양한 기능을 가지고 있습니다.
   그 중에는 물론 이미지의 크기를 조절할 수 있습니다.
   Pillow 패키지에서 가장 중요한 클래스는 이미지를 표현하는 Image 클래스입니다.

pip install Pillow # 설치
from PIL import Image
Image.open(): 기존 이미지 파일을 열 때 사용
Image.new(): 새로운 이미지 파일을 생성할 때 사용
Image.save():  이미지 파일을 저장할 때 사용

    # 이미지 형태 확인 - Ex 1
from PIL import Image
# 이미지 열기
im = Image.open('python.png')
# 이미지 크기 출력
print(im.size)
# 이미지 JPG로 저장
im.save('python.jpg')

    # 이미지 형태 확인 - Ex 2
# 이미지 불러오기 및 확인
img = Image.open('./train/cat.1.jpg')
plt.imshow(img)
plt.show()

    # 이미지 부분 잘라내기
 - 이미지의 일부를 잘라내는 것을 Cropping 이라 부르는데, 이미지 객체에서 crop() 메서드를 사용하여 일부 영역을 잘라내는데,
   crop() 메서드에서 리턴된 이미지는 부분 이미지로서 이를 저장하면 잘라낸 이미지만 저장된다.
   crop()의 파라미터는 (좌, 상, 우, 하) 위치를 갖는 튜플로 지정한다.
from PIL import Image
im = Image.open('python.png')
cropImage = im.crop((100, 100, 150, 150))
cropImage.save('python-crop.jpg')

    # 이미지 회전 및 Resize
 - 이미지를 회전하기 위해서는 이미지 객체에서 rotate(회전각도) 메서드를 호출하면 된다.
   또한, 이미지의 크기를 확대/축소하기 위해서는 이미지 객체에서 resize(크기튜플) 메서드를 호출한다.
from PIL import Image
im = Image.open('python.png')

# 크기를 600x600 으로
img2 = im.resize((600, 600))
img2.save('python-600.jpg')

# 90도 회전
img3 = im.rotate(90)
img3.save('python-rotate.jpg')

    # 이미지 필터링
 - Pillow 패키지는 이미지를 필터링하기 위한 여러 기본적인 필터들을 제공하고 있다.
   이미지 필터를 위해서는 이미지 객체에서 filter(필터종류) 메서드를 호출하면 되는데,
   필터종류는 ImageFilter 모듈을 import 하여 지정한다.
   예를 들어, Blur 이미지를 위해서는 ImageFilter.BLUR 를 사용하고, 윤곽만 표시하기 위해서는 ImageFilter.CONTOUR 를 사용한다.
from PIL import Image, ImageFilter
im = Image.open('python.png')
blurImage = im.filter(ImageFilter.BLUR)
blurImage.save('python-blur.png')



    # glob
 - glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다.
   단, 조건에 정규식을 사용할 수 없으며 엑셀 등에서도 사용할 수 있는 '*'와 '?'같은 와일드카드만을 지원한다.

   # glob - 특정 파일만 출력하기
# 현재 디렉톨에서 확장자가 jpg인 파일만 모아서 출력한다.
import glob
for filename in glob.glob('*.jpg'):
    print(filename)

# 재귀적으로 현재 폴더의 모든 하위폴더까지 탐색하여 확장자가 jpg인 파일을 출력한다
for filename in glob.iglob('**/*.jpg', recursive=True):
    print(filename)

# 예제: GuineaPig 폴더를 재귀적으로 돌며 jpg 파일을 출력
for filename in glob.iglob('GuineaPig/**/*.jpg', recursive=True):
    print(filename)


# '*'는 임의 길이의 모든 문자열을 의미한다.
output = glob.glob('dir/*.txt')
print(output)
['dir\\file1.txt', 'dir\\file101.txt', 'dir\\file102.txt', 'dir\\file2.txt', 'dir\\filea.txt', 'dir\\fileb.txt']

# '?'는 한자리의 문자를 의미한다.
output = glob.glob('dir/file?.*')
print(output)
['dir\\file1.bmp', 'dir\\file1.txt', 'dir\\file2.bmp', 'dir\\file2.txt', 'dir\\filea.txt', 'dir\\fileb.txt']

# recursive=True로 설정하고 '**'를 사용하면 모든 하위 디렉토리까지 탐색한다.
# 기본값은 False이며, 파일이 너무 많을 경우에 사용하면 과도한 cost가 소모된다고 한다.
output = glob.glob('dir/**', recursive=True)
print(output)
['dir\\', 'dir\\file1.bmp', 'dir\\file1.txt', 'dir\\file101.txt', 'dir\\file102.txt', 'dir\\file2.bmp', 'dir\\file2.txt', 'dir\\filea.txt', 'dir\\fileb.txt', 'dir\\subdir', 'dir\\subdir\\subfile1.txt', 'dir\\subdir\\subfile2.txt']













───────────────────────────────────────────────────────────────
