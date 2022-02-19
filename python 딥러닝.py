


# 목차
* Pytorch Basic
* DNN
* CNN
* RNN
* AE
* GAN
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
º nn.L1Loss: 각 원소별 차이의 절댓값을 계산한다.L1
º nn.MSELoss: Mean Squared Error(평균제곱오차) 또는 squared L2 norm을 계산한다.MSE
º nn.CrossEntropyLoss: Cross Entropy Loss를 계산한다. nn.LogSoftmax() and nn.NLLLoss()를 포함한다. weight argument를 지정할 수 있다.CE
º nn.CTCLoss: Connectionist Temporal Classification loss를 계산한다.
º nn.NLLLoss: Negative log likelihood loss를 계산한다.NLL
º nn.PoissonNLLLoss: target이 poission 분포를 가진 경우 Negative log likelihood loss를 계산한다.PNLL
º nn.KLDivLoss: Kullback-Leibler divergence Loss를 계산한다.KLDiv
º nn.BCELoss: Binary Cross Entropy를 계산한다.BCE
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

# pretrained model load
resnet18 = models.resnet18(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
...




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
       self.features = features
       self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
       self.classifier = nn.Sequential(
           nn.Linear(512 * 7 * 7, 4096),
           nn.ReLU(True),
           nn.Linear(4096, 4096),
           nn.ReLU(True),
           nn.Linear(4096, num_classes),
       )
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
    }















---------- RNN ----------


    # torchtext
# 자연어처리(NLP)를 다룰 때 쓸 수 있는 좋은 라이브러리가 있다.
# 이는 자연어처리 데이터셋을 다루는 데 있어서 매우 편리한 기능을 제공한다.
 º 데이터셋 로드
 º 토큰화(Tokenization)
 º 단어장(Vocabulary) 생성
 º Index mapping: 각 단어를 해당하는 인덱스로 매핑
 º 단어 벡터(Word Vector): word embedding을 만들어준다. 0이나 랜덤 값 및 사전학습된 값으로 초기화할 수 있다.
 º Batch 생성 및 (자동) padding 수행





















---------- AE ----------













---------- GAN ----------










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







---------- 기타 설명 ----------

* 데이터셋
* Pytorch Layer의 종류
* MNIST Dataset
* CIFAR10 Dataset
* CIFAR100 Dataset
* FashionMNIST Dataset
* Hymenoptera_data


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



───────────────────────────────────────────────────────────────
