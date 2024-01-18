

    # 목차
----- Array ------
----- Numpy ------
----- Pandas ------
----- Pandas Visualization ------
----- Matplotlib Visualization ------
----- Seaborn Visualization ------
----- Import codes ------
----- 연습문제들 ------










---------- Array ----------

# Vectors (1D tensors)
v = np.array([1,2,3,4,5,6,7,8]) # Vectors (1D tensors)
v.ndim # # the number of axes (dimensions) of the vec 몇차원인지
v.shape # 각 차원별로 몇개가 들어있는지 알려줌
v.size # 전체 array에 있는 원소의 갯수, 차원에 관계없음
v[::-1] # 역순 출력
v[v > 3]  # 조건 인덱싱


# Matrices (2D tensors)
M = np.array([[1,2,3,4],[5,6,7,8]])
M.ndim, M.shape, M.size
M.T # 전치행렬
M.sum(axis=0)
M.mean(axis=1)
M.cumsum(axis=1)  # 누적 합도 가능하다.
M[M[:,1] > 3,:] # 3보다 큰 것이 있는 행을 전부 뽑아라. # 조건 인덱싱


# Cubes (3D tensors)
a3 = np.array([[(1,2),(3,4)], [(5,6),(7,8)]], dtype = float) # 3차원 행렬
a3.ndim, a3.shape, a3.size
	# 리스트끼리는 많은 연산을 할 수 없지만 array로는 다양한 연산이 가능하다.
A = np.array([[1,1], [0,1]])
B = np.array([[2,0], [3,4]]) #일때
A+B, A-B
A*B # 대응하는 원소끼리 곱함
 -> 행렬 내적은 행렬 곱이며, 두 행렬 A와 B의 내적은 np.dot()을 이용해 계산.
A@B # 행렬의 곱을 연산
A.dot(B) = np.dot(A,B) = A@B
dot_product =  np.dot(A, B)
print('행렬 내적 결과: \n', dot_product)
 -> 전치행렬 :  넘파이의 transpose()를 이용해 전치 행렬을 쉽게 구할 수 있습니다.
transpose_mat = np.transpose(A)
print('A의 전치 행렬: \n', transpose_mat)


# Reshaping: chaning the shape of an array
M.ravel() # 일차원으로 펴짐 # returns the array, flattened
M.reshape(8) # (2,4) -> (8) 일차원으로 봐꺼라
M.reshape(4,2) # (2,4) -> (4,2) 로 바꿀 수 있다.
M.reshape(4,-1) # 행은 4개 만들어 놓고 -1은 내가 지정안할테니 너가 알아서 계산해라.
M.reshape(-1,8) # (2,4) -> (1, 8)
np.arange(7,15).reshape(2,4)


# Linear algebra
M @ M.T # 선형대수와 관련된 연산이 가능하다
np.linalg.inv(m) # 역행렬 계산
 = np.round(np.dot(a,np.linalg.inv(a)))
np.linalg.det(m) # matrix determinant
e1,e2 = np.linalg.eig(m) # eigenvalues and eigenvectors
 np.linalg.solve(x, y) #연립방정식 # Solve the system of equations 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8:



    # axis
# axis = 0 -> (5,3) 행렬에서 처음 5라고 생각하면 됨
# axis = 1 -> (5,3) 행렬에서 3이라 생각하면 됨
#  계산: axis=0하면 5개의 원소를 다 더해서 3개가 나옴
#       axis=1하면 3개의 원소씩 더해서 5개의 원소가 리턴됨.


---------- Numpy ----------

# np.arange 메소드
np.arange(10) # array와 range가 합쳐진 것. <-> # 리스트와 비교: list(range(10))
np.arange(1,2,0.1) # 1부터 2미만까지 0.1단위로 쪼개서 담음
		# 출력값: array([1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])


# np.linspace 메소드
np.linspace(1,2,9) # 균등하게 띄워서 9개를 만들어라.
		# 출력값에 1과 2도 포함해서 9개 (끝값, 2는 포함됨 !!)
		# 그래프 그릴 때 주로 사용.


# zeros, ones 메소드
np.zeros((2,3)) # 만드는데 안에 값이 모두 0으로 시작
np.zeros_like(M) # M의 모양을 본따서 0으로 채워라.
np.ones((2,3)) # 원은 제로와 같은데 1로 채워지는 것
np.ones_like(M)


# Sort 메소드
np.sort(a1) # 정렬
np.sort(a1)[::-1] # 역정렬
np.sort(a2, axis=0) # 2차원일때는 행별 또는 열별로 정렬 가능하다
np.sort(a2, axis=1)
np.sort(a2, axis=None) # 일차원으로 펼쳐지면서 정렬이 됨


# Random 패키지
np.random.seed(123) #seed 씨드로 난수고정 -> 난수 생성 코드와 같은 셀안에 존재하여야 함
np.random.rand(10) # 0부터 1사이의 균일 분포에서 난수 matrix array 생성
np.random.randn() # 가우시안 표준 정규분포에서 난수 matrix array 생성
np.random.randint(1, 41, (5,2)) # 5by2행렬에 1이상 41미만 값을 넣어 출력 / 균일 분포의 정수 난수 생성
 = np.random.randint(5, size=[3,4])
np.random.choice(x, len(x), replace=False) # 비복원추출: replace =True하면 복원추출
 			# p라는 인자를 주어 랜덤선택의 확률조정가능
np.random.shuffle() # 기존의 데이터 순서 바꾸기
np.random.uniform(0, 5, 10) # 0~5 사이의 균일분포를 따르는 난수를 생성
np.random.normal(0, 1, 10) # 정규분포를 따르는 난수를 생성
np.random.normal(0, 1, (5,2))


# Stacking 메소드
np.vstack([M, s]) # 위에서 아래로 쌓음, stack arrays vertically
np.hstack([M, s]) # 옆으로 쌓음, stack arrays horizontally


# unique, nunique 메소드
np.unique() # 데이터에서 중복된 값을 제거하고 중복되지 않는 값의 리스트를 출력
np.nunique() # 데이터에서 중복된 값을 제거하고 중복되지 않는 값들의 수를 반환


# np.digitize() 메소드
    # numpy.digitize 함수는 입력한 어레이1의 값이 입력한 어레이2의 어느 값에 해당하는지 인덱스를 반환합니다.
    # label: 0,1,2, ... 순서의 양의 정수 자동
np.digitize(X, bins)
np.digitize(scores, bins= [60,70,80,90]) # 93은 4구간, 78은 2구간, 77은 2구간, ... (60이하의 구간, 90이상의 구간이 더 있다.)


# np.where 메소드
    # np.where()는 numpy에서 제공하는 함수로 excel의 if()함수와 같다.
df['pop2'] = np.where(df['pop'] > 2, df['pop'] + 2, df['pop'] + 1.5)



    # np.clip(a, a_min, a_max, out=None)
a = np.arange(-5, 5)
a
# array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])

np.clip(a, 0, 4)
# array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
a
# array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])

# out=a를 하면 반환 값을 배열 a에 저장할 수 있음
np.clip(a, 0, 4, out=a)
# array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
a
# array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])

# 최소값 기준만 적용해서 간단하게 '0'보다 작은 수는 모두 0으로 바꿈
a = np.arange(-5, 5)
a
# array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])

a.clip(0)
# array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])



    # np.isfinite
# 객체 또는 원소에서 infinity가 아닐 경우 True를 반환
np.isfinite(array객체)



---------- Pandas ----------

# Series
##판다스 Series와 파이썬 list의 차이점
 1. series는 벡터에 대한 연산을 하지만 list는 백터에 대한 연산을 하지 않는다.
 2. 판다스 Series는 문자로 인덱싱을 할 수 있는 반면 파이썬의 list는 문자로 인덱싱이 안된다. Series쪽에 좀 더 Dictionary 같은 동작을 보여준다.
##판다스 Series와 numpy ndarray와의 차이점
 1. ndarray는 숫자를 이용한 인덱싱만 가능한데 Series는 숫자 뿐만 아니라 인덱싱에 문자를 넣을 수도 있다.
 2. ndarray는 null 값에 대한 표현이 어려운 반면 Series는 null값이 NaN이라는 문자를 표시 해 줘서 없는 값에 대한 표시가 좀 더 편리하다.

x = pd.Series([7, 3, 5, 8], index=['서울','대구','부산','광주'])
x[['서울', '대구']] # 문자 인덱싱 가능
x.index # 시리즈의 인덱스와 데이터타입을 반환
x.values # 시리즈의 배열값들 반환



## Read Files
    # csv 파일
	# index_col = 0 으로 첫번째 컬럼을 안불러 올 수 있음. 인덱스를 불러와서 Unnamed:0인 컬럼이 불러와 질 때 사용
    # tab 파일 같은 경우 tab으로 셀을 구분해서 구분자인자로 \t를 넣어주어 불러옴
df = pd.read_csv('train_profiles.csv',encoding ='cp949', index_col = 0) # encoding은 한글폰트 깨지는 것 방지 cp949가 한글을 말함
store_data = pd.read_csv('data_store_trans.tab', sep='\t')
    # read_csv의 다양한 기능
	# header로 칼럼 없을시 알려줌(자동으로 첫위의 값들이 칼럼으로 설정되서)
	# names로 칼럼 설정
	# 특정한 값을 NaN으로 취급하고 싶으면 na_values 인수에 NaN 값으로 취급할 값을 넣는다.
	# parse_dates로 칼럼합침
	# usecols로 나타낼 칼럼만 나타냄
	# dtype로 타입 바꿈
	# nrows로 나타낼 행수 조절
pd.read_csv(fpath, header=None, names=col_names, na_values={'sunspots':[' -1']}, parse_dates=[[0, 1, 2]],
                 usecols=[0,1,2,4,5], dtype={'definite': 'category'}, nrows=10000)
    # 엑셀 파일
    # 콤마가 아닌 다른 ex)스페이스바 로 구분자가 구분이 될때는 table과 sep으로 다른 구분자를 알려줘야 한다

excelFile = pd.read_excel('tttt1.xlsx', sep=' ')
exam2 = pd.read_table('exam2.csv', sep=' ')
    # 엑셀에서 일부 내용을 복사한 후 클립보드를 이용하여 읽을 수 있다.
exam3 = pd.read_clipboard()



## DF Creation
(1). 리스트, 배열로 데이터프레임 만들기
col_names = ['col1', 'col2', 'col3']
list2= [[1, 2, 3],
        [11, 12, 13]]
array = np.array(list2)
    # 2차원 리스트로 만든 데이터프레임
df_list2 = pd.DataFrame(list2, columns=col_names)
    # 2차원 ndarray로 만든 데이터프레임
df_array2 = pd.DataFrame(array, columns=col_names)
    # arange로 만든 데이터 프레임 / 인덱스와 칼럼 설정
df1 = pd.DataFrame(
    np.arange(6).reshape(3, 2),
    index=['a', 'b', 'c'],
    columns=['데이터1', '데이터2'])
df2 = pd.DataFrame(
    5 + np.arange(4).reshape(2, 2),
    index=['a', 'c'],
    columns=['데이터3', '데이터4'])


(2). 딕셔너리로 데이터프레임 만들기
    # Key는 문자열 칼럼명으로 매핑, Value는 리스트 형(또는 ndarray) 칼럼 데이터로 매핑
dict = {'col1':[1,11], 'col2':[2,22], 'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)
    # 인덱스 설정
df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]}, index =['A','B','C'])
    # 인덱스를 두 개 설정하는 방법(멀티 인덱스 설정방법)
df = pd.DataFrame({"a" : [4 ,5, 6], "b" : [7, 8, 9], "c" : [10, 11, 12]}, index = pd.MultiIndex.from_tuples([('d',1),('d',2),('e',2)], names=['n','v']))
    # zip과 딕셔너리를 활용한 데이터프레임 생성방법
df = pd.DataFrame(dict(zip(con,coninfo)))


(3). 임의의 데이터프레임 만들기
    # 가우시안 표준 정규분포에서 난수 matrix array 생성
df = pd.DataFrame(np.random.randn(6,3)) # 6행 x 3열 데이터프레임 생성
    # 인덱스를 날자로 설정, 컬럼 설정, cumsum()을 이용해 누적 값으로 value 설정
df = pd.DataFrame(np.random.randn(100, 3), \
    index=pd.date_range('1/1/2018', periods=100), columns=['A', 'B', 'C']).cumsum
    # 리스트로 칼럼을 만드는 방법
df = pd.DataFrame(np.random.rand(4,3), columns = list('abc'))


(4). 빈 데이터프레임 만들기 / set_index 메소드
train_feature = pd.DataFrame({'CUS_ID':[x for x in range(1,2501)]})
train_feature = train_feature.set_index('CUS_ID') # 데이터프레임 형성할 때 필요시 붙여주기



## DataFrame을 넘파이 배열, 리스트, 딕셔너리로 변환하기
array3 = df_dict.values # 배열
list3 = df_dict.tolist() # 리스트
dict3 = df_dict.to_dict('list') # 딕셔너리
    ## 목록을 리스트타입(list)으로 바꾸기
df.columns.tolist() # 필드명 리스트
df.values.tolist() # 전체 값 리스트(2차원 리스트로 출력)
df.칼럼명.tolist() # 특정 칼럼만 리스트로 출력
df.index.tolist() # 인덱스 -> 리스트로 출력


## reshape
M.reshape(4,2)
M.reshape(4,-1) # 행은 4개를 만들어 놓고 -1은 내가 지정안할테니 너가 알아서 해라.
array.reshape(-1,) # 1차원으로 변환



## 모듈 작성, 저장 및 불러오기
    # 메직 메소드를 이용해 py파일 저장하기
%%writefile mymath.py
def add(a, b):
    return a + b
def sub(a, b):
    return a - b

import mymath
mymath.add(5, 6)
mymath.sub(5, 6)

import mymath as m
m.add(5, 6)

from mymath import * # 모듈의 모든 함수와 변수를 불러오려면 `import` 문 다음에 `*`를 지정한다.
    # 메직 메소드를 이용해 csv파일 저장하기
%%writefile test2.csv
    # 파일 저장 메소드
exam.to_excel('exam.xlsx', index=False)
examf.to_csv('examf.csv', index=False) # index=True로 하면 앞에 Unnamed: 0 으로 인덱스가 새파일에 열로써 추가 됨.
       .to_csv('Demo2.csv', index=False, encoding = 'cp949')

# 오늘날짜
now = datetime.datetime.now()
nowDate = now.strftime('%Y-%m-%d')
nowDate

## 데이터 정보 체크 메소드 / Check Datas
    # 데이터 구조
df.info()
df.ndim # 몇 차원인지
df.size # 구성,차원과 관련없이 전체 원소가 몇개인지
df.shape # (행, 열)의 갯수
df.shape[0] 행의 갯수 = len(df) #행의 갯수
df.shape[1] 열의 갯수
    # 데이터 타입 구조
df.dtypes # 칼럼별로 데이터타입을 알려줌
df.get_dtype_counts() # 칼럼의 데이터 타입별 갯수 알려줌
df.select_dtypes(['number'])
df.select_dtypes(['float'])
df.select_dtypes(['int'])
df.select_dtypes(['object'])
    = data.columns[data.dtypes == object]
    # 데이터 인덱스
df.index
df.index.values
    # 데이터 칼럼
df.columns
df.columns[-1]
    # 데이터프레임의 값들 배열
df.values #인덱스랑 칼럼 제외하고 값만 보여줌



## pandas에서 데이터프레임 보는 옵션 설정하는 방법
pd.set_option('max_columns', 20, 'max_rows', 20)
pd.set_option('display.max_colwidth', -1) # 각 칼럼 width 최대로
pd.set_option('display.max_rows', 500) # rows 500
pd.set_option('display.max_columns', 500) # columns 500
pd.set_option('display.width', 1000) # columns
pd.set_option('display.max_rows', None) # 모든 행 보기



## Data Copy
x = df.copy() # df수정되도 이후에 x는 변경사항 없음
repl_data = string_data.copy() #원본 유지를 위해서


## 최빈값 (mode)
DataFrame.mode(axis=0, numeric_only=False, dropna=True)
 - mode메서드는 대상 행/열의 최빈값을 구하는 메서드입니다.
 - 최빈값이 여러개일 경우 모두 표시합니다.

df.mode(axis=0, numeric_only=False, dropna=True)
axis : {0 : index / 1 : columns} 최빈값을 구할 축 입니다.
numeric_only : True일 경우 숫자, 소수, 부울값만 있는 열에대해서만 연산을 수행합니다.
dropna : 결측치를 계산에서 제외할지 여부입니다. False일 경우 결측치도 계산에 포함됩니다.

print(df.mode(dropna=False))





## Query
df.query('year in [2000,2001] and state =="Ohio"')
df.query('year>2001').filter(['state','year','pop'])
    = df.loc[df.year>2001, 'state':'pop']
df.query('c > 7')
    = df[df.c > 7]
    = df[df['c'] > 7]
df.query('a>b')
    = df[df.a>df.b]
df.query('b != 7')
    = df[df['b'] != 7]



## filter
    # 변수 선택
df.filter(['state','year'])
    # regex 인자 활용 : _가 들어있는 칼럼 불러온다
df.filter(regex='_')
    = df.filter(like='s') # regex='s' 와  같은 기능
    # 끝이 length로 끝나는 칼럼들 불러온다
df.filter(regex='=length$')
    # 앞 시작이 se로 시작하는 칼럼들 불러온다
df.filter(regex='=^se')



## loc & iloc
 -> 명칭 기반 인덱싱과 위치 기반 인덱싱의 차이를 이해하는 것입니다.
    DataFrame의 인덱스나 칼럼명으로 데이터에 접근하는 것은 명칭 기반 인덱싱입니다.
    0부터 시작하는 행, 열의 위치 좌표에만 의존하는 것은 위치 기반 인덱싱입니다.
 -> iloc[]는 위치 기반 인덱싱만 가능합니다. 따라서 행과 열 위치 값으로 정수형 값을 지정해 원하는 데이터를 반환합니다.
 -> loc[]는 명칭 기반 인덱싱만 가능합니다. 따라서 행 위치에 DataFrame 인덱스가 오며, 열 위치에는 칼럼 명을 지정해 원하는 데이터를 반환합니다.
	명칭 기반 인덱싱에서 슬라이싱을 '시작점:종료점'으로 지정할 때 시작점에서 종료점을 포함한 위치에 있는 데이터를 반환합니다.
    # 위치 기반 인덱싱
df.iloc[0,:]
df.iloc[:4]
    = df.loc[:3]
df.iloc[:,[1,2,4]]
df.iloc[-1,:] #마지막 행 뽑기 #iloc 행옵션쪽에 낱개로 들어가도 됨.
    # 명칭 기반 인덱싱
df.loc[2:5,'sepal_width':'petal_width']
df.loc[:2,['state','pop']]
    # loc에 조건식을 넣어서 인덱싱할 수도 있다
df.loc[df.year>2001, 'state':'pop']
df.loc[df['pop']>df['pop'].mean(), ['year','pop']]



## 글자 포함 여부 확인 contains
df[df['공간제목'].str.contains('소설원')]




## Handling Missing Data(null, nan)
    # null 값 찾기. null이면 True 반환
df.isnull()
pd.isnull(df5)
    # not null 값 찾기.
df.notnull()
    = ~df.isnull()
pd.notnull(df5.data1)
    # 칼럼별 결측치의 갯수 세기
df.isnull().sum()
    # 칼럼별 결측치가 아닌 것의 갯수 세기
df.notnull().sum()
    # NaN값인 행 불러오기
data[data['duration_previous'].isna()]['duration_previous']

    # nan값을 0으로 바꿔줌
df.fillna(0)
X.Age = X.Age.fillna(X.Age.mean()) # inplace 인자를 줄 수도 있으나 안 주는 것을 권장함.
    # nan값을 열마다 다르게 바꿀수도 있음
df5.fillna({'data1': 1.5, 'data2': 0.5, 'lkey': 'Y', 'rkey': ''})
    # 열에 하나라도 nan값이 있으면 drop 하여라
    # how가 default로 any임, axis는 default로 0(행)
df.dropna(axis=1, how='any')
df.dropna(axis=0, how='any') # 행에 하나라도 nan값이 있으면 drop 하여라
df.dropna(axis=0, how='all') # 행에 모두 nan값이 있으면 drop 하여라
df5.dropna(how='all').reset_index(drop=True) #사라진 행 빼고 인덱스를 다시 세움



## Drop Columns or Index
    # 칼럼 삭제방법1
df.drop(columns=['Length','Height'])
    # 칼럼 삭제방법2
    # del과 같은 inplace=True는 권장하지 않음
df.drop('eastern', axis=1, inplace=True)



## 특정 컬럼에서 가장 큰 값과 작은 값 가져오기
    # 가장 큰 값 3개 불러온다
    # NaN 값이 있을 경우 제외하고 불러온다(5개중 NaN값 1개 있고 5개 다 불러오면 4개만 불러온다.)
    # 숫자에만 적용가능하다
df.nlargest(3, 'a')
    # 이런식으로 2번째로 큰 값을 인덱싱 할 수 있다.
df.nlargest(2, 'a').iloc[1]
    # 가장 작은 값 2개 불러온다
df.nsmallest(2, 'c')



## Data Sampling
    # 데이터를 0.5퍼센트 만큼 랜덤으로 가져온다
df.sample(frac=0.5)
    # 1로 하면 row를 랜덤으로 가져오기 때문에 row가 랜덤으로 섞인다 !
df.sample(frac= 1)
    # n개 만큼의 row를 랜덤으로 가져온다 !
df.sample(n = 4)
    # replace 인자 활용 : True하면 복원추출, False하면 비복원추출
data.sample(n=3, replace=True)
    # 씨드값 고정
data.sample(n=3, random_state=42)



## isin (안에 있는지 확인 -> 결과가 boolean으로 출력)
df['a'].isin([5])
    = df.a.isin([5,6])
df.year.isin([2000,2001]) # boolean 타입으로서 True, False 여부나옴



## value_counts() 메소드
    # 카테고리같은 데이터를 셀 때 주로 사용한다.
    # 데이터프레임에는 안되고 시리지에만 할 수 있음.
df['species'].value_counts()
    # value counts를 이용해서 데이터 프레임을 재구성해서 사용할 수 있음
pd.DataFrame(df['species'].value_counts())
    # 안에 아무거나 쓰면 카테고리별 비율도 구해낼 수 있다 !!
    # normalize 변수에 값이 들어가게 된 것
tr.groupby(['cust_id'])['season'].value_counts(1)
    # 기본 인자
lc_loans.grade.value_counts(
    normalize=False, # 비율
    sort=True,       # 정렬
    ascending=False, # 기본적으로 내림차순
    bins=None,
    dropna=True,     # 기본적으로 na값을 제외하고 계산됨
)

    # Bins 사용법
 - 정수나 리스트를 bins 인수에 전달합니다
 - 정수를 전달하면, 입력에 따라 값이 동일한 크기의 bin으로 분할됩니다.
 - 리스트를 전달하면, 리스트에 지정된 간격에 따라 값이 binning 됩니다.
exData['score'].value_counts(bins=[70,80,90,100], sort=False)




## unique()
    # unique() 변수의 중복되지 않는 값들을 나열하여 반환
df['species'].unique()
    # 변수의 중복되지 않는 값들의 갯수
df['species'].nunique()



## describe()
    # 숫자가 있는 필드에 대해서만 요약통계를 보여줌
df.describe()
df.describe?
df.describe(percentiles=[.1,.2,.3,.7])  # 더 자세히 1분위 2분위 3분위 7분위로 보여줄 수도 있다.
df.describe().astype('int')
df.describe(include='all')
df.describe(include=np.object)
df.describe(include=np.number)
df.describe(exclude=np.object) = df.describe()



## max, min와 기타 메소드
df.max(axis=1) # 칼럼에서 제일 큰 값
df.max(axis=0) # 행에서 제일 큰 값
df.min(axis=1)
df.min(axis=0)
    # 기타 메소드
median(), var(), std(), quantile([0.25, 0.75])
df.count() # 널값 아닌것만 카운트
df.abs() # 절댓값 구하기
    # 덧셈, 누적 덧셈, 누적 곱셈, 누적 최댓값반환, 누적 최솟값반환
df.cumsum()
df.sum()
df.cumprod()
df.cummax()
df.cummin()



## sort values() 메소드
    # 칼럼을 기준으로 데이터를 정렬, 기본적으로 오름차순
frame.sort_values(by='b')
frame.sort_values(by=['a', 'b'])
frame.sort_values(by='b', ascending=False)
df.sort_values('mpg', ascending=False) # axis 인자도 있어서 축을 조절할 수도 있다.
site_rank = train_c.groupby(['CUS_ID','SITE_NM'])[['SITE_NM_CNT']].count().sort_values(['CUS_ID','SITE_NM_CNT'],ascending=[True, False]).reset_index()



## sort_index() 메소드
    # 인덱스를 기준으로 데이터를 정렬
df.sort_index()
df.sort_index(by='year')
df.sort_index(by='mpg', ascending=False)



## reset_index() 메소드
    # reset_index()는 인덱스를 칼럼으로 끌어올리는 것.
df.reset_index()
    # 시리즈를 데이터프레임화 시키는 역할도 함.
    # reset_index()이후 name인자로 칼럼명 변경이 가능하다
songdf.groupby('user_id')['title'].nunique().reset_index(name = 'newname')
    # reset_index 메서드를 호출할 때 인자 drop=True 로 설정하면 인덱스 열을 보통의 자료열로 올리는 것이 아니라 그냥 버리게 된다.
df.reset_index(drop=True)



## map 메소드
    # map 메소드는 아래와 같이 어떤 값을 치환할 때 유용하게 사용할 수 있다
X.Sex = X.Sex.map({'male':1, 'female': 2})
    # frame.target.replace({1:'Y', 0:'N'}) # replace 메소드도 동일한 기능을 한다.
frame.target = frame.target.map({'Y':1, 'N':0})
    # 변수의 이름을 바꿀 때 rename대신 map을 활용하여 다음과 같이 할 수도 있다
num_name = [i for i in range(len(str_name))]
f.주구매상품대분류 = f.주구매상품대분류.map(dict(zip(str_name, num_name)))



## Renaming Columns 컬럼명 변경
df = df.rename(columns = {'model_year':'year'})
    # inplace 인자를 줄 수도 있음
df.rename(columns = {'pop': '인구', 'state':'주', 'year':'연도'}, inplace=True)
    # 이런식으로 칼럼명 대신으로 할 수도 있음
pop_Seoul.rename(columns={pop_Seoul.columns[0] : '구별',
                          pop_Seoul.columns[1] : '인구수',
                          pop_Seoul.columns[2] : '한국인',
                          pop_Seoul.columns[3] : '외국인',
                          pop_Seoul.columns[4] : '고령자'}, inplace=True)
    # 인덱스 이름 바꿀 수도 있음
df.rename(index = {'old_nm': 'new_nm'), inplace = True)



## Reordering Datas
    # 인덱스 순서 조정
df = df.reindex(index = [3,0,2,1,4])
    # 컬럼 순서 조정
df = df.reindex(columns = ['연도', '주', '인구'])
df = df.reindex(columns=sorted(df.columns))
    # 대가로 두번해서 그냥 칼럼명 나열해도 그대로 나옴
submit5 = submit5[['CUS_ID', 0, 1, 2, 3, 4, 5]]



## Merge Data
	# merge 명령은 두 데이터 프레임의 공통 열 혹은 인덱스를 기준으로 두개의 테이블을 합친다. 이 때 기준이 되는 열, 행의 데이터를 키(key)라고 한다.
	# merge 명령으로 위의 두 데이터프레임 df1, df2 를 합치면 공통 열인 고객번호 열을 기준으로 데이터를 찾아서 합친다. 이 때 기본적으로는 양쪽 데이터프레임에 모두 키가 존재하는 데이터만 보여주는 inner join 방식을 사용한다.
    # on의 default값은 'inner' 공통된 값이 있는 것만 join하여 반환, 중복되지 않는 것들은 NaN처리
pd.merge(df1, df2)
    = pd.merge(df1, df2, on='key')
    = pd.merge(df1, df2, on='key', how='inner')
    # indicator: 값이 어느 데이터프레임에 있었는지 알려준다.
    # outer로 하면 공통되지 않는 열들도 모두 merge하여 출력
pd.merge(df1, df2, how='outer', indicator=True)
    # df1를 다 가져오고 df2중에서 df1과 겹치는 것만 join하여 반환
    # df2에 안겹치는 것은 제외됨
pd.merge(df1, df2, how='left', on='x1')
    # df2를 다 가져오고 df1중에서 df2와 겹치는 것만 join하여 반환
    # df1에서 안겹치는 것은 제외됨
pd.merge(df1, df2, how='right', on='x1')
    # 반대로 키가 되는 기준열의 이름이 두 데이터프레임에서 다르다면 left_on, right_on 인수를 사용하여 기준열을 명시해야 한다.
pd.merge(df3, df4, left_on='lkey', right_on='rkey')



## Concatenate Data
	# concat 명령을 사용하면 기준 열(key column)을 사용하지 않고 단순히 데이터를 연결(concatenate)한다.
	# 기본적으로는 위/아래로 데이터 행을 연결한다. 단순히 두 시리즈나 데이터프레임을 연결하기 때문에 인덱스 값이 중복될 수 있다.
    #반드시 리스트로 묶어주어야 한다.
pd.concat([df1, df5], axis=1)
    # join인자를 사용 : inner(교집합), outer(합집합) , default는 outer
pd.concat([df1, df3], join='inner')
    # ignore_index인자를 주어 인덱스 값을 새로 정렬해준다
pd.concat(df_list, ignore_index=True)
    # 계층적 index 사용하려면 keys 튜플 입력
    # names인자로 index의 이름 부여 가능
pd.concat([s1,s1], keys = ['s1','s2'], names=['Series name', 'Row ID'])



## Drop_Duplitcates 메소드
기본 사용법
df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
subset : 중복값을 검사할 열 입니다. 기본적으로 모든 열을 검사합니다.
keep : {first / last} 중복제거를할때 남길 행입니다. first면 첫값을 남기고 last면 마지막 값을 남깁니다.
inplace : 원본을 변경할지의 여부입니다.
ignore_index : 원래 index를 무시할지 여부입니다. True일 경우 0,1,2, ... , n으로 부여됩니다.

df.drop_duplicates()
data.drop_duplicates(['k1'])
    # inplace인자와 ignore_index 인자도 있다.
df.drop_duplicates(inplace = True)
    # Duplicated and drop_duplicates by default keep the first observed value combination.
df.drop_duplicates(keep = 'first')
    # Passing *keep='last'* will return the last one
df.drop_duplicates(keep = 'last')
data.drop_duplicates(['k1', 'k2'], keep='last')


## 중복 행 찾기, 중복 행 없애기
- 중복된 값이 어떤 것인지 가지고 오기
test_df[test_df.duplicated()]
 
- 중복된 값을 없애고, 테이블을 가지고 올 때 
test_df[~test_df.duplicated()]
출처: https://bramhyun.tistory.com/66 [日日新又日新:티스토리]


## Rank 메소드 : 칼럼 순서 바뀌는 것 없이 칼럼별로 순위의 동률을 처리하는 메서드
frame.rank()
frame.rank(method = 'min')
	'max' : 같은 값을 가지는 그룹을 높은 순위로 매김.
	'min' : 같은 값을 가지는 그룹을 낮은 순위로 매김.
	'average' : (default)같은 값을 가지는 항목의 평균 값을 순위로 삼음.
	'first' : 데이터 내에서 위치에 따라 순위를 매김.



## Groupby 메소드
	# groupby() 함수의 절차를 [split-apply-combine] 이라고 한다.
	# 기준열을 지정하여 특정열로 그룹별로 나누고 - 각 그룹에 통계함수를 적용하고 - 최종적인 통계량이 산출된 것은 통합해서 표시해주기 때문이다.
df[['state','pop']].groupby('state')['pop'].sum() # Series 임
df[['state','pop']].groupby('state')[['pop']].sum() # DataFrame 임
df.groupby(['key1', 'key2']).mean()
df.groupby(['key1', 'key2'], as_index=False).mean()
    = df.groupby(['key1','key2']).mean().reset_index() # as_index=False 또는 reset_index() 쓰는 것 권장
df.groupby(['key1','key2'])[['data2']].mean().reset_index() # 바로 위는 Series, 이 코드는 DataFrame



## Agg 메소드
    # df.groupby('key1').agg('max')는 df.groupby('key1').max()와 같음
    # 근데 agg를 이용해주는 이유는 집계함수를 여러개 쓸 수 있기 때문.
    # 데이터 그룹화의 결과물에 .agg()를 이용해서, 일반적인 통계함수도 적용시킬 수 있다. 이 때, 인자에 "문자열"로 해당 함수를 호출하게 된다.
df.groupby('key1')['data1'].agg(['mean', 'std'])
df.groupby('key1')['data1'].agg([('평균','mean'), ('표준편차','std')])
    # 딕셔너리 형태로 각 열마다 다른 통계함수를 적용시킬 수 있음
    # groupby후에 agg 메소드를 적용할 열을 딕셔너리 형태로 각각 선택
titanic_df.groupby('Pclass').agg({'Age':'max', 'SibSp':'sum', 'Fare':'mean'})
df.groupby('key1').agg({'data1' : 'mean', 'data2' : 'std'}) #데이터1에서 대해서는 평균, 데이터2에 대해서는 표준편차를 계산해라.
tr.groupby(['gds_grp_mclas_nm','gds_grp_nm']).agg({'gds_grp_nm': 'count', 'amount': 'sum'})
tr.groupby('cust_id').agg({
    'goods_id': [('구매상품종류1', lambda x: x.nunique())],
    'gds_grp_nm': [('구매상품종류2', lambda x: x.nunique())],
    'gds_grp_mclas_nm': [('구매상품종류3', lambda x: x.nunique())]
})
    # 한 열에 대해서 여러가지 집계함수를 적용하고 이름을 명명함
tr.groupby('cust_id')['amount'].agg([
    ('총구매액', np.sum),
    ('구매건수', np.size),
    ('평균구매액', lambda x: np.round(np.mean(x))),
    ('최대구매액', np.max)
]).reset_index()



## Apply 메소드
    # 함수를 Series의 각 element에 적용한다.
df['pop2'] = df['pop'].apply(f)
df['sepcies_4'] = df['species'].apply(sampleEX) # sampleEX라는 함수 만들어서 species 칼럼에 apply를 통한 적용 그리고 sepcies_4라는 새로운 칼럼을 추가해줌.
Series(range(1,5)).apply(np.log)
    # lambda 함수를 적용시킬 수도 있음
df['species'].apply(lambda x: x[:3])
df['pop2'] = df['pop'].apply(lambda x: x + 2 if x > 2 else x + 1.5)
cs['거주지역_광역'] = cs.거주지역.apply(lambda x: x.split()[0])
cs['거주지역_기초'] = cs.거주지역.apply(lambda x: x.split()[1])
Series(range(1,5)).apply(lambda x: 1/x)
Series(range(1,5)).apply(lambda x,y: x+y, args=(3,))  # 추가적으로 파라미터를 받고 싶으면 뒤에 args를 써주어 y값에 넣어준다.
					    # args=(3,) 콤마가 없으면 튜플이 아님.
Series(range(1,5)).apply(lambda x,y,z: x+y+z, args=(3,10))
    # 데이터프레임의 모든 원소에 함수를 적용하고 싶을 때
frame.applymap(np.square)



## pivotTable 피벗테이블 메소드
	# aggfunc에 lambda식도 스스로 만들어 대입할 수 있다
pd.pivot_table(tr ,values='pageview' ,index='id', columns='site', aggfunc=sum, fill_value=0).reset_index()
	# 피벗테이블 만들 때 인자에 dropna=True 도 존재한다.
pd.pivot_table(tr ,values='pageview' ,index='id', columns='site', aggfunc=sum, fill_value=0, dropna=True)
    # 피벗테이블 행별 합치기
TIME_table['0005'] = TIME_table.iloc[:,0:6].sum(axis = 1, skipna = 1)



## melt
    # 똑같은 data에 대해서 melt()와 pivot_table() 을 적용할 경우
    # melt()는 ID가 칼럼으로 존재하는 반면에, pivot_table()은 ID가 index로 들어갔습니다.
pd.melt(df)
    # pd.melt(data, id_vars=['id1', 'id2', ...]) 를 사용한 데이터 재구조화
pd.melt(data, id_vars=['cust_ID', 'prd_CD'])
pd.melt(df, id_vars=['A'], value_vars=['B'])
pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
pd.melt(df, value_vars=['A', 'B', 'C'])
    # pd.melt() 의 variable 이름, value 이름 부여하기 : var_name, value_name
pd.melt(data, id_vars=['cust_ID', 'prd_CD'], var_name='pch_CD', value_name='pch_value')



    # unstack 검색
# melt의 반대 느낌
# Groupby와 unstack()을 활용하여 데이터를 핸들링 해보자
# 데이터에서 성별,생존여부에 따른 나이대의 평균을 groupy를 통해 구하면 아래코드와 같다
df.groupby(['Sex','Survived'])['Age'].mean()
# 이렇게 groupby한 결과물도 DataFrame이다.
# 이걸 matrix 형태로 변환시키기 위해 unstack 메서드를 사용한다
# 이렇게 하면 'Sex'와 'Survivied'를 index, column으로 가지는 DataFrame을 얻을 수 있다.
new_df = df.groupby(['Sex','Survived'])['Age'].mean().unstack()



## Binning
	# cut: 실수 값의 경계선을 지정하는 경우
	# The input array to be binned. Must be 1-dimensional.
    # Group a number of more or less continuous values into a smaller number of "bins"
profile['age_group'] = pd.cut(profile.age, 3) # 나이를 3그룹으로 나눔 (bins 설정 없으면 같은 길이(차이)로 나눔)
profile['age_group'] = pd.cut(profile.age, 3, labels= False) # 라벨표시 안하고 0,1,2 으로 세 그룹을 구분
    # bins=[start, end] : (미포함, 포함)
    # bin 구간 대비 작거나 큰 수:
	#    1. bin 첫번째 구간보다 작으면 --> NaN
	#    2. bin 마지막 구간보다 크면 --> NaN
    # label: 사용자 지정 가능(labels option)
    # 반환(return): a list of categories with labels
profile['age_group'] = pd.cut(profile.age, bins=[0, 19, 29, 70]) # 나이를 0~19세, 20~29세, 30~70세의 세 그룹으로 구분: 19세 미만이 아니라 이하라 19세포함임.
profile['age_group'] = pd.cut(profile.age, bins=[0, 19, 29, 70], # 영역을 넘는 값은 NaN으로 처리된다.
				right = False,
				labels=['10대', '20대', '30대이상'])

	# qcut: 갯수가 똑같은 구간으로 나누는 경우
	# qcut 명령은 구간 경계선을 지정하지 않고 데이터 갯수가 같도록 지정한 수의 구간으로 나눈다.
	# 예를 들어 다음 코드는 1000개의 데이터를 4개의 구간으로 나누는데 각 구간은 250개씩의 데이터를 가진다.



## pandas 시간 관련 메소드
    ## date range 메소드 (타입이 datetime인 series를 형성)
pd.Series(pd.date_range('2018-04-16', periods=5))
pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
pd.DataFrame(np.random.randn(1000, 4),
            index=pd.date_range('1/1/2000', periods=1000, columns=list('ABCD'))
dt_index = pandas.date_range(start='20160901', end='20161031')
    # 해당 기간 매주 월요일들만 추출하고 싶을 때
pd.date_range(start='20160901', end='20161031',freq='W-MON')

    ## to_datetime 메소드 (날자시리즈의 타입을 datetime으로 바꾸어줌)
t = pd.to_datetime(s)
    # 타입이 datetime인 인스턴스에서는 year,month,day,weekday를 나타낼 수 있다.
t[0].year
t[0].month
t[0].day
t[0].weekday()
    # 열에 적용시킬 때
t.apply(lambda x: x.weekday())
    # 시리즈의 datetime 인스턴스 받아서 요일을 출력
t.apply(lambda x: ('월', '화', '수', '목', '금', '토', '일')[x.weekday()])
	# 함수 만들경우
def fw(x) :
    return(('월', '화', '수', '목', '금', '토', '일')[x.weekday()])
t.apply(fw) + '요일'

    # 영어로 요일 출력
t.dt.weekday_name
t.dt.weekday_name.str.upper() # 대문자화
t.dt.weekday_name.str.upper().str[:3] # 'MON' 같이 3글자만 출력.

    # 경과일 계산 (Elapsed time)
edays = (t - pd.to_datetime('2018-01-01')) # 2018-01-01부터의 경과일 계산
edays.astype('timedelta64').astype('int')




---------- Pandas Visualization ----------


# Pandas의 시리즈나 데이터프레임은 `plot()`이라는 시각화 메서드를 내장하고 있다.
# `plot()`은 matplotlib를 내부적으로 임포트하여 사용한다.
# 자세한 내용은 다음 Pandas 문서를 참조한다.
# * https://pandas.pydata.org/pandas-docs/stable/visualization.html


    # plot()
# plot안에 설정 가능 기능들:
# kind='barh', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# kind 인수를 바꾸면 여러가지 플롯을 그릴 수 있다.
* `bar` 막대
* `pie` 원형
* `hist` 히스토그램
* `kde` 밀도
* `box` 박스
* `scatter` 스케터
* `area`
* `hexbin`


    # Default: Line Graph
# plot 공통 인자
# kind='', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# 그래프는 판다스로 그리고, 상세설정은 plt으로 설정
df.plot(figsize=(7,7))
plt.title("Pandas의 Plot")
plt.xlabel("시간")
plt.ylabel("Data")
plt.show()

# subplots 파라미터를 True로 설정하여
# DataFrame의 각 필드(Series)를 서로 다른 플롯(subplot)에 도식할 수 있다.
df.plot(subplots=True,figsize=(6, 6))
plt.show()

# x축 값과 y축 값을 설정
df3.plot(x='A', y='B')



    # Bar
# plot 공통 인자
# kind='bar', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# kind 인수에 문자열을 쓰는 대신 'plot.bar()''처럼 직접 메서드로 사용할 수도 있다.
# 하지만 불편한 점이 많아서 그냥 kind 치는 것을 권장.
df.plot(kind='barh', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False)

# rot밑에 0123~19 숫자 각도를 조정해줌
iris.sepal_length[:20].plot(kind='bar', rot=0)
iris[:5].plot(kind='bar', rot=0)

# 가로 방향으로 바 차트를 그리려면 'barh' 명령을 사용한다.
iris[:5].plot(kind='barh', rot=0)
    = iris[:5].plot.barh(rot=0)
plt.axhline(0, color='k')

# 누적 막대그래프
# 'stacked' 파라미터를 True로 설정하면, 누적막대그래프를 그릴 수 있다.
df2.plot(kind='bar', stacked=True, legend=True, figsize=(7,7))

# legend를 False로 하면 범례가 사라짐.
iris[:5].plot(kind='barh', stacked=True, legend=True)

# 전치 행렬도 적용가능하다
# 행렬이 바뀜
df.T.plot.bar(rot=0)

# axes에 플롯을 설정하였을 때
ax = status_tab.plot.bar()
ax.set_xlabel('Status')
ax.set_ylabel('Frequency')



    # Pie
# plot 공통 인자
# kind='pie', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# 타이타닉 데이터 예시
# value_counts() 해주어야함.
df = titanic.pclass.value_counts()

# autopct로 파이 안에 수치를 나타냄
df.plot(kind='pie',autopct='%.2f%%')
plt.title("선실별 승객 수 비율")
# 이 코드를 안쳐주면 그래프가 타원으로 넓적하게 타원형으로 나옴
plt.axis('equal')
plt.show()

# sns.countplot과 같이 나타낸 pie plot - Ex 1
fig, axes = plt.subplots(1, 2, figsize=(10,5))
sns.countplot(data=df, y=df['score'], palette='Blues', ax=axes[0])
df['sentiment_label'].value_counts().plot(kind='pie',autopct='%.2f%%', explode=(0,0.03),
    labels=['부정','긍정'], legend=True, colors=['lightcoral', 'lightblue'], textprops={'fontsize': 13}, ax=axes[1])
plt.tight_layout()
plt.show()

# pie plot - Ex 2
df.plot(kind='pie', subplots=True, figsize=(9,7))
series.plot.pie(labels=['AA', 'BB', 'CC', 'DD'], colors=['r', 'g', 'b', 'c'],
                autopct='%.2f', fontsize=20, figsize=(6, 6))



    # Histogram
# plot 공통 인자
# kind='hist', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# bins의 갯수가 중요함, 몇 그룹을 할꺼냐.
# edgecolor밖의 색깔, color 안의 색깔, alpha는 투명도
df4.plot(kind='hist', bins=20, alpha=0.5, edgecolor='w', color='darkred', figsize=(6,6))
df4[['a','b','c']].plot(kind='hist', color='k', alpha=0.5, bins=10)
data.hist(by=np.random.randint(0, 4, 1000), figsize=(8, 6))

# stacked = True, 다수의 데이터를 쌓아 히스토그램을 표현
df4.plot(kind='hist', stacked=True, bins=20, figsize=(6,6))

# cumulative = True, 누적 히스토그램, 마지막 값은 모든 데이터세트를 가지게 됨
# orientation = horizontal, barh(가로 타입 막대그래프)가 그려짐 (기본은 vertical)
df4['a'].plot(kind='hist', orientation='horizontal', cumulative=True, figsize=(6,6))

# 앞의 값과의 차분을 구해준다.
df4['a'].diff()
df4['a'].plot(kind='hist', figsize=(6,6))

# density = True, 확률밀도를 형성켜서 정규화, 히스토그램 값들의 면적이 1이 됨
# 요소의 수로 나누어 정규화되는 것이 아니라 막대(bin)의 너비를 이용하여 정규화되는 것에 유의
# stacked 역시 True일 경우, 히스토그램의 합은 1로 정규화된다.



    # KDE
# plot 공통 인자
# kind='kde', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
ser.plot(kind='kde', figsize=(6,6))

# kde와 density 그래프는 똑같다
ser.plot(kind='density', figsize=(6,6))

# diagonal='kde' 커널밀도함수를 의미
# 커널밀도 함수 = 확률밀도 함수
# 대각선에는 커널밀도 함수를 통해서 정규분포를 그리게 됨, 나머지는 산포도를 그린다.
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(9, 9), diagonal='kde')



    # Box
# plot 공통 인자
# kind='box', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False

    # Box plot 설명
# • 상자그림은 탐색적 데이터분석(EDA: exploratory data analysis)에서 유용한 분석도구이다.
# 이는 다음의 다섯숫자 요약(five-number summary)에 의해 만들어진다.: xmin, Q1, Q2, Q3, xmax
# • 상자의 양쪽 끝에서 수평으로 길게 뻗은 선을 보통 수염(whiskers)이라고 부르며,
# 수염의 길이는 데이터 분포의 양쪽 고리가 긴 정도, 즉 분포의 정도를 나타낸다.
# • 또한 상자그림은 데이터의 중심(center)과 변동성(variability)을 보여준다

# notch = True, 가운데 상자를 중앙값 부근에서 V자 형태로 골이 패이게 그림
df.plot(kind='box', grid=True, figsize=(6,6))

# box plot에 그래프 기능별 색상을 넣어줌
# sym 이상치의 색상과 모양을 설정하는 인자
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}
df.plot(kind='box', sym='r+', color = color)

# vert = False, 상자그림을 가로로
df.plot.box(vert=False, positions=[1, 4, 5, 6, 8])

# 박스플롯에 대해서는 추가적인 기능을 가진 `boxplot` 명령이 별도로 있다.
# * `boxplot`: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html
bp = df.boxplot(by='X')
iris.boxplot(by='species')

# tight_layout() adjusts spacing between subplots to minimize the overlaps
plt.tight_layout(pad=3, h_pad=1)



    # Scatter
# plot 공통 인자
# kind='box', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# s는 점의 사이즈
iris.plot(kind='scatter', x='sepal_length', y='sepal_width', s=iris.petal_length*10, figsize=(8,6))
# grid 설정 가능
df.plot(kind='scatter', x='a', y='b', s=50, grid=True)
# label 설정 가능
ax = df.plot(kind='scatter', x='a', y='b', color='DarkRed', label='Group1')
df.plot(kind='scatter', x='c', y='d', color='DarkBlue', label= 'Group2')
df.plot(kind='scatter', x='c', y='d', color='DarkBlue', label= 'Group2', grid=True, ax=ax)

# 함수 이용하여 컬러 설정
color = np.where(iris.species == 'setosa', 'r', np.where(iris.species == 'versicolor', 'g', 'b'))
iris.plot.scatter(x='sepal_length', y='sepal_width', c=color)



    # Area
# plot 공통 인자
# kind='box', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
# area플롯은 stacked=True가 기본값이라 값을 누적하여 보여준다.
# 예를 들어 3개의 칼럼이 있으면 첫번째부터 세번째까지 그래프를 누적 + 색을 채워서 보여줌
df.plot(kind='area', figsize=(8,4))
df.plot(kind='area', stacked=False, grid=True, figsize=(8,4))

# stacked=False, 알파값이 자동으로 조정되어서 값들이 누적되지 않고 보여짐
df.plot(kind='area', stacked=False)

# x, y값 설정가능
ax = df.plot(kind='area', x='day')



    # Hexbin
# plot 공통 인자
# kind='box', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False
#gridsize로 격자의 갯수를 조정해 준다
df.plot.hexbin(x='a', y='b', gridsize=15, figsize=(8,6))

# 데이터가 클 때 scatter plot (산점도)의 단점을 보완할 수 있다.
# np.mean, np.max, np.min, np.median
df.plot(kind='hexbin', x='a',y='b', C='z', reduce_C_function=np.max, gridsize=15, figsize=(8,6))





---------- Matplotlib Visualization ----------

    # library
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
%matplotlib inline


    # 한글 출력을 위한 설정(1)
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)


    # 한글 출력을 위한 설정(2)
import platform
your_os = platform.system()
if your_os == 'Linux':
    rc('font', family='NanumGothic')
elif your_os == 'Windows':
    ttf = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)
elif your_os == 'Darwin':
    rc('font', family='AppleGothic')
rc('axes', unicode_minus=False)  # 그래프에서 마이너스 기호 깨지지 않도록 설정


    # 폰트 설정
# 폰트를 설정하는 방법은 크게 두가지 이다.
# * rc parameter 설정으로 이후의 그림 전체에 적용
# * 인수를 사용하여 개별 텍스트 관련 명령에만 적용
# 개별적으로 폰트를 적용하고 싶을 때는 다음과 같이 폰트 패밀리(폰트명), 색상, 크기를 정하여
#           플롯 명령에 dictionary 또는 unpack 형태의 인수로 넣는다.

import matplotlib.font_manager as fm
# ttf 폰트 전체개수
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(len(font_list))
# OSX의 설치된 폰트를 가져오는 함수
font_list_mac = fm.OSXInstalledFonts()
print(len(font_list_mac))

# 시스템 폰트에서 읽어온 리스트에서 상위 10개만 출력
font_list[:10]

    # 폰트 설정 - Ex 1
# 개별적 폰트 설정
# family는 폰트명을 말함
font1 = {'family' : 'NanumMyeongjo', 'color':  'black', 'size': 24}
font2 = {'family': 'NanumBarunpen',
         'color':  'darkred', 'weight': 'bold', 'size': 18}
font3 = {'family': 'NanumBarunGothic',
         'color':  'blue', 'weight': 'light', 'size': 12}

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)
plt.plot(x, y, 'k')
plt.title('한글 제목', font1) # 방법은 어떤거를 써도 동일함.
plt.xlabel('엑스 축', font2)
plt.ylabel('와이 축', **font3)
plt.subplots_adjust()
plt.show()



    # 그림의 구조
# Matplotlib가 그리는 그림은 Figure 객체, Axes 객체, Axis 객체 등으로 구성된다.
# Figure 객체는 한 개 이상의 Axes 객체를 포함하고 Axes 객체는 다시 두 개 이상의 Axis 객체로 포함한다.
# Figure는 그림이 그려지는 캔버스나 종이를 뜻하고 Axes는 하나의 플롯, 그리고 Axis는 가로축이나 세로축 등의 축을 뜻한다.
# 다음 그림은 이 구조를 설명하고 있다.
	# <img src="https://datascienceschool.net/upfiles/4e20efe6352e4f4fac65c26cb660f522.png" style="width: 80%">
	# <img src="https://matplotlib.org/_images/anatomy.png" style="width: 70%">

    # Figure 객체
# 모든 그림은 Figure 객체. 정식으로는  `Matplotlib.figure.Figure` 클래스 객체에 포함되어 있다. 내부 플롯(inline plot)이 아닌 경우에는 하나의 Figure는 하나의 아이디 숫자와 윈도우(Window)를 가진다.
# 주피터 노트북에서는 윈도우 객체가 생성되지 않지만 파이썬을 독립 실행하는 경우에는 하나의 Figure당 하나의 윈도우를 별도로 가진다. Figure 객체에 대한 자세한 설명은 다음 웹사이트를 참조한다.
# * http://Matplotlib.org/api/figure_api.html#Matplotlib.figure.Figure
# 원래 Figure를 생성하려면 `figure` 명령을 사용하여 그 반환값으로 Figure 객체를 얻어야 한다. 그러나 일반적인 `plot` 명령 등을 실행하면 자동으로
# Figure를 생성해주기 때문에 일반적으로는 `figure` 명령을 잘 사용하지 않는다. `figure` 명령을 명시적으로 사용하는 경우는 여러개의 윈도우를 동시에 띄워야 하거나(line plot이 아닌 경우),
# Jupyter 노트북 등에서(line plot의 경우) 그림의 크기를 설정하고 싶을 때이다. 그림의 크기는 figsize 인수로 설정한다.

    # Axes 객체와 subplot 명령
# 때로는 하나의 윈도우(Figure)안에 여러개의 플롯을 배열 형태로 보여야하는 경우도 있다. Figure안에 있는 각각의 플롯은 Axes라고 불리는 객체에 속한다.
# Axes 객체에 대한 자세한 설명은 다음 웹사이트를 참조한다
# * http://Matplotlib.org/api/axes_api.html#Matplotlib.axes.Axes
# Figure 안에 Axes를 생성하려면 원래 subplot 명령을 사용하여 명시적으로 Axes 객체를 얻어야한다. 그러나 plot 명령을 바로 사용해도 자동으로 Axes를 생성해 준다.

    # colors
# https://matplotlib.org/stable/tutorials/colors/colormaps.html



    # 기본 설정
plt.title('') # 제목
plt.xlabel('') # x축 라벨
plt.ylabel('') # y 축 라벨
	# 플롯의 x축 위치와 y축 위치에는 각각 그 데이터가 의미하는 바를 표시하기 위해 라벨(label)를 추가할 수 있다.
	# 라벨을 붙이려면 `xlabel`. `ylabel` 명령을 사용한다.
	# 또 플롯의 위에는 `title` 명령으로 제목(title)을 붙일 수 있다.
plt.xlim(,) # x축 범위 설정: 그림의 범위가 되는 x축, y축의 최소값과 최대값을 지정한다
plt.ylim(,) # y축 범위 설정
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi]) # x축에 틱을 나타내 준다: 틱의 위치나 틱 라벨은 Matplotlib가 자동으로 정해주지만 만약 수동으로 설정하고 싶다면 xticks 명령이나 yticks 명령을 사용한다.
plt.yticks([-1,0,1]) # y축에 틱을 나타내 준다 # 틱을 없애려면 비스트안을 비우면 된다.
plt.xticks(x, xticklabel) #x축에 숫자인 틱을 xticklabel안에 있는 내용으로 라벨을 넣어 출력해줌. 리스트 변수를 이용한 것 뿐 밑에 코드와 유사
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"]) # 틱 -1, 0, 1에 라벨을 넣어 대신 나타낸다.
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$']) # 틱 라벨 문자열에는 $$ 사이에 LaTeX 수학 문자식을 넣을 수도 있다.
plt.grid(True) # 틱 위치를 잘 보여주기 위해 그림 중간에 그리드 선을 나타내려면 grid(True)를 사용한다
plt.legend(loc=0) # 라벨을 보여준다. 0은 가장 좋은 자리를 자동지정해서 보여주는 것
plt.legend(loc='upper right', title='군집')
plt.show()
plt.tight_layout() # tight_layout 명령을 실행하면 플롯간의 간격을 자동으로 맞춰준다.



    # plt.plot
plt.plot([1,4,9,16])
plt.plot([10,20,30,40],[1,4,9,16])
    # 첫 글자는 컬러의 색깔b 두번째는 마커의 종류o, 세번째는 선의 종류 -.
plt.plot([10,20,30,40],[1,4,9,16],'bo-.')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    #라벨 설정
plt.plot(x, y1, 'r', label='$e^x$')
plt.plot(x, y2, 'b--', label='$2^x$')
    # 여러개의 선을 그리기
	# 라인 플롯에서 선을 하나가 아니라 여러개를 그리고 싶은 경우에는 , x데이터, y데이터, 스타일 문자열을 반복하여 인수로 넘긴다.
	# 이 경우에는 하나의 선을 그릴 때처럼 x데이터나 스타일 문자열을 생략할 수 없다.
plt.plot(t, t, 'r--', t, 0.5 * t**2, 'bs:', t, 0.2 * t**3, 'g^-')
    # 겹쳐서 그리기: 하나의 plot 명령이 아니라 복수의 plot 명령을 하나의 그림에 겹쳐서 그릴 수도 있다.
plt.plot([1, 4, 9, 16],
         c="b", lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
plt.plot([9, 16, 4, 1],
         c="k", lw=3, ls=":", marker="s", ms=10, mec="m", mew=5, mfc="c")
    # scatter 플롯과 같이 그리는 예
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')



	# 스갈 타일 문자열은 색깔(color), 마커(marker), 선 종류(line style)의 순서로 지정한다. 만약 이 중 일부가 생략되면 디폴트 값이 적용된다.
	# 자주 사용되는 색깔은 한글자 약자를 사용할 수 있으며 약자는 아래 표에 정리하였다. (다른 색들은 색 이름으로 지정)

	# 색(color)
# color를 지정할 때 다양한 방법이 있겠지만 다음과 같이 np.where를 활용하는 방법도 있음
# color = np.where(iris.species == 'setosa', 'r', np.where(iris.species == 'versicolor', 'g', 'b'))
	| 문자열 | 약자 |
	|-|-|
	| `blue` | `b` |
	| `green` | `g` |
	| `red` | `r` |
	| `cyan` | `c` |
	| `magenta` | `m` |
	| `yellow` | `y` |
	| `black` | `k` |
	| `white` | `w` |   #  전체 색깔 목록(named color)은 다음과 같다.
	<img align="left" src="https://matplotlib.org/_images/named_colors.png" alt="matplotlib color">

# 색깔 모음집
from matplotlib import cm
colors = cm.get_cmap('Set2')(np.arange(4))


	# 마커(marker): 데이터 위치를 나타내는 기호를 마커라고 한다. 마커의 종류는 다음과 같다.
	| 마커 문자열 | 의미 |
	|-|-|
	| `.` | point marker  |
	| `,` | pixel marker |
	| `o` | circle marker |
	| `v` | triangle_down marker |
	| `^` | triangle_up marker |
	| `<` | triangle_left marker |
	| `>` | triangle_right marker |
	| `1` | tri_down marker |
	| `2` | tri_up marker |
	| `3` | tri_left marker |
	| `4` | tri_right marker |
	| `s` | square marker |
	| `p` | pentagon marker |
	| `*` | star marker |
	| `h` | hexagon1 marker |
	| `H` | hexagon2 marker |
	| `+` | plus marker |
	| `x` | x marker |
	| `D` | diamond marker |
	| `d` | thin_diamond marker |

	# 선 스타일(linestyle)
# 선 스타일에는 실선(solid), 대시선(dashed), 점선(dotted), 대시-점선(dash-dit) 이 있다. 지정 문자열은 다음과 같다.
	| 선 스타일 문자열 | 의미 |
	|-|-|
	| `-` |  solid line style
	| `--` |  dashed line style
	| `-.` |  dash-dot line style
	| `:` |  dotted line style

	# 기타 스타일
# 라인 플롯에서는 앞서 설명한 세 가지 스타일 이외에도 여러가지 스타일을 지정할 수 있지만 이 경우에는 인수 이름을 정확하게 지정해야 한다.
# 사용할 수 있는 스타일 인수의 목록은 `Matplotlib.lines.Line2D` 클래스에 대한 다음 웹사이트를 참조한다.
# * http://Matplotlib.org/api/lines_api.html#Matplotlib.lines.Line2D
# 라인 플롯에서 자주 사용되는 기타 스타일은 다음과 같다.
	| 스타일 문자열 | 약자 | 의미 |
	|-|-|-|
	| `color` | `c`  | 선 색깔 |
	| `linewidth` | `lw` | 선 굵기 |
	| `linestyle` | `ls` | 선 스타일 |
	| `marker` |   | 마커 종류 |
	| `markersize` | `ms`  | 마커 크기 |
	| `markeredgecolor` | `mec`   |	마커 선 색깔 |
	| `markeredgewidth` | `mew`   |	마커 선 굵기 |
	| `markerfacecolor` | `mfc`   |	마커 내부 색깔 |

    # 기타 스타일 적용 예시
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], c="b",
         lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
plt.title("스타일 적용 예")
plt.show()




	# 범례
# 여러개의 라인 플롯을 동시에 그리는 경우에는 각 선이 무슨 자료를 표시하는지를 보여주기 위해 legend 명령으로 범례(legend)를 추가할 수 있다.
# 범례의 위치는 자동으로 정해지지만 수동으로 설정하고 싶으면 loc 인수를 사용한다. 인수에는 문자열 혹은 숫자가 들어가며 가능한 코드는 다음과 같다.
| loc 문자열 | 숫자 |
|-|-|
| `best` |  0 |
| `upper right` |  1 |
| `upper left` |  2 |
| `lower left` |  3 |
| `lower right` |  4 |
| `right` |  5 |
| `center left` |  6 |
| `center right` |  7 |
| `lower center` |  8 |
| `upper center` |  9 |
| `center` |  10 |



    # Subplot
# `subplot` 명령은 그리드(grid)형태의 Axes 객체들을 생성하는데 Figure가 행렬(Matrix)이고 Axes가 행렬의 원소라고 생각하면 된다.
# 예를 들어 위와 아래 두 개의 플롯이 있는 경우 행이 2이고 열이 1인 2x1 행렬이다.
# `subplot` 명령은 세개의 인수를 가지는데 처음 두 개의 원소가 전체 그리드 행렬의 모양을 지시하는 두 숫자이고 세번째 인수가 네 개 중 어느것인지를 의미하는 숫자이다.
# 따라서 위/아래 두개의 플롯을 하나의 Figure 안에 그리려면 다음처럼 명령을 실행해야 한다.
# 여기에서  첫번째 플롯을 가리키는 숫자가 0이 나리라 1임을 주의하라. 숫자 인덱싱은 파이썬이 아닌 Matlab 관행을 따르기 때문이다.
# EX codes: subplot(2, 1, 1) # 윗부분에 그릴 플롯 명령 실행
# 	 	    subplot(2, 1, 2) # 아랫부분에 그릴 플롯 명령 실행
#       	subplot의 인수는 (2,2,1)를 줄여서 221라는 하나의 숫자로 표시할 수도 있다.
# subplot이랑 subplots 명령이랑 다르다
# subplots 명령으로는 복수의 Axes 객체를 동시에 생성할 수 있다. 이때는 2차원 ndarray 형태로 Axes 객체가 반환된다.

	# subplot - EX 1
# x1, x2, y1, y2 설정
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)
# figure와 axes설정 및 그래프  그리기
ax1 = plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

ax2 = plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.tight_layout()
plt.show()

	# subplot - EX 2: 만약 2x2 형태의 네 개의 플롯이라면 다음과 같이 그린다
    # 이 때 subplot 의 인수는 (2,2,1)를 줄여서 221 라는 하나의 숫자로 표시할 수도 있다. Axes의 위치는 위에서 부터 아래로, 왼쪽에서 오른쪽으로 카운트한다.
np.random.seed(0)

plt.subplot(221)
plt.plot(np.random.rand(5))
plt.title('axes 1')

plt.subplot(222)
plt.plot(np.random.rand(5))
plt.title('axes 2')

plt.subplot(223)
plt.plot(np.random.rand(5))
plt.title('axes 3')

plt.subplot(224)
plt.plot(np.random.rand(5))
plt.title('axes 4')

plt.tight_layout()
plt.show()

	# subplots의 사용 - EX 3
# 변수명은 맘대로 해도 되는데 통상적으로 fig나 axe를 사용한다.
# subplots 에는 그래프별로 x축과 y축을 공유하는 sharex, sharey 인자가 존재한다
# 총 네개의 그림을 2행2열로 그리겠다.
np.random.seed(0)
fig, axes = plt.subplots(2, 2, figsize=(6,6))
axes[0, 0].plot(np.random.rand(5)) # 위 왼쪽
axes[0, 0].set_title('axes 1')
axes[0, 1].plot(np.random.rand(5)) # 위 오른쪽
axes[0, 1].set_title('axes 2')
axes[1, 0].plot(np.random.rand(5)) # 아래 왼쪽
axes[1, 0].set_title("axes 3")
axes[1, 1].plot(np.random.rand(5)) # 아래 오른쪽
axes[1, 1].set_title("axes 4")

plt.tight_layout()
plt.show()

    # subplots - EX 4
fig, axes = plt.subplots(1,2, figsize=(12,6))
axes[0].bar(status_tab.index, status_tab, color=colors)
axes[0].set_xlabel('Status')
axes[0].set_ylabel('Frequency')

axes[1].pie(status_tab, labels = status_tab.index, autopct = "%.1f%%", colors=colors)
plt.show()



    # twinx
# 여러가지 플롯을 하나의 Axes 객체에 표시할 때 y값의 크기가 달라서 표시하기 힘든 경우가 있다.
# 이 때는 다음처럼 `twinx` 명령으로 대해 복수의 y 축을 가진 플롯을 만들수도 있다.
# `twinx` 명령은 x 축을 공유하는 새로운 Axes 객체를 만든다.
# subplots 에도 x축과 y축을 공유하는 sharex, sharey 인자가 존재한다

    # twinx - EX1
fig, ax0 = plt.subplots()
ax1 = ax0.twinx() # twinx 설정: x축 공유

ax0.plot([10, 5, 2, 9, 7], 'r-', label='y0')
ax0.set_title('2개의 y축을 한 figure에서 사용하기')
ax0.set_xlabel('공유되는 x축')
ax0.set_ylabel('y0')
ax0.set_ylim(0, 12)

ax1.plot([100, 200, 220, 180, 120], 'g:', label='y1')
ax1.set_ylabel('y1')

plt.show()

    # twinx - EX2
fig, ax0 = plt.subplots(figsize=(16,8))
ax1 = ax0.twinx()

ax0.set_title('월간 평균 지진규모와 빈도')
ax0.set_ylabel('지진규모')
ax1.set_ylabel('빈도')

line1, = ax0.plot(eq1_summary['eq_규모_달평균'], color='black')
line2, = ax1.plot(eq1_summary['eq_빈도_달평균'],'r--')

plt.show()



    # 바 차트, bar
# x 데이터가 카테고리 값인 경우에는 `bar` 명령과 `barh` 명령으로 바 차트(bar chart) 시각화를 할 수 있다.
# 가로 방향으로 바 차트를 그리려면 `barh`명령을 사용한다.
# 자세한 내용은 다음 웹 사이트를 참조한다.
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh

    # bar - Ex 1
y = [2, 3, 1]
x = np.arange(len(y))
xticklabel = ['가', '나', '다']
plt.title("Bar Chart")
plt.bar(x, y)
plt.xticks(x, xticklabel) #x축에 숫자인 틱을 xticklabel을 통해 가나다로 바꿈
plt.yticks(sorted(y))
plt.xlabel("가나다")
plt.ylabel("빈도 수")
plt.show()

    # bar - Ex 2
# `xerr` 인수나 `yerr` 인수를 지정하면 에러 바(error bar)를 추가할 수 있다.
# x의 표준편자를 보여줌, seaborn은 자동으로 나오는데 이거는 스스로 정해줘야 되서 좀 불편.
# `alpha`는 투명도를 지정한다. 0이면 완전 투명, 1이면 완전 불투명이다.
np.random.seed(0)
people = ['몽룡', '춘향', '방자', '향단']
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.title("Barh Chart")
plt.barh(y_pos, performance, xerr=error, alpha=0.7)
plt.yticks(y_pos, people)
plt.xlabel('x 라벨')
plt.show()



    # 파이 차트, pie
# 카테고리 별 값의 상대적인 비교를 해야 할 때는 `pie` 명령으로 파이 차트(pie chart)를 그릴 수 있다.
# 파이 차트를 그릴 때는 윈의 형태를 유지할 수 있도록 다음 명령을 실행해야 한다.
# plt.axis('equal')
# 자세한 내용은 다음을 참조한다.
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pie
labels = ['개구리', '돼지', '개', '통나무']
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0) # explode는 파이에서 설정한 부분만 설정한 값만큼 띄워준다
plt.title("Pie Chart")
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90) # autopct는 파이안에 수치를 나타내어 준다
plt.axis('equal')
plt.show()

    # pie 도넛모양1 - Ex 1
# wedgeprops 인자 이용
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
axes[0].pie(before_manless.비율, labels=before_manless.단어, autopct='%.1f%%',
    startangle=347, explode=[0.005, 0.005, 0.005, 0.005], wedgeprops=wedgeprops, colors=colors)

    # pie 도넛모양2 - Ex 2
# Circle를 이용해 가운데를 다른색으로 채워서 도넛 모양으로 그리는 방법
circle = plt.Circle((0, 0), 0.7, color='white')



    # 히스토그램, Histogram
# `hist` 명령은 `bins` 인수로 데이터를 집계할 구간 정보를 받는다.
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
np.random.seed(0)
x = np.random.randn(1000)
plt.title("Histogram")
plt.hist(x, bins=10, edgecolor='w')
plt.show()



    # 스캐터 플롯, Scatter
# 두 개의 실수 데이터 집합의 상관관계를 살펴보려면 `scatter` 명령으로 스캐터 플롯을 그린다.
# 스캐터 플롯의 점 하나의 위치는 데이터 하나의 x, y 값이다.
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.title("Scatter Plot")
plt.scatter(X, Y)
plt.show()

    # scatter - Ex 1
	# 버블차트
# 데이터가 2차원이 아니라 3차원 혹은 4차원인 경우에는 점 하나의 크기 혹은 색깔을 이용하여 다른 데이터 값을 나타낼 수도 있다.
# 이런 차트를 버블 차트(bubble chart)라고 한다. 크기는 `s` 인수로 색깔은 `c` 인수로 지정한다.
N = 30
np.random.seed(0)
x = np.random.rand(N)
y1 = np.random.rand(N)
y2 = np.random.rand(N)
y3 = np.pi * (15 * np.random.rand(N))**2
plt.title("Bubble Chart")
plt.scatter(x, y1, c=y2, s=y3) #c는 컬러, s는 사이즈
plt.show()

    # scatter - Ex 2
# 아래 국가별 데이터를 이용하여 그림과 동일한 Bubble Chart를 도식하시오.
# 단 버블의 색깔은 서로 상이하도록 하고 버블의 크기는 GDP_size로 설정할 것.
# <img src="http://drive.google.com/uc?export=view&id=1bgy_Og_K9BNaVAlyYArhywjB5qzNf7Om" style="width: 50%">
# <img src="http://drive.google.com/uc?IJC0T5Q0RPwqpG2z52zgMip" style="width: 50%">
ctn = ['Korea','China','Japan','USA','Germany']
pop = [0.4,0.6,-0.1,0.8,-0.2]
exp = [-1.4,0.8,-3.4,3.9,9.4]
gdp = [330,2096,1184,4254,961]
plt.scatter(pop, exp, c=np.random.rand(5), s=gdp)
plt.grid()
plt.xlim(-0.4,1)
plt.ylim(-6,12)
plt.xlabel('Population_growth')
plt.ylabel('Export_increase')
# 그래프에 텍스트를 넣는 방법 (plt.text 메소드, 인자: x위치, y위치, 넣을 텍스트)
for i in range(5):
    plt.text(pop[i],exp[i],ctn[i])
plt.show()

    # scatter - Ex 3
# 그래프에 점을 찍고 좌표를 나타내기
plt.figure(figsize=(16,4))
plt.margins(0.1,0.1) # x축, y축의 여백을 조정하는 메소드
for x in range(0,8):
    for y in range(0,4):
        plt.scatter(x,y,c='b')
        plt.text(x+0.1, y, f'point({x},{y})', fontsize=10)



    # Imshow
# 잘 쓰지 않지만 기록 #이미지는 참고삼아
# 이미지 데이터처럼 행과 열을 가진 행렬 형태의 2차원 데이터는 `imshow` 명령을 써서 2차원 자료의 크기를 색깔로 표시하는 것이다.
# * http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.images[0]
X
plt.title("mnist digits; 0")
plt.imshow(X, cmap=plt.cm.bone_r)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0.35, right=0.65, bottom=0.35, top=0.65)
plt.show()

    # imshow - EX 1
digits = datasets.load_digits()
# digits.data는 1차원 데이터이기 때문에 이미지를 도식하려면 2차원으로 변경해야 함
plt.imshow(digits.data[100].reshape(8,8), cmap=plt.cm.gray_r)
plt.show()
print('\n', digits.target[100])





    # Color Map
# 데이터 수치를 색으로 바꾸는 함수는 칼라맵(color map)이라고 한다.
# 칼라맵은 `cmap` 인수로 지정한다. 사용할 수 있는 칼라맵은 `plt.cm`의 속성으로 포함되어 있다.
# 아래에 일부 칼라맵을 표시하였다. 칼라맵은 문자열로 지정해도 된다. 칼라맵에 대한 자세한 내용은 다음 웹사이트를 참조한다.
# * https://matplotlib.org/tutorials/colors/colormaps.html
dir(plt.cm)[:10]
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].set_title("plt.cm.Blues")
axes[0].imshow(X, cmap=plt.cm.Blues)
axes[1].set_title("plt.cm.Blues_r")
axes[1].imshow(X, cmap=plt.cm.Blues_r)
axes[2].set_title("plt.BrBG")
axes[2].imshow(X, cmap='BrBG')
axes[3].set_title("plt.BrBG_r")
axes[3].imshow(X, cmap='BrBG_r')
plt.show()



    # 수평선, 수직선 표시하기
# 그래프의 특정 위치에 수직선/수평선을 표시하기 위해서 matplotlib.pyplot 모듈은 아래의 네가지 함수를 지원함

    # plt.axhline(): 축을 따라 수평선을 표시
# axhline은 가로선의 데이터 좌표에서 y위치에 가로선을 표시합니다. xmin 에서 xmax 사이에서 0.0 과 1.0 사이에 있어야합니다.
# 여기서 0.0 은 멀리 있습니다 플롯의 왼쪽은 1.0 이 플롯의 가장 오른쪽입니다.
matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, hold=None, **kwargs)
plt.axhline(y=5, xmin=0.1, xmax=0.9)

    #  plt.axvline(): 축을 따라 수직선을 표시
# axvline은 ymin 에서 ymax 사이에서 시작하여 0.0과 1.0 사이에 있어야 하는 세로선의 데이터 좌표에서 x위치에 세로선을 표시합니다.
# 여기서 0.0은 맨 아래입니다. 플롯의 1.0은 플롯의 상단입니다.
matplotlib.pyplot.axvline(x=0, ymin=0, ymax=1, hold=None, **kwargs)
plt.axvline(x=5, ymin=0.1, ymax=0.9)

    # plt.hlines(): 지정한 점을 따라 수평선을 표시

    # plt.vlines(): 지정한 점을 따라 수직선을 표시
# Ex - vlines
plt.vlines( x, ymin, ymax, colors='k', linestyles='solid', label='', *, data=None, **kwargs,)
 ex) plt.vlines(x, 0, possion.pmf(x, mu), colors='b', lw=5, alpha=0.5)



    # 그래프 영역 채우기
# Keyword: plt.fill_between(), plt.fill_betweenx(), plt.fill()
    # 기본 사용 - fill_between()
plt.fill_between(x[1:3], y[1:3], alpha=0.5)
    # 기본 사용 - fill_betweenx()
plt.fill_betweenx(y[2:4], x[2:4], alpha=0.5)
    # 두 그래프 사이 영역 채우기
plt.fill_between(x[1:3], y1[1:3], y2[1:3], color='lightgray', alpha=0.5)
    # 다각형 영역 채우기 - fill()
plt.fill([1.9, 1.9, 3.1, 3.1], [1.0, 4.0, 6.0, 3.0], color='lightgray', alpha=0.5)



    # 텍스트 넣기
# text 명령을 사용하여 (x, y좌표, 텍스트(문자열))를 인수로 넘겨주면 된다.
x = np.arange(-3, 3, 0.1)
y = 1 / (1 + np.exp(-1))
plt.plot(x, y, 'r--')
 # x축 정 가운데로 설정함
plt.text(0, 5, 'Great!', horizontalalignment='center', size=50, alpha=.5, rotation=30)
plt.text(-2, .8, r'$\frac{1}{1 + e^{-x}}$', size=25) # x, y로 위치 정해준것.
plt.show()



    # 화살표와 문자열 넣기
# annotate 메소드로 그래프에 화살표를 그린후 그 화살표에 문자열을 출력할 수 있다.
x = np.arange(-5, 5, 0.1)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y, 'b--')
plt.annotate('Linear', xy=(0.1, 0.5), xytext=(1.5, 0.3), size=15,
            arrowprops={'color':'red', 'width':2}
plt.show()



    # 사진 저장하기, save
x = np.arange(1, 5, 0.1)
y1 = np.exp(x)
y2 = 2 ** x
plt.plot(x, y1, 'r', label='$e^x$')
plt.plot(x, y2, 'b--', label='$2^x$'),
plt.legend(loc='best')
plt.savefig('plot.png')
plt.savefig('plot.pdf')



    # 플롯 스타일
# plt.style.use 명령을 사용하여 라인, 포인트, 배경 등에 대한 기본 스타일을 설정할 수 있다.

    # 플롯 스타일 - Ex 1
plt.style.use('ggplot')
np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.title("Scatter Plot")
plt.scatter(X, Y)
plt.show()

# 아래 코드를 입력하면 사용 가능한 스타일들이 출력된다.
plt.style.available

# matplotlib의 디폴트 플롯 스타일로 돌아가려면 아래 명령을 실행한다.
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='NanumGothicCoding')
plt.rc('axes', unicode_minus=False)
%matplotlib inline






---------- Seaborn Visualization ----------
import seaborn as sns


	# pandas 와 seaborn의 코드 비교
# pandas는 df의 행렬을 그대로 xy로써 시각화하고 seaborn은 본래의 DF에서 x와 y를 설정하여 시각화한다.
# pandas는 인덱스를 그래프에 자동으로 나타내주므로 seaborn과 다르게 reset_index()를 해줄 필요가 없다.
# pandas
	df = pd.pivot_table(md, index='상품중분류명', columns='성별', values='구매수량', aggfunc=sum, fill_value=0)
	#df.index.name = None
	df.plot(kind='barh', figsize=(8,6), subplots=True, layout=(1, 2), sharey=True, legend=False)
	plt.show()
# seaborn
	f, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
	sns.barplot(data=df.reset_index(), y='상품중분류명', x='남', ax=axes[0])
	sns.barplot(data=df.reset_index(), y='상품중분류명', x='여', ax=axes[1])
	plt.show()



    # Seaborn
# Seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지이다.
# 기본적인 시각화 기능은 Matplotlib 패키지에 의존하며 통계 기능은 Statsmodels 패키지에 의존한다
# * https://seaborn.github.io/
# Seaborn을 임포트하면 색상 등을 Matplotlib에서 제공하는 기본 스타일이 아닌 Seaborn에서 지정한 기본 스타일로 바꾼다.
# 따라서 동일한 Matplotlib 명령을 수행해도 Seaborn을 임포트 한 것과 하지 않은 플롯은 모양이 다르다.
# * http://seaborn.pydata.org/tutorial/aesthetics.html

    # Figure style
# set: set명령으로 색상, 틱 스타일 등 전반적인 플롯 스타일을 Seaborn 스타일로 바꾼다.
# set_style: 이 명령은 틱 스타일만 바꿀 수 있다. 'darkgrid', 'whitegrid', 'dark', 'white', 그리고 ticks 스타일을 제공한다.
# set_color_codes: 이 명령으로 기본 색상을 가리키는 문자열을 바꾼다. 예를 들어 명령 적용 이전과 이후에 red 문자열이 가리키는 실제 색상 코드는 다르다.
# * `set`: http://seaborn.pydata.org/generated/seaborn.set.html
# * `set_style`: http://seaborn.pydata.org/generated/seaborn.set_style.html
# * `set_color_codes`: http://seaborn.pydata.org/generated/seaborn.set_color_codes.html
# Seaborn의 Style을 변경했으면, 한글처리를 위해 아래 명령어를 다시 실행해야 한다.
# rc('font', family='malgun gothic')
# rc('axes', unicode_minus=False)
# 스타일 바꾸면 한글설정 다시해야함.

    # Color palette
# Seaborn은 스타일 지정을 위한 색상 팔렛트(color palette)라는 것을 지원한다.
# 색상 팔렛트는 Matplotlib의 칼라맵(colormap)으로 사용할 수도 있다.
# * http://seaborn.pydata.org/tutorial/color_palettes.html
current_palette = sns.color_palette()
sns.palplot(current_palette)
sns.palplot(sns.color_palette("autumn"))
#sns.set_palette("autumn")
# 다음 실행하면 팔레트 리스트가 출력된다.
plt.colormaps()



    # seaborn plot 소개 순서
* lineplot
* histplot
* pairplot
* countplot
* barplot
* boxplot
* violinplot
* jointplot  = scatterplot type 1
* stripplot  = scatterplot type 2
* swarmplot  = scatterplot type 3
* heatmap
* clusterMap
* kdeplot
* displot = histplot + kdeplot + rugplot
* regplot
* implot



    # lineplot
    # lineplot - Ex 1
sns.lineplot(data=population, x='Year', y='Population', marker='o')
plt.ylim(0, 7e9)
plt.show()

    # lineplot - Ex 2
# 계단 형식으로 보여줌
sns.lineplot(data=postage, x='Year', y='Price', drawstyle='steps-post')
plt.xticks(np.arange(1991, 2011), rotation=90)
plt.show()



    # histplot
# 변수에 대한 히스토그램을 표시한다.
# 하나 혹은 두 개의 변수 분포를 나타내는 전형적인 시각화 도구로 범위에 포함되는 관측수를 세어 표시한다.
sns.histplot(x=df['total_bill'])
    = sns.displot(x=df['total_bill'], kind='hist')
sns.histplot(x=df['total_bill'], y=df['tip'])
    = sns.displot(x=df['total_bill'], y=df['tip'], kind='hist')



    # Pair Plot
# pairplot은 데이터프레임을 인수로 받아 그리드(grid) 형태로 각 데이터 열의 조합에 대해 스캐터플롯을 그린다.
# 데이터셋을 통째로 넣으면 숫자형 특성에 대하여 각각에 대한 히스토그램과 두 변수 사이의 scatter plot을 그린다.
# 같은 데이터가 만나는 대각선 영역에는 해당 데이터의 히스토그램을 그린다.

# 칼럼간의 관계도를 나타냄. 대각선 히스토그램
sns.pairplot(data=iris)

# 전체의 대각포함 반절만 표시: 대칭으로 그림이 같기 때문
sns.pairplot(iris, corner=True)

# hue는 species라는 칼럼의 데이터별 다른 색을 지정해주는 것
sns.pairplot(data=iris, hue='species'),

# species의 각 종별 마커모양을 다르게 표시
sns.pairplot(iris, hue="species", markers=["o", "s", "D"]

# vars는 나타낼 칼럼지정
sns.pairplot(iris, vars=["sepal_width", "sepal_length"])

# x축으로 나타낼 칼럼 설정, y축으로 나타낼 칼럼 설정
sns.pairplot(iris,
                 x_vars=["sepal_width", "sepal_length"],
                 y_vars=["petal_width", "petal_length"])

# reg로 출력결과 회귀분석으로 나타냄: 인자로 scatter와 reg가 있는데 디폴트가 scatter임.
sns.pairplot(data_result, x_vars=["인구수","CCTV수","CCTV비율"], y_vars='범죄율', kind='reg')

# size로 크기 키워줌
sns.pairplot(data=iris, vars=['sepal_length', 'petal_width'], kind='reg', size=3)

# Use kernel density estimates for univariate plots
sns.pairplot(iris, diag_kind="kde")

# Scatterplot matrix with different color by group and kde
sns.pairplot(iris,
             diag_kind='kde',
             hue="species",
             palette='bright') # pastel, bright, deep, muted, colorblind, dark



    # Count Plot
# countplot 명령을 사용하면 각 카테고리 값별로 데이터가 얼마나 있는지 표시할 수 있다.
# 범주형 변수의 발생 횟수를 샌다.
# 일변량(univariate) 분석이다.
# * `countplot`: http://seaborn.pydata.org/generated/seaborn.countplot.html
sns.countplot(data=titanic, x='class')
sns.countplot(data=titanic, x='class', hue='who')

# plot the bars horizontally - 개인적으로 이게 더 보기 편함.
sns.countplot(data=titanic, y='class', hue='who')

# Use a different color palette
sns.countplot(data=titanic, x="who",  palette="Set3")

# Use matplotlib.axes.Axes.bar() parameters to control the style.
sns.countplot(data=titanic, x="who",
                   facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 3))

    # countplot - Ex 1
f, ax = plt.subplots(1, 2, figsize=(12, 5))
titanic[['sex', 'survived']].groupby(['sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')

sns.countplot(data=titanic, x ='sex', hue='survived', ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()


    # catplot - Ex 2
# catplot를 이용하지만 kind 인자를 count로 주어서 countplot처럼 그림
sns.catplot(data=titanic, kind="count",
            x="class", hue="who", col="survived",
            height=4, aspect=.7, size=6)

    # countplot - Ex 3
fig, axes = plt.subplots(1,2, figsize=(12,6))
sns.countplot(x='status', data=lc_loans, ax = axes[0])
sns.countplot(x='grade', data=lc_loans, ax = axes[1])
plt.show()



    # Bar Plot
# barplot은 카테고리 값에 따른 실수 값의 평균★과 편차★를 표시하는 기본적인 바 차트를 생성한다.
# 평균은 막대의 높이로, 편차는 에러바(error bar)로 표시한다.
# 이변량(bivariate)분석을 위한 plot이다.
# x축에는 범주형 변수, y축에는 연속형 변수를 넣는다.
# * `barplot`: http://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(data=tips, x="day", y="total_bill")
sns.barplot(data=titanic, x='sex', y='survived', hue='class')

# Draw a set of horizontal bars
sns.barplot(data=tips, x="tip", y="day")

# Control bar order by passing an explicit order
sns.barplot(data=tips, x="time", y="tip", order=["Dinner", "Lunch"])

# Use median as the estimate of central tendency
from numpy import median
ax = sns.barplot(data=tips, x="day", y="tip", estimator=median)

# Show the standard error of the mean with the error bars
# ci 인자로 표준오차값 범위 조절
sns.barplot(data=tips, x="day", y="tip", ci=68)

# 에러바에 caps  추가: Add “caps” to the error bars
sns.barplot(data=tips, x="day", y="tip", capsize=.2)

# Use a different color palette for the bars
ax = sns.barplot(data=tips, "size", y="total_bill", palette="Blues_d")

# Plot all bars in a single color
ax = sns.barplot(data=tips, "size", y="total_bill", color="salmon", saturation=.5)

# Use matplotlib.axes.Axes.bar() parameters to control the style.
ax = sns.barplot(data=tips, "day", "total_bill",
                 linewidth=2.5, facecolor=(1, 1, 1, 0),
                 errcolor=".2", edgecolor=".2")

# catplot, kind='bar'
sns.catplot(data=tips, kind="bar",
            x="sex", y="total_bill",
            hue="smoker", col="time",
            height=4, aspect=.7)
plt.show()




    # Box Plot
# 박스 플롯은 박스와 박스 바깥의 선(whisker)으로 이루어진다.
# * `boxplot`: http://seaborn.pydata.org/generated/seaborn.boxplot.html
# 박스는 실수 값 분포에서 1분위수(Q1)와 3분위수(Q3)를 뜻하고 이 3분위수와 1분위수의 차이는(Q3-Q1)를 IQR(interquartile range)라고 한다.
# 박스 내부의 가로선은 중앙값을 나타낸다.
# 박스 외부의 세로선은 1사분위 수보다 1.5 x IQR 만큼 낮은 값과 3사분위 수보다 1.5 x IQR 만큼 높은 값의 구간을 기준으로
# 그 구간의 내부에 있는 가장 큰 데이터와 가장 작은 데이터를 잇는 선분이다.
# 그 바깥의 점은 아웃라이어(outlier)라고 부르는데 일일히 점으로 표시한다.

    # 상자그림 - 설명 2
# • 상자그림은 탐색적 데이터분석(EDA: exploratory data analysis)에서 유용한 분석도구이다.
# 이는 다음의 다섯숫자 요약(five-number summary)에 의해 만들어진다.: xmin, Q1, Q2, Q3, xmax
# • 상자의 양쪽 끝에서 수평으로 길게 뻗은 선을 보통 수염(whiskers)이라고 부르며,
# 수염의 길이는 데이터 분포의 양쪽 고리가 긴 정도, 즉 분포의 정도를 나타낸다.
# • 또한 상자그림은 데이터의 중심(center)과 변동성(variability)을 보여준다
# 단일 연속형 변수에 대해 수치를 표시하거나, 연속형 변수를 기반으로 서로 다른 범주현 변수를 분석할 수 있다.
sns.boxplot(x = df['total_bill'])
sns.boxplot(y = df['total_bill'], x = df['smoker'])
sns.boxplot(data=titanic, x='survived', y='age', hue='adult_male')

# tips = sns.load_dataset("tips")
sns.boxplot(data=tips, x='day', y='total_bill', hue='time', linewidth=2.5)
sns.boxplot(data=tips, x='time', y='tip', order=['Dinner', 'Lunch'])

# 박스플롯을 가로로 그림
sns.boxplot(data=iris, orient="h", palette="Set2")

# Use swarmplot() to show the datapoints on top of the boxes
sns.swarmplot(data=tips, x="day", y="total_bill", color=".25")

# catplot, kind='box'
g = sns.catplot(data=tips, kind="box",
                x="sex", y="total_bill",
                hue="smoker", col="time",
                height=4, aspect=.7);'
plt.show()

    # Boxplot - Ex 1
feature = numeric_feature
# Boxplot 을 사용해서 데이터의 분포를 살펴봅니다.
plt.figure(figsize=(20,15))
plt.suptitle("Boxplots", fontsize=40)
for i in range(len(feature)):
    plt.subplot(4,3,i+1) # 수치형 데이터가 11개이므로 4*3=12개 자리가 필요합니다.
    plt.title(feature[i])
    plt.boxplot(data[feature[i]])
plt.show()





    # Violinplot
# Box Plot과 비슷하지만 분포에 대한 보충 정보가 제공된다.
# boxplot이 중앙값, 표준편차 등, 분포의 간략한 특성만 보여주는데 반해
# violinplot, stripplot, swarmplot 등은 카테고리 값에 따른
# 각 분포의 실제 데이터나 전체 형상을 보여준다는 장점이 있다.
# violinplot은 세로 방향으로 커널 밀도 히스트그램을 그려주는데 위 아래가 대칭이 되도록 하여 바이올린처럼 보인다.
# * `violinplot`: http://seaborn.pydata.org/generated/seaborn.violinplot.html

# hue값을 주지 않으면 설정한 값에 대해 대칭으로 나옴
sns.violinplot(x=tips["total_bill"])
sns.violinplot(data=tips, x="day", y="total_bill")

# hue값을 주게 되면 hue값에 따라 대칭적인 비교를 할 수 있다
sns.violinplot(data=titanic, x='sex', y='age', hue='survived', split=True)

# order값으로 나타낼 순서 조정 가능
sns.violinplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])

# palette 인자, scale 인자 있음
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
                    split=True, palette="Set2",  scale="count")

# boxplot처럼 플롯 내에 quartile를 나타냄
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
                split=True, palette="Set2", scale="count", inner="quartile")

# catplot, kind='violin'
sns.catplot(data=tips, kind="violin",
            x="sex", y="total_bill", hue="smoker", col="time",
            split=True, height=4, aspect=.7)



    # jointplot = Scatterplot type 1
# 스캐터 플롯을 그리기 위해서는 Seaborn 패키지의 jointplot 명령을 사용한다.
# jointplot 명령은 스캐터 플롯뿐 아니라 차트의 가장자리(margin)에 각 변수의 히스토그램도 그린다.
# * `jointplot`: http://seaborn.pydata.org/generated/seaborn.jointplot.html
# 기본 옵션으로 dropna=True 들어가있다
# scatter도 기본 옵션이므로 구지 kind='scatter' 안써줘도 됨.
sns.jointplot(data=iris, kind='scatter', x='sepal_length', y='sepal_width')

# hue인자를 사용하면 가장자리에 각 변수의 kdeplot를 그려줌
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")

# kind='kde'로 하면 데이터를 scatterplot 대신에 kdeplot으로 나타내줌
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
                hue="species", kind="kde")

# kind='reg'로 하면 데이터에 scatter와 regplot()을 활용한 선형회귀를 나타내줌
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="reg")

# kind='hist'로 하면 데이터를 histplot()을 활용하여 나타냄
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="hist")

# scatter plot 대신 hex plot으로 정의할 수도 있다.
# kind='hex', hist로 하는 것보다 보기 좋은 것 같음
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="hex")

# 내부 scatterplot를 어떻게 표현하여 나타낼지와 가장자리 히스토그램 세팅인자
sns.jointplot(
    data=penguins, x="bill_length_mm", y="bill_depth_mm",
    marker="+", s=100, marginal_kws=dict(bins=25, fill=False),
)

# height와, ratio로 figure 사이즈 설정, marginal_ticks로 가장자리 히스토그램 수치축 표현
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
        height=5, ratio=2, marginal_ticks=True)
# To add more layers onto the plot, use the methods on the JointGrid object that jointplot() returns:
g = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)



    # Strip Plot = Scatterplot type 2
# stripplot은 마치 스캐터 플롯처럼 모든 데이터를 점으로 그려준다.
# jitter=True를 설정하면 가로축상의 위치를 무작위로 바꾸어서 데이터의 수가 많을 경우에 겹치지 않도록 한다.
# * `stripplot`: http://seaborn.pydata.org/generated/seaborn.stripplot.html
# stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs
sns.stripplot(x=tips["total_bill"])
sns.stripplot(x="day", y="total_bill", data=tips)
sns.stripplot(x="day", y="total_bill", data=tips, jitter=0.05)

# Draw horizontal strips
sns.stripplot(x="total_bill", y="day", data=tips)
sns.stripplot(x="total_bill", y="day", data=tips, linewidth=1)
sns.stripplot(x="sex", y="total_bill", hue="day", data=tips)
sns.stripplot(x="day", y="total_bill", hue="smoker",
                   data=tips, palette="Set2", dodge=True) # Draw each level of the hue variable at different locations on the major categorical axis
sns.stripplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])
sns.stripplot("day", "total_bill", "smoker", data=tips,
                   palette="Set2", size=20, marker="D",
                   edgecolor="gray", alpha=.25)



    # Swarm Plot = Scatterplot type 3
# Strip plot과 violin plot의 조합이다.
# 데이터 포인트 수와 함께 각 데이터의 분포도 제공한다.
# swarmplot은 stripplot과 비슷하지만 데이터를 나타내는 점이 겹치지 않도록 옆으로 이동한다.
# * `swarmplot`: http://seaborn.pydata.org/generated/seaborn.swarmplot.html
# swarmplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)
sns.swarmplot(data=iris, x="species", y="petal_length")

# tips = sns.load_dataset("tips")
sns.swarmplot(x=tips["total_bill"])
sns.swarmplot(x="day", y="total_bill", data=tips)
sns.swarmplot(x="total_bill", y="day", data=tips)

# Draw horizontal swarms
sns.swarmplot(x="total_bill", y="day", data=tips)
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
sns.swarmplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set2", dodge=True)
sns.swarmplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])



    # Heatmap
# Heat map을 통해 데이터 간의 수치에 따라 색상을 입힘으로써 직관적인 통찰을 얻을 수 있다.
# 만약 데이터가 2차원이고 모든 값이 카테고리 값이면 변수 간 상관관계를 보기위해 heatamp 명령을 많이 사용한다.
# `heatmap`: http://seaborn.pydata.org/generated/seaborn.heatmap.html

# heatmap 인자
# vmin, vmax : 나타낼 비율의 최소 최대 범위 설정
# center = 0 : 중간 비율을 0으로 두고 히트맵을 나타냄
# annot = True : 히트맵 셀 내부에 수치를 표현
# fmt = 'd' : 히트맵 셀 내부 수치를 비율이 아닌 정수로 표현
# linewidths = .5 : 히트맵 셀 구분 선 두께
# cmap = 'YlGnBu' : 히트맵 색상
# xticklables=2 : x축 ticks들 간의 차이 한칸을 2로 설정하여 나타냄
# cbar = False : 옆에 컬러바를 안보이게 함
# cbar_kws={"orientation": "horizontal"} # 컬러바를 가로로 나타냄
# square = True : heatmap을 반절만 나타냄

# titanic.corr()-->correlation matrix
sns.heatmap(titanic.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
plt.show()

    # Heatmap - Ex 1
df = raw.corr()
fig, ax = plt.subplots( figsize=(7,7) )
# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# 히트맵을 그린다
sns.heatmap(df,
            cmap = 'RdYlBu_r',
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )
plt.show()



    # Cluster Map
# 행렬 데이터를 가지고 있고, 유사성에 따라 몇몇 특징들을 그룹화하기 원한다면 Cluster Map을 사용하면 된다.
# 유사도가 높은 것들 순서대로 계층적으로 클러스터(Hiearchical Clustering)가 형성된다.
sns.clustermap(df.corr(), annot=True, cmap='viridis')



     # KDE Plot
# 하나 혹은 두 개의 변수에 대한 분포를 그린다.
# histplot은 절대량이라면 kdeplot은 밀도 추정치를 시각화한다.
# 그래서 결과물로는 연속된 곡선의 그래프를 얻을 수 있다
# 커널 밀도(kernel density)는 커널이라는 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포 곡선을 보여주는 방법이다.
# * `kdeplot`: http://seaborn.pydata.org/generated/seaborn.kdeplot.html

# x축
sns.kdeplot(data=tips, x="total_bill")

# y축, 그래프를 가로로 나타냄
sns.kdeplot(data=tips, y="total_bill")

# Plot distributions for each column of a wide-form dataset:
sns.kdeplot(data=iris)

# Use less smoothing:
sns.kdeplot(data=tips, x="total_bill", bw_adjust=.2)

# Use more smoothing, but don’t smooth past the extreme data points:
ax= sns.kdeplot(data=tips, x="total_bill", bw_adjust=5, cut=0)

# Plot conditional distributions with hue mapping of a second variable:
sns.kdeplot(data=tips, x="total_bill", hue="time")

# Modify the appearance of the plot:
sns.kdeplot(
    data=tips, x="total_bill", hue="size",
    fill=True, common_norm=False, palette="crest",
    alpha=.5, linewidth=0,
)

# 예를 들어 3개의 데이터 세트를 가지고 KDE 플롯을 그린다고 하면 이렇게
# shade: 곡선 아래의 공간을 음영 처리할지 결정 (True/False)
sns.kdeplot(dataset1, shade=True)
sns.kdeplot(dataset2, shade=True)
sns.kdeplot(dataset3, shade=True)
plt.legend()
plt.show()



    # Distribution Plot
# displot() + kdeplot() + rugplot()으로 기능을 합쳐놓은 것
# seaborn의 displot 명령은 러그와 커널 밀도 표시 기능이 있어서 matplotlib의 hist 명령보다 많이 사용된다.
# 러그(rug)란 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 실제 데이터들의 위치를 보여주는 것을 말한다.
# * `displot`: http://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot
# (data=None, *, x=None, y=None, hue=None, row=None, col=None, weights=None,
#   kind='hist', rug=False, rug_kws=None, log_scale=None, legend=True, palette=None,
#   hue_order=None, hue_norm=None, color=None, col_wrap=None, row_order=None, col_order=None, height=5, aspect=1, facet_kws=None, **kwargs)

# 기본은 histgram으로 나타내짐
sns.displot(data=penguins, x="flipper_length_mm")

# kind='kde', 밀도함수그래프로 나타냄
sns.displot(data=penguins, x="flipper_length_mm", kind="kde")

# kind='ecdf', empirical cumulative distribution functions (ECDFs), 경험적 누적분포함수로 나타냄
sns.displot(data=penguins, x="flipper_length_mm", kind="ecdf")

# kde=True, While in histogram mode, it is also possible to add a KDE curve
sns.displot(data=penguins, x="flipper_length_mm", kde=True)

# To draw a bivariate plot, assign both x and y:
sns.displot(data=penguins, x="flipper_length_mm", y="bill_length_mm")

# Additional keyword arguments are passed to the appropriate underlying plotting function, allowing for further customization:
# 그래프를 누적분포 히스토그램으로 표현해줌
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")

# The figure is constructed using a FacetGrid, meaning that you can also show subsets on distinct subplots, or “facets”:
# col 설정으로 sex칼럼에 대해 남여별 subplot해서 두 개의 그래프를 보여줌
sns.displot(data=penguins, x="flipper_length_mm", hue="species", col="sex", kind="kde")

    # Subplot displot - Ex 1
f, axes = plt.subplots(2, 2, figsize=(7,7), sharex=True)
sns.displot(iris.sepal_length, kde=False, color='b', ax=axes[0, 0])
sns.displot(iris.sepal_length, hist = False, rug=True, color='r', ax=axes[0, 1])
sns.displot(iris.sepal_length, hist = False, color='g', kde_kws={'shade':True}, ax=axes[1, 0])
sns.displot(iris.sepal_length, color='m', ax=axes[1, 1])
plt.show()


    # regplot
# Regression 결과를 그래프로 보여준다.
sns.regplot(x = 'tip', y = 'total_bill', data = df)



    # Implot
# 이 plot은 regplot()과 faceGrid를 결합한 것이다.
# Implot은 산포도에 직선을 그어서 추세선을 확인할 수 있게 한다.
# hue에 들어간 컬럼의 값을 구분하여 따로따로 모델링하여 결과를 보여준다.
# col 옵션주어서 해당 column의 데이터 카테고리 종류별로 그래프를 나타낼 수 있다
# * ` lmplot`: https://seaborn.pydata.org/generated/seaborn.lmplot.html
sns.lmplot(data=iris, x='sepal_width', y='sepal_length', hue='species', fit_rug=True)

# tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", data=tips)
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
               markers=["o", "x"])
 sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
               palette="Set1")
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
               palette=dict(Yes="g", No="m"))
sns.lmplot(x="total_bill", y="tip", col="smoker", data=tips)
sns.lmplot(x="size", y="total_bill", hue="day", col="day",
               data=tips, height=6, aspect=.4, x_jitter=.1)
sns.lmplot(x="total_bill", y="tip", col="day", hue="day",
               data=tips, col_wrap=2, height=3)
plt.show()


---------- Import codes ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import time
import arrow

import collections

import requests

import os
import glob

from functools import pari
from math import exp

from tqdm import tqdm_notebook
from tqdm import tqdm
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)
import seaborn as sns

# !pip install tqdm

---------- 연습문제들 ----------


    # < 분프 연습 문제 >
    # 분프 연습문제 - Ex 1
    # 이율이 3%, 5%, 10%일 때 100만원 예금 시 10년 동안의 원리금 변화를 아래 그림과 같이 도식하는 프로그램
deposit = 1000000
year = 10
interest = 3
y1 = [deposit * (1.0 + interest / 100.0) ** i for i in range(1, year+1)]
y2 = [deposit * (1.0 + (interest+2)  / 100.0) ** i for i in range(1, year+1)]
y3 = [deposit * (1.0 + (interest+7) / 100.0) ** i for i in range(1, year+1)]
plt.plot(range(1,11),y1,label='이율 3%')
plt.plot(range(1,11),y2,label ='이율 5%')
plt.plot(range(1,11),y3,label='이율 10%')
plt.xlabel('예금기간')
plt.ylabel('원리금')
plt.grid(True)
plt.legend(loc=0)
plt.show()


    # 분프 연습문제 - Ex 2
    # 윤년 평년 판정 문제
'윤년' if year%400==0 or (year%4==0 and year%100!=0) else '평년'


    # 분프 연습문제 - Ex 3
## 삼중이상 if문 (if elif else 한 줄로 쓰는 것으로 삼중이든 사중이든 다 가능함)
# - if else문 함수 만들때는 np.where 으로 만드는게 나을 때도 있으니 참고:
# 	scatter플롯의 c(color) 옵션에는 np.where로 넣는게 나음.
#	color = np.where(iris.species == 'setosa', 'r', np.where(iris.species == 'versicolor', 'g', 'b'))
EX1)  'Tall' if (num>=185) else 'short' if (num<145) else 'Regular' # 키를 입력받아 185cm이상이면 tall, 145cm이하면 short, 그외에는 regular 출력
EX2) '참 잘했습니다.' if score in ('A','B') else '좀 더 노력하세요' if score in ('C','D') else '다음 학기에 다시 수강하세요'
EX3) 'Big' if np.mean([random.randrange(9,21) for i in range(4)])>=15 else 'Small'


    # 분프 연습문제 - Ex 4
    # best seller 구하기
tr.상품중분류명.value_counts().index[0]


    # 분프 연습문제 - Ex 5
    # 마케팅 문제
cs.query('성별==2 and 70<=연령<80')['거주지역'].value_counts()[:5]


    # 분프 연습문제 - Ex 6
    # 구매일자에서 요일을 추출하여 새로운 칼럼 판매요일을 만들기
import arrow
tr['판매요일']=tr['구매일자']%100
tr['판매요일'] = [arrow.get(date, 'YYYYMMDD').format('ddd', locale='ko') for date in tr.구매일자.astype(str)]


    # 분프 연습문제 - Ex 7
    # 경과일 계산
edays = (t-pd.datetime('2018-01-01'))
edays.astype('timedelta64[D]')astype('int')


    # 분프 연습문제 - Ex 8
    # 중복되는 소분류만 뽑기
act_list = list(train_act_list.intersection(test_act_list))


    # 분프 연습문제 - Ex 9
    # 문자열 칼럼 찾기 (데이터열에 한개라도 문자값이 있을 경우 해당됨)
string_col = [i for i in string_data.columns[string_data.dtypes =='object']]


    # 분프 연습문제 - Ex 10
    # list간의 모든 조합 구하기 (두 리스트의 원소들의 모든 조합)
allCaseLst = [(e, n) for e in lst1 for n in lst2]


    # 분프 연습문제 - Ex 11
    # 딕셔너리에 단어 넣어 세기
# 딕셔너리에 키가 없으면 에러가 뜨는데, get이란 함수 써서
# 키 없으면 오류를 출력하는 대신 지정한 값을 넣어줌. 아래는 새로운 함수가 나오면 1로 시작한다
counts = dict()
names = ['빅데이터', '경영', '통계', '경영', '경영', '정보']
for name in names :
    counts[name] = counts.get(name, 0) + 1
print(counts)

    # 딕셔너리의 get 메소드를 크롤링에도 사용 가능
import requests
url = ''
res = requests.get(url)
text = res.text
words = text.split()
counts = dict()
for name in words(): # words 라는 리스트가 존재
	counts[name] = counts.get(name,0)+1


    # 분프 연습문제 - Ex 12
    # 딕셔너리에서 값이 가장 큰 것 뽑아내기
y = 0
for w,c in counts.items():
    print(w,c)
    if y <  c:
        y = c
        m = w
print()
print(m,y)


    # 분프 연습문제 - Ex 13
    # 연립방정식 풀기
# Solve the system of equations 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8:
x = np.array([[3,1], [1,2]])
y = np.array([9,8])
np.linalg.solve(x, y)


    # 분프 연습문제 - Ex 14
    # 어떤 달력을 출력하던 나오게 만들어 보아라
import arrow
year = int(input('년도 :'))
month = int(input('월 :'))
print('{}년 {}월'.format(year,month))
print('일 월 화 수 목 금 토')
wd_1 = arrow.get(year,month,1).weekday()
if month in [1,3,5,7,8,10,12]:
    for i in range((wd_1 +1)%7):
        print('{:2}'.format(' '), end=' ')
    for i in range(1,32):
        print('{:2}'.format(i), end=' ')
        if arrow.get(year,month,i).weekday() == 5:
            print()

elif month in [4,6,9,11]:
    for i in range((wd_1 +1)%7):
        print('{:2}'.format(' '), end=' ')
    for i in range(1,31):
        print('{:2}'.format(i), end=' ')
        if arrow.get(year,month,i).weekday() == 5:
            print()
elif month ==2:
    if ((year%4 == 0) and (year%100 != 0)) or (year%400 == 0):
        for i in range((wd_1 +1)%7):
            print('{:2}'.format(' '), end=' ')
        for i in range(1,30):
            print('{:2}'.format(i), end=' ')
            if arrow.get(year,month,i).weekday() == 5:
                print()
    else:
        for i in range((wd_1 +1)%7):
            print('{:2}'.format(' '), end=' ')
        for i in range(1,29):
            print('{:2}'.format(i), end=' ')
            if arrow.get(year,month,i).weekday() == 5:
                print()


    # 분프 연습문제 - Ex 15
    # 중요한 날 또는 기념일을 입력하면 현재 날짜 기준으로 몇일 남았는지 또는 몇일 지났는지 또는 오늘인지 알려주는 프로그램
datd = input('기념일을 입력하세요. (형식 YYYY-MM-DD): ')
days = (arrow.get(date) - arrow.now()).days
if days >0:
    print('{}일 남았습니다.'.format('days'))
elif days == 0:
    print('오늘입니다.')
else: print('{}일 지났습니다.'.format(days))



    # 분프 연습문제 - Ex 16
    # 년, 월, 일을 입력받아 요일(한글명)을 출력하는 프로그램
year = int(input('연도: '))
month = int(input('월: '))
day = int(input('일: '))
arrow.get(year, month, day).format('dddd', locale='ko')


    # 분프 연습문제 - Ex 17
    # zip과 format의 이용
import random
names= ['덕상','홍쓰','성민','성식']
heights = [random.randrange(160,200) for i in range(4)]
weights= [random.randrange(50,90) fofr i in range(4)]
for i,j,k in zip(names, heights, weights):
    print('{}는 {}이며, 몸무게는 {}입니다'.format(i,j,k))


    # 분프 연습문제 - Ex 18
    # 날짜를 나타내는 문자열 리스트 dates로부터 각 날짜에 상응하는 요일을 출력하는 코드이다.
	# 이 코드를 1) list comprehension, 2) map 함수를 이용하여 다시 작성하시오.
dates = ['1945/8/15', '1946/9/1', '1960/4/19', '1980/5/18', '2013/3/1']
# list comprehension 사용
[arrow.get(date).format('dddd', locale='ko') for date in dates]
# map과 lambda 함수 사용
lam = lambda date: arrow.get(date).format('dddd', locale='ko')
print(list(map(lam,dates)))


    # 분프 연습문제 - Ex 19
    # humanize
date = arrow.now().shift(days=2)
date.humanize() # => 출력결과: 'in 2 days'
date.humanize(locale='ko') # => 출력결과: '2일 후'


    # 분프 연습문제 - Ex 20
    # 인덱싱을 통해 리스트 x에서 최대값을 추출
for i in range(len(x)):
    if y<x[i] : y = x[i]
print(y)

    # 리스트 안의 값과 비교하여 리스트 원소중 최소값을 추출
for i in x:
    if y>i : y=i
print(y)


    # 분프 연습문제 - Ex 21
    # 아래 딕셔너리 변수를 값 기준으로 정렬하시오.
c = {'a': 10, 'c': 1, 'b': 22}
lst = sorted([(v,k) for k,v in c.items()])
c = {k:v for (v,k) in lst}
print(c)
	# 역정렬
lst = sorted([(v,k) for k,v in c.items()], reverse=True)
c = {k:v for (v,k) in lst}
print(c)
	# 만약 v 값이 앞에 와도 괜찮다면 한줄로 감싸서 풀 수 있다
dict(sorted([(j,i) for i,j in c.items()], reverse=True))


    # 분프 연습문제 - Ex 22
    # 오바마 연설문에서 가장 많이 나타나는 단어와 그것의 횟수를 출력하는 프로그램을 작성하시오
import requests
url = 'http://cfile221.uf.daum.net/attach/1332E3064975C0E70F1AFF' #오바마 연설문 화일
res = requests.get(url)
text = res.text
words = test.split()

counts = dict()
for word in words:
    counts[word] = counts.get(word, 0) +1

y = 0
for w,c in counts.items():
    if y < c:
       y = c
       m = w
print(m, y)

# Revision #1: list comprehension과 max() 함수를 사용하여 가장 많이 나타나는 단어와 그것의 횟수를 출력
max([(v, k) for k, v in counts.items()]) # 한줄로 표현 가능
# Revision #2: list comprehension과 sorted() 함수를 사용하여 출현 빈도가 가장 높은 상위 10개 단어 리스트 출력
sorted([(v,k) for k, v in counts.items()], reverse = True)[:10]


    # 분프 연습문제 - Ex 23
    # -10에서 10까지의 정수 리스트 x가 주어질 때,
    # map과 lambda를 이용하여 $y = \cfrac{1}{1+exp(-x)}$인 실수 리스트 y를 만드시오.
    # 그리고 x와 y간의 관계를 matplotlib의 plot()을 이용하여 도식하시오.
x = range(-10, 11)
y = list(map(lambda x: 1/(1+exp(-x)), x))
plt.plot(x, y)
plt.show()



    # 마케팅 질문들 (롯데마트 & 롯데백화점 데이터 분석할 때의 기본질문들)
# 60대 여성 고객리스트를 출력하시오
# 남성고객과 여성고객은 각각 몇명인가?
# 여성고객의 평균나이는 얼마인가?
# 고객의 거주지역 리스트를 출력하시오.
# 고객들이 사는 지역의 수는?
# 70대 여성 고객들은 주로 어느 지역에 거주하고 있는가? 상위 5개 지역만 나열하시오.
# (상품중분류명 기준) 판매 상품의 종류은 몇개인가?
# (상품중분류명 기준) Best seller는 무엇인가?
# 전 지역에서 판매량이 가장 많은 상품의 총 매출액은 얼마인가?
# 축산물은 하루 중 언제 가장 많이 팔리는가?
# 수산물이 몇 월에 얼마나 팔리는지 확인
# 구매일자에서 월(month)을 추출하여 새로운 컬럼 판매월을 만드시오.
# 구매일자에서 요일을 추출하여 새로운 컬럼 판매요일을 만드시오.
# 성별 컬럼의 값을 남, 여로 바꾸시오.
# 거주지역을 시와 구 또는 도와 시로 나누어 두 개의 새로운 컬럼(거주지역_광역, 거주지역_기초)을 만드시오.
# 구매지역을 시와 구 또는 도와 시로 나누어 두 개의 새로운 컬럼(구매지역_광역, 구매지역_기초)을 만드시오
# 60대 고객의 총 구매건수는 얼마인가?
# 남녀별 평균구매액를 계산하여 출력하시오.
# 총 구매액이 가장 많은 사람부터 적은 사람 순으로 정렬하여 상위 10명만 출력하시오. -> vvips, vips, gold, silver, bronze...
						# 퍼센트로 나타내도 좋을듯
#  전국에 사는 고객들의 지역별(광역) 구매액를 계산하여 출력하시오. 단, 지역명은 열로 나타낼 것.
# 각 고객 별로 월 별 판매액(amount)의 합계를 계산하여 출력하시오.
# 남녀 별로 요일 별 판매액(amount)의 합계를 계산하여 출력하시오.
	# 시각화
# 각 고객의 총구매건수를 구한 후 Histogram을 출력하시오. 단, bin의 크기는 30로 설정.
# 남녀별 시간대별 구매수량을 Bar Chart로 도식하시오.
# 상품별(중분류) 성별 구매수량 계산하여,  subplot이 적용된 Chart를 출력하시오
# 서울시에 거주하는 실버고객의 지역별(구별) 분포를 Pie Chart로 시각화하시오
# 변동계수 구하기 : 표준편차 / 평균






    # Load Datas
# 분류나 회귀 연습용 예제 데이터
# - 사이킷런에 내장된 이 데이터 세트는 일반적으로 딕셔너리 형태로 돼 있습니다.
#     키는 본통 data, target, target_name, feature_names, DESCR로 구성돼 있습니다. 개별 키가 가리키는 데이터 세트의 의미는 다음과 같습니다.
# - data는 피처의 데이터 세트를 가리킵니다.
# - target은 분류 시 레이블 값, 회귀일 때는 숫자 결괏값 데이터 세트입니다.
# - target_names는 개별 레이블의 이름을 나타냅니다.
# - feature_names는 피처의 이름을 나타냅니다.
# - DESCR은 데이터 세트에 대한 설명과 각 피처의 설명을 나타냅니다.

from sklearn import datasets
datasets.load_boston()	 : 회귀 용도이며, 미국 보스턴의 집 피처들과 가격에 대한 데이터 세트
datasets.load_breast_cancer() : 분류 용도이며, 위스콘신 유방암 피처들과 악성/음성 레이블 데이터 세트
datasets.load_diabetes() 	 : 회귀 용도이며, 당뇨 데이터 세트
datasets.load_digits()	 : 분류 용도이며, 0에서 9까지 숫자의 이미지 픽셀 데이터 세트
datasets.load_iris()		 : 분류 용도이며, 붓꽃에 대한 피처를 가진 데이터 세트

    # datasets iris - Ex 1
from sklearn.datasets import load_iris
iris = sns.load_dataset('iris')
        = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris.head(3)

# datasets diabetes - Ex 2
diabetes_data = pd.read_csv('diabetes.csv')
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]



────────────────────────────────────────────────
