
	# 전체 흐름, 과정
* Basic Methods
* Function
* List
* Dictionary
* Set
* 문자열
* 시간, 요일
* 파일 읽고 쓰기
* 객체지향 프로그래밍


---------- Basic Methods ----------

# 산술 연산자
/ 나누기
// 나누기(정수)
% 나머지
**제곱
abs(x) 절대값
y = float('-inf') # 파이썬에서 무한대는 float('inf')또는 math.inf로 표현한다.


# continue
	# `while` 문이나 `for` 문을 사용하는 반복문에서 `break`를 사용하면 반복문 전체를 빠져나오게 된다.
	# 그런데 반복문 전체를 빠져나오는 것이 아니라 해당 조건만 건너뛰고 싶을 때는 어떻게 하면 될까?
	# (Ex: 1부터 10까지 출력하는데 짝수인 경우는 출력하지 않음).  이럴 때 사용할 수 있는 것이 바로 `continue` 문이다.
continue # 조건에 해당하는 것을 실행하지 않고 다음 반복으로 넘어간다.
pass # 아무 일도 하지 않는다.



# 무슨 타입인지 알고 싶을때 안에 넣으면 타입출력
type(x)


# x에 적용할 수 있는 기능들을 모두 알려줌
dir(x)


# all, any
all(x) # 반복가능한(iterable) 자료형 x를 입력 인수로 받으며 이 x가 모두 참이면 True, 거짓이 하나라도 잇으면 False를 돌려준다
any(x)# x중 하나라도 참이 있으면 True, 모두 거짓일 때에만 False를 돌려준다.


## *args
	# 가변인자(variadic parameters)를 사용하고자 할 때 이를 나타낼 수 있는 문법이 필요한데,
	# 이것이 인자의 `packing`과 `unpacking`이다.
	# 함수를 정의할 때 가변적인 인자를 `*args`로 나타내면 된다. (여기서 args라는 변수명은 얼마든지 바꿀 수 있지만 그대로 사용하는 것이 관습)
def ssum(*args):
	return sum([i for i in args])

def ssum(*args):
    return sum([*args])

print(*args) # 튜플에 가로가 없어져서 나온다. print(2,3,4,5)와 동치임 !!
    	   # 이때, `*args`는 unpacking을 나타낸다. (즉, 튜플의 괄호를 없앤다고 생각하면 됨)



# 글로벌 변수
# 전역변수 지정으로 함수 내외에서 모두 a를 사용,수정 가능하도록 함
global a



# 정렬
sorted(x) # 오름차순 정렬
sorted(x.index)
sorted(x.values)
sorted(x, reverse=True) # 내림차순 정렬 = sorted(x)[::-1]
sorted(L, key=lambda x: len(x)) # 정렬의 키를 지정할 수 있다. 예시는 문자의 길이를 키로 지정하여 정렬하도록 한 것.
	# sort() 메소드 예시
L = [{'name': 'John', 'score':83}, {'name':'Paul', 'score':92}]
L.sort(key=lambda x: x['score'], reverse=True) # 레코드들을 점수 높은 순으로 정렬



# else문 줄이기
'large' if b>5 else 'small'
'two two' if i==2 else 'three three' if i==3 else '.'
'미성년' if age <20 else '청년' if age <40 else '중년' if age < 65 else '노인'



# Python Permutation (순열)
from itertools import permutations
a = [1,2,3]
permute = permutations(a,2)
print(list(permute))
# 결과
[(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]



# Python Combination (조합)
from itertools import combinations
a = [1,2,3]
combi = combinations(a,2)
print(list(combi))
# 결과
[(1,2),(1,3),(2,3)]



    # assert
# - 가정 설정문(assert)
# - assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다.
#    >>> a = 3
#    >>> assert a == 2
# 결과
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AssertionError

    # 왜 assert가 필요한 것일까?
# 어떤 함수는 성능을 높이기 위해 반드시 정수만을 입력받아 처리하도록 만들 수 있다.
# 이런 함수를 만들기 위해서는 반드시 함수에 정수만 들어오는지 확인할 필요가 있다.
# 이를 위해 if문을 사용할 수도 있고 '예외 처리'를 사용할 수도 있지만 '가정 설정문'을 사용하는 방법도 있다.
# 아래 코드는 함수 인자가 정수인지 확인하는 코드이다.
lists = [1, 3, 6, 3, 8, 7, 13, 23, 13, 2, 3.14, 2, 3, 7]
def test(t):
    assert type(t) is int, '정수 아닌 값이 있네'

for i in lists:
    test(i)
#결과
AssertionError: 정수 아닌 값이 있네
# lists에 실수가 하나 있으므로 AssertionError가 발생했다.

    # assert 문은 다음 형식으로 작동한다.
assert 조건, '메시지'
'메시지'는 생략할 수 있다.
# assert는 개발자가 프로그램을 만드는 과정에 관여한다.
# 원하는 조건의 변수 값을 보증받을 때까지 assert로 테스트 할 수 있다.
# 이는 단순히 에러를 찾는것이 아니라 값을 보증하기 위해 사용된다.
# 예를 들어 함수의 입력 값이 어떤 조건의 참임을 보증하기 위해 사용할 수 있고 함수의 반환 값이 어떤 조건에 만족하도록 만들 수 있다.
# 혹은 변수 값이 변하는 과정에서 특정 부분은 반드시 어떤 영역에 속하는 것을 보증하기 위해 가정 설정문을 통해 확인 할 수도 있다.
# 이처럼 실수를 가정해 값을 보증하는 방식으로 코딩 하기 때문에 이를 '방어적 프로그래밍'이라 부른다.



	# 문제 풀이시 참고사항
# len(s)==4 or len(s)==6
#  -> 이런건 len(s) in (4,6) 으로.



---------- Function ----------

# 재귀함수 Recursion Function: 자기 자신을 호출하는 함수
    -> 재귀가 중지되는 조건 반드시 필요



# Lambda
	# lambda 함수 활용의 예1
fill_cat2_func = lambda x : np.nan if len(x.mode()) == 0 else x.mode()[0]0
df['Product_Category_2'].fillna(df.groupby(['Gender', 'Age', 'Occupation', 'Product_Category_1'])['Product_Category_2'].transform(fill_cat2_func), inplace = True)



# 코사인 유사도 구하기
def cos_similarity(v1, v2):

    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1)))*np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm

    return similarity




---------- List ----------

## List Methods
x.append('f') # x에 f를 추가
x.pop() # 리스트에서 마지막 값을 제거
x.insert() # 지정한 위치에 값을 삽입한다. (리스트명.insert(위치, 값))
x.remove() # 리스트에서 지정한 값을 삭제. 지정 값이 여러개일 경우 첫번째 값만 삭제. (리스트명.remove(지울값))
x.extend() # 리스트 뒤에 리스트를 추가한다. 리스트 + 리스트와 동일. (리스트명.extend(추가 리스트))
x.index('c') # c의 위치를 찾는다
x.index('c', 40) # 찾으려는 문자가 여러개 있을 때 시작위치를 입력하면 그 시작위치 이후로 찾음
x.rindex('c') # 찾으려는 문자가 여러개 있을 때 마지막번호의 인덱스를 출력
x.count() # 리스트에서 해당 값의 개수를 센다
x.copy() # 리스트의 내용을 새로운 리스트에 복사한다.
x.clear() # 리스트의 내용을 모두 지운다

x[::-1]  # 리스트의 순서 뒤집기

del x[-1] # 마지막 값을 제거 # del 리스트[인덱스번호]
del x[:2] # 첫 두개의 값 제거

a[1:1] = [500, 600] # a -> [10, 500, 600, 20, 30] # 리스트 중간에 요소 여러개를 추가하고 싶다면 슬라이스에 요소 할당하기를 활용



## List comprehension
    # 리스트 컴프리핸션 예시문제
for i in range(1,n+1):
  if i%2==1:
    ans.append('수')
  else:
    ans.append('박')
''. join(ans)
>>>>>>>>>>>>>>>>''.join(['수' if (i%2==1) else '박' for i in range(1,n+1)])



## map
map(function, iterable) # map은 입력받은 자료형의 각 요소를 함수 f가 수행한 결과를 묶어서 돌려받는 함수이다.
list(map(lambda x: x**k, [1,2,3,4,5])) # 값을 받을 때 list또는 dict으로 받아줘야 한다.



## zip
	# 같은 길이의 여러 리스트로부터 인덱스가 같은 값끼리 묶어서 리스트로 반환한다.
a = "YUN"
b = [1,2,3]
c = ("하나","둘","셋")
print(list(zip(a,b,c)))
print(set(zip(a,b,c)))
	# zip을 이용해 학생들의 총점구하기 / 반복문에 zip을 이용
학생 = list('abcd')
중간고사 = [34, 30, 26, 40]
기말고사 = [45, 48, 25, 50]
과제점수 = [5, 10, 8, 10]
총점 = []
for i, j, k in zip(중간고사, 기말고사, 과제점수):
    총점.append(i+j+k)
{k:v for k in 학생 for v in 총점}
	# 점수를 합치지 않고 나타내면
중간고사 = [34, 30, 26, 40]
기말고사 = [45, 48, 25, 50]
과제점수 = [5, 10, 8, 10]
학생 = list('abcd')
학생별점수 = list(zip(중간고사,기말고사,과제점수))
dict(zip(학생,학생별점수))



## 두 리스트의 원소들의 모든 조합
col1 = ['sweet', 'annoying', 'cool', 'grey-eyed']
col2 = ['john', 'alice', 'james']
comb_lst = [(e, n) for e in col1 for n in col2]
comb_lstl



## enumerate
		# 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력받아
		# `for` 문을 `enumerate()`와 같이 사용하면, 각 요소의 값과 인덱스 값을 동시에 얻을 수 있다.
name = ['사과', '포도', '딸기']
for i, j in enumerate(name):
    print(i, j) ## 인덱스와 값을 각각 받겠다
name = ['사과', '포도', '딸기']
for k in enumerate(name):
    print(k) ##  튜블을 통채로 받게 된다.
name = ['사과', '포도', '딸기']
for _,j in enumerate(name):
    print(j) ##값만 받겠다
name = ['사과', '포도', '딸기']
for i, j in enumerate(name, 100): # 만약 인덱스의 시작값을 바꾸고 싶다면, 원하는 시작값을 넘겨줄 수 있다. 예시로 인덱스 값을 100로 시작하게 바꿔보자.
    print(i, j)
list(enumerate(name,100))

Ex문제) enumerate()를 이용하여 ['a', 'b', 'c']라는 리스트를 {0:'a', 1:'b', 2:'c'}라는 딕셔너리로 바꾸시오.
lst = list('abc')
adict = {}
for i,v in enumerate(lst, 100):
	adict[i] = v
	# 딕- 컴프리핸션으로 풀어보기
lst = list('abc')
dic = {k,v for k,v in enumerate(lst,100)}
	# 딕으로 감싸서 풀어보기
lst = list('abc')
dic = dict(enumerate(lst,100)) # 아마 제일 효율적
	# zip을 이용해 풀어보기
k = range(3)
v = 'abc'
dic = dict(zip(k,v))
	# zip을 이용해 한줄로 풀면
dict(zip(range(3),list('abc')))


## collections
import collections # 컨테이너에 동일한 값의 자료가 몇개인지 파악하는데 사용하는 객체
collections.Counter()는 산술/집합 연산이 가능하다. #덧셈,뺄셈,교집합,합집합
collections.Counter(participant)-collections.Counter(completion)
collections.Counter(lst) # 입력값으로 리스트나 딕셔너리를 넣는다

collections.Counter(a=2,b=3,c=2) == ['a','a','b','b','b','c','c']

container = collections.Counter()
container.update('aabcdeffgg') # update()는 Counter의 값을 갱신하는 것을 의미
print(container)
for k,v in container.items():
	print(k,':',v)

list(c.elements()) # elements()는 입력된 값의 요소에 해당하는 값을 풀어서 무작위로 반환.(1보다 작을 경우 출력하지 않음)
sorted(c.elements())
c2.most_common() # 입력된 요소들 중 빈도수가 높은 순으로 리스트안에 튜플형태로 반환
c2.most_common(3) #가로 안에 3 입력하면 빈도수가 3인 것들 나옴








---------- Dictionary ----------

## Dictionary
# 사전의 키에는 숫자, 문자, 튜플 등 불변(immutable)인 값만 넣을 수 있다. 리스트처럼 가변 (mutable)적인 값은 키로 사용할 수 없다.
fruit = {'사과': 10, '딸기': 20} # 사전의 생성
fruit['사과'] # 사전을 키로 인덱싱하면, 그에 대응하는 값을 돌려받을 수 있다.
fruit['사과'] = 100 # 또한 기존의 값을 덮어 쓸 수도 있다.
fruit['수박'] = 30 # 기존에 없는 키로 인덱싱하고 값을 할당하면 새로운 원소가 추가된다.
    # 키가 중복될 경우
repeated_dict = {'a': 1, 'a': 5, 'b': 10}  # 사전에서는 키가 중복될 경우 먼저 나온 키는 무시한다. 'a' : 5로 됨.
	# 사전형 메서드
fruit.keys() # 사전의 키를 모두 확인하고 싶다면, .keys() 메소드를 부른다.
fruit.values() # 사전의 값을 모두 확인하고 싶다면, .values() 메소드를 부른다.
fruit.items() # 사전의 키와 값을 짝지어 보고 싶다면, .items() 메소드를 부른다.
fruit.get('포도', 500)  # 사전에 존재하지 않는 키를 찾으면, KeyError 가 발생하는데, .get() 메소드는 이를 방지할 수 있게 해준다.
		  # 즉, KeyError를 일으키는 대신 아무 것도 반환하지 않거나 설정된 기본값을 반환한다.

'사과' in fruit # in 연산자로 특정 값이 사전에 있는지 확인할 수 있다.
del fruit['사과']  # '사과' 삭제
fruit.clear() # 사전을 전부 비우고 싶다면, .clear() 메소드를 사용한다.

# Dictionary comprehension
{i: np.sin(i) for i in range(1,21)}




---------- Set ----------

## Set
x = {'사과', '포도', '바나나'}  # 집합 생성
set(['사과', '사과', '포도', '바나나', '바나나', '사과']) # set() 함수에 다른 값을 넣으면 집합으로 변환된다. 만약 겹치는 항목이 집합에 들어있다면, 반복되는 항목은 제거된다.
	# 집합형 메서드
x.add('오렌지')  # 집합에 '오렌지' 추가
x.update(['수박', '딸기'])  # 집합에 '수박', '딸기' 추가. 하나 이상의 원소는 .update() 메소드를 사용한다.
x.remove('오렌지') # 원소를 제거하고 싶을때는 .remove() 메소드를 사용한다.
'바나나' in x # in 연산자로 특정 값이 집합에 있는지 확인할 수 있다.


a | b # 두 집합 사이의 합집합은 |으로 구한다. # 집합1.union(집합2)
a & b # 두 집합 사이의 교집합은 &으로 구한다. # 집합1.intersection(집합2)
a - b # 두 집합 사이의 차집합은 -으로 구한다.
합집합: 집합1.union(집합2) / 집합1 | 집합2
교집합: 집합1.intersection(집합2) / 집합1 & 집합2
차집합: 집합1 - 집합2




---------- 문자열 ----------


# 문자열 함수: 구성파악
문자열.isdigit(), # 문자가 숫자로 구성되어 있는지
문자열.isnumeric(), # 이건 1/2 이런 특수문자도 True 판단함
문자열.isalpha(), # 문자가 문자로 구성되어 있는지
문자열.isalnum(), # 문자가 숫자 또는 문자로 구성되어 있는지 -> 특수문자 구분
문자열.islower(), # 문자가 소문자로만 구성되어 있는지
문자열.isupper(), # 문자가 대문자로만 구성되어 있는지
문자열.isspace(), # 문자가 공란인지
문자열.swapcase(), # 대소문자 바꿈
문자열.title(), # 주어진 문자열에서 알파벳 외의 문자로 나누어져 있는 영단어들의 첫 글자를 모두 대문자로 변환시킨다.
문자열.capitalize(), # 주어진 문자열에서 맨 첫 글자를 대문자로 변환시킨다.

문자열.upper(), 문자열.lower(),
문자열.split(), # 디폴트로 아무것도 안넣으면 스페이스바를 기준으로함
문자열.startswith(부분 문자열),
문자열.endswith(부분 문자열),
문자열.find(부분 문자열),
    # 인덱스를 이용하여 reverse시키기
v[::-1]
    # 따옴표 세번
[""" """] : 세개의 따옴표 내부의 모든 자료를 통으로 문자열로 인식한다.
  => 여러줄 주석을 달 때 많이 활용


# 문자열 함수: 분리와 결합, 정렬과 채우기
'/'.join(sd.split(' ')).upper() # split()안에 문자를 주면 해당 문자를 기준으로 나눔
.splitlines() # 긴 문장에서 개행(줄 단위 = \n)로 구분하고자 할 때
- join
' '.join(nlst) # 리스트안의 것들 ' '로 연결시켜줌
','.join('abcd') # => 'a,b,c,d' : 문자열을 특정 글자로 합쳐준다.
''.join(sorted(s,reverse=True)) #''.join()하면 리스트 글자 붙여짐
				#sorted한것 역정렬: 안에 reverse=True붙이기
.center(자리수, 문자) # 가운데 정렬, 문자열을 넣으면 빈 공간에 문자열을 넣는다)
.ljust(자리수, 문자) # 왼쪽 정렬, 문자열을 넣으면 빈 공간에 문자열을 넣는다)
.rjust(자리수, 문자) # 오른쪽 정렬, 문자열을 넣으면 빈 공간에 문자열을 넣는다)
.zfill(자리수) # 자리수만큼 채워넣고, 오른쪽 정렬후 남는 자리는 0으로 채운다.


## 바이트 byte 코드로 바꾸기
문자열.encode() # 문자열을 바이트코드로 변경
문자열.decode() # 원래 파이썬 문자열 타입인 유니코드로 문자열을 변경



# 문자열 함수: 공백 또는 특정문자 지우기, 변경하기
'  hello  '.strip() # 앞 뒤의 공백을 지운다, 안에 문자열을 넣을 경우 양쪽에 있는 해당 문자열을 지우고 출력
'  hello  '.rstrip() #  오른쪽 공백을 지운다, 안에 문자열 넣을 경우 오른쪽에 있는 해당 문자열을 지우고 출력
'  hello  '.lstrip() # 왼쪽 공백을 지운다, 안에 문자열 넣을 경우 왼쪽에 해당 문자열을 지우고 출력


# replace
'파이쏜은 참 쉽습니다?'.replace('쏜','썬') # 문자열 교체
	# replace는 .replace('','')하고 뒤에다가 또 .replace()가능 and replace안에 리스트로 바꿔주는 것도 가능


## print()문의 기능들과 서식
end = ''  # 줄 바꿈 없이 입력
print('a', 'b', end = ' ')
sep = '' # 여러개의 출력 간의 구분자 입력
f-string
 - print(f'전공 : {"빅데이터":20}\\ 학번 : {"20162533":20}\\ 이름 : {"홍길동":20}') # 글자 출력시 쌍따옴표 이용, 뒤에서 글자 출력 갯수 조절가능
 - print(f'정수 {123}의 8진수는 {123:o}, 16진수는 {123:x}') # 8진수 입력시 o, 16진수 입력시 x
 - print(f'\t{123.456789:.8}') # 소수점 표시는 소수점 아닌 자리부터 소수점까지 출력 숫자 세도록 되어 있다









---------- 시간, 요일 ----------
[time]

import time
start = time.time() # 시작 시간 설정
 print(f'소요시간: {time.time() - start:.3f}초')

# localtime() 함수는 주어진 timestamp 값을 현지 시간대 기준의 time_struct 타입 데이터로 변환해줍니다.
tm = time.localtime() # time.localtime(time.time()) 와 동일

# strptime() 함수는 strftime() 함수와 정반대로 특정 포멧의 문자열을 time_struct 타입으로 변환을 해줍니다.
time.strftime('%Y-%m-%d %I:%M:%S %p', tm)


[arrow]

import arrow
arrow.now() # `arrow.now()`를 이용하여 현재시간을 알 수 있다
arrow.get('2019-04-10') # 2019년 4월 10일을 arrow 형태로 가져온다.
	# == arrow.get(2019,4,10) # 문자열을 사용하지 않고도 날짜를 지정할 수 있다.
arrow.get('20190410', 'YYYYMMDD')  # 연도 네 자리, 월 두자리, 일 두자리로 인식하게 지정
arrow.get('04/10/19', 'MM/DD/YY')
date.format('YY년 MM월 DD일') # format: 날짜의 출력 형식을 지정할 수 있게 해준다.
	- `Y`는 연도, `M`는 월, `D`는 일,
	- `h`는 시, `m`은 분, `s`는 초를 뜻한다.
date.format('YYYY.MMMM')  # M 4개를 사용하면 숫자 대신 영어로 월이 출력된다.
date.format('dddd')  # d 4개를 사용하면 숫자 대신 영어로 요일이 출력된다.
date.format('dddd', locale='ko') # locale 인자를 'ko'로 지정하면 우리말로 요일이 출력된다.
three_weeks_later = date.shift(weeks=3) # date 를 3 주후의 시간으로 옮긴다.
three_weeks_before = date.replace(weeks=-3) # date 를 3주 전의 시간으로 옮긴다.
date.replace(days=3) #3일 후
date.replace(day=3) #3일로
date.weekday() #0은 월요일이고 6은 일요일이다.
date = arrow.now().shift(days=3) = date.humanize() = date.humanize(locale='ko')
dates2 = [arrow.get(i).format('dddd', locale='ko') for i in dates]
dates3 = list(map(lambda x: arrow.get(x).format('dddd',locale='ko'),dates)
arrow.get(int(year),int(month),int(day)).format('dddd',locale='ko')



[function]
## 요일 만드는 방법
def fw(x):
    return(('월','화','수','목','금','토','일')[x.weekday()])
md['요일']=pd.to_datetime(md.구매일자.astype(str)).apply(fw)+'요일'
md['요일']=md.요일.astype('category').cat.reorder_categories(['월요일','화요일','수요일','목요일','금요일','토요일','일요일'],ordered=True)
# 위에가 요일 순서 정렬방법



# 날짜 빼기
# arrow 에서 `-`을 사용해 날짜에서 날짜를 뺄 수 있다.
date1 = arrow.get('2019-11-12')  # date1을 11월 12일로 지정한다.
date2 = arrow.get('2019-12-01')  # date2를 12월 1일로 지정한다.
difference = date2 - date1  # date2 에서 date1 을 빼준다.
difference.days  # difference.days 로 총 며칠인지 확인한다.




---------- 파일 읽고 쓰기 ----------


# 파일 입출력
 - 파일 열기 / 닫기
변수명 = open('파일명', '옵션', encoding='utf-8')
변수명.close()
옵션: r(default, 읽기모드), w(쓰기모드, 기존에 파일이 있으면 덮어쓴다),
      r+(읽기/쓰기 겸용모드), a(쓰기 모드, 기존에 파일이 있으면 이어서 쓴다. append의 약어이다)
      t(텍스트 모드, 텍스트 파일을 처리한다, 기본값이다), b(이진 모드, 이진 파일을 처리한다.)

with open('파일명', '옵션', encoding='utf-8') as 변수명:
-> 변수명.close()를 사용하지 않아도 된다


## 파일 읽기
read() 파일전체
readline() 라인별
readlines() 라인별로 읽어서 리스트화

    # EX1: read()
infp = None
instr = ""
# infp = open('ch12.txt', 'r')
with open('ch12.txt', 'r') as infp:
    instr = infp.read()
    print(instr, end=' ')
# infp.close()

    # EX2: readline()
infp = None
instr = ""
# infp = open('ch12.txt', 'r')
with open('ch12.txt', 'r') as infp:
    instr = infp.readline()
    print(instr, end='')
    instr = infp.readline()
    print(instr, end='')
    instr = infp.readline()
    print(instr, end='')
# infp.close()

    # EX3: readlines()
infp = None
instr = ""
# infp = open('ch12.txt', 'r')
with open('ch12.txt', 'r') as infp:
    instr = infp.readlines()
    print(instr, end='')
# infp.close()


## 파일 쓰기
write()
writelines()
    # Ex1 : write()
outfp = None
outstr = ""
# outfp = open('ch12.txt', 'w')
with open('ch12.txt', 'w') as outfp:
    while True:
        outstr = input()
        if outstr != "":
            outfp.write(outstr + '\n')
        else:
            break;
# outfp.close()
    # Ex2 : writelines()
outfp = None
outstr = ""
# outfp = open('ch12_01.txt', 'w')
with open('ch12_01.txt', 'w') as outfp:
    while True:
        outstr = input()
        if outstr != "" :
            outfp.writelines(outstr + '\n')
        else:
            break;
# outfp.close()


## 파일 및 디렉토리 다루기
 - shutil, os, os.path 활용
# shutil
shutil.copy('원본파일', '대상파일') # 파일 및 디렉터리 복사
shutil.copytree('원본 폴더', '대상 폴더')
shutil.rmtree('삭제 폴더경로 및 폴더명')

# os
os.getcwd() # 현재 작업 폴더 얻기
os.mkdir('생성 폴더경로 및 폴더명') # 디렉터리 생성
os.makedirs('./a/b/c', exist_ok=True)
 - mkdir, makedirs 차이
          우선 mkdir은 한 폴더만 생성이 가능합니다. ./a/b/c 와 같이 폴더 내의 폴더는 생성할 수 없습니다.<br/>
          단, 기존에 new_folder라는 폴더가 있으면 os.mkdir('./new_folder/a') 를 통해 a라는 폴더 하나를 생성할 수 있습니다.<br/>
          다만 이와 같은 경우에 new_folder 폴더가 없으면 exception 에러가 뜨게 됩니다.
          makedirs는 './a/b/c' 처럼 원하는 만큼 디렉토리를 생성할 수 있습니다.<br/>
          exist_ok라는 파라미터를 True로 하면 해당 디렉토리가 기존에 존재하면 에러발생 없이 넘어가고, 없을 경우에만 생성합니다.<br/>
          반대로, exist_ok를 True로 설정하지 않았을 때 이미 해당 디렉토리가 존재하는 경우에는 exception에러가 뜨게 됩니다.
os.remove('경로 및 파일명') # 파일 삭제

# os.path
os.path.join('C:\Tmp', 'a', 'b') # 경로를 병합하여 새 경로 생성(경로를 출력해주는 것) -> # "C:\Tmp\a\b"
os.path.getsize('경로 및 파일명') # 파일 크기 확인
os.path.exists('경로 및 파일명') # 파일 및 디렉터리 존재 유무 확인
os.path.isfile('경로 및 파일명')
os.path.isdir('경로명')



##  이진 파일 읽고 쓰기
readpic = None
writepic = None
pic = ""
## readpic = open('C:\\Users\\Adminstrator\\Desktop\\이미지\\y.jpg', 'w')
## writepic = open('C:\\Users\\Adminstrator\\Desktop\\이미지\\y.jpg', 'w')
with open('C:\\Users\\Adminstrator\\Desktop\\이미지\\y.jpg', 'rb') as readpic:
    with open('C:\\Users\\Adminstrator\\Desktop\\파이썬\\chapter12 파일 입출력\\y.jpg', 'wb') as writepic:
        while true:
            pic = readpic.read()
            if not pic:
                 break;
            writepic.write(pic)

## readpic.close()
## writepic.close()



    # 특정 파일 리스트 가져오기(listdir, glob)
# os 와 glob 에 차이점이 있다면 glob는 결과물에 디렉토리까지 같이 출력해 준다는 것이다
import os
path = './'
file_list = os.listdir(path)  # path에 있는 파일들의 이름을 리스트로 묶어 반환
print('file_list: {}'.format(file_list))
	# py 파일만 가져오려면 다음과 같은 코드 추가
	file_list_py = [file for file in file_list if file.endwith('.py')]

import glob
path = "./*"
file_list = glob.glob(path)
print ("file_list: {}".format(file_list))
	# file_list_py = [file for file in file_list if file.endswith(".py")]






---------- 객체지향 프로그래밍 OOP: Object Oriented Programming ----------


	# OOP Car Example
	# Car class - Ex 1
class Car:
    color = ""
    speed = 0

    def upSpeed(self, value):
        self.speed += value
    def downSpeed(self, value):
        self.speed -= value

# 메인 코드 부분
myCar1 = Car()
myCar1.color = 'red'
myCar.speed = 0

myCar.upSpeed(30)
print(f'자동차 1의 색상은 {myCar1.color}이며, 현재 속도는 {myCar1.speed}km 입니다.')

	# Car class - Ex 2
# 생성자: Constructor
# __init__() 또는 __init__(매개변수 리스트): 인스턴스를 생성하면서 필드값을 초기화하는 함수

class Car:
    color = ''
    speed = 0
    count = 0 # 클래스 변수

    def __init__(self):
        self.color = 'red'
        self.speed = 0
        Car.count += 1

    def __init__(self, value1, value2):
        self.color = value1
        self.speed = value2
        Car.count += 1

    def upSpeed(self, value):
        self.speed += value

    def downSpeed(self, value):
        self.speed -= value

myCar1 = Car('red', 30)
myCar1.upSpeed(30)
print(f'자동차1의 색상은 {myCar1.color}이며, 현재 속도는 {myCar1.speed}km이고, 생산된 자동차는 총 {Car.count}대입니다.')

myCar2 = Car('blue', 60)
myCar2.downSpeed(60)
print(f'자동차2의 색상은 {myCar2.color}이며, 현재 속도는 {myCar2.speed}km이고, 생산된 자동차는 총 {Car.count}대입니다.')

	# Car class - Ex 3
# 상속과 메서드 오버라이딩(상위 클래스에서 정의된 메서드를 서브 클래스에서 재정의하는 것)
class Car:
    speed = 0

    def upSpeed(self, value):
        self.speed += value
        print(f'현재 속도(슈퍼 클래스): {self.speed}')

class Sedan(Car):

    def upSpeed(self, value):
        self.speed += value
        if self.speed > 150:
            self.speed = 150
        print(f'현재 속도(서브 클래스): {self.speed}')

class Truck(Car):

    pass

truck = Truck()
sedan1 = Sedan()

print('트럭 -> ', end='')
truck.upSpeed(200)

print('승용차 ->', end='')
sedan1.upSpeed(200)





──────────────────────────────
