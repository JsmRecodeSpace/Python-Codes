

-----------------------------------------------------------Markdown----------------------------------------------------------

    # Markdown 목차
* 바로가기
* 제목
* 줄바꿈
* 목록
* 수평선
* 강조
* 블록인용
* 코드블록
* 링크
* 이미지
* 수식
* 테이블 만들기
* 텍스트 색
* 체크리스트



    # 바로가기
클릭해서 해당부분으로 바로가는 링크를 만들어준다.
두번째 괄호 안에 #(시작)과 -(글자 사이)를 꼭 넣어줘야 한다.
* [1. 제목](#1.-제목)
* [2. 줄바꿈](#2.-줄바꿈)
* [3. 목록](#3.-목록)
* [4. 수평선](#4.-수평선)
* [5. 강조](#5.-강조)
* [6. 블록인용](#6.-블록인용)
* [7. 코드블록](#7.-코드블록)
* [8. 링크](#8.-링크)
* [9. 이미지](#9.-이미지)
* [10. 수식](#10.-수식)




    # 1. 제목
마크다운에서는 #을 사용해서 6가지의 제목을 나타낸다.
h1부터 h6까지 표현할 수 있으며, # 의 개수로 표현 가능합니다
html에서 제목을 다루는 태그는 `<h1>, <h2>, <h3>, <h4>, <h5>, <h6>` 이다.

    # 마크다운 작성 시
# 첫번째 큰 제목
## 두번째 큰 제목
### 세번째 큰 제목
#### 네번째 큰 제목
##### 다섯번째 큰 제목
###### 여섯번째 큰 제목

    # =, -를 활용
=, -를 각각 2개 이상 사용하면 아래와 같이 h1, h2의 #을 대체할 수 있습니다.
여러 개를 사용해도 마찬가지예요.
큰 제목을 써줄 때는 문서 아래에 = 기호를 사용한다.
작은 제목을 써줄 때는 문서 아래에 - 기호를 사용한다.
제목
=====
부제목
------------




    # 줄바꿈
마크다운에서는 간단하게 띄어쓰기 두 번을 하고 enter을 누르면 줄을 바꿀 수 있다.
html에서는 `<br>` 태그를 이용해서 줄바꿈을 사용할 수 있다.

첫번째 문장입니다.
두번째 문장입니다.
첫 번째 문단<br/><br/>
두 번째 문단




    # 목록
숫자와 함께 쓰면 순서가 있는 목록이 되고, 글머리 기호와 함께 쓰면 순서가 없는 목록이 된다. (+, -, * 지원)
글머리 기호를 쓴 후 내용을 쓸 때 띄어쓰기를 해야한다.
+ 첫번째
- 두번째
* 세번째

목록 안의 목록을 쓰고 싶다면 tab을 한 번 해주고, 쓰면 된다.
tab을 각각 해주면 색이 바뀌는 것을 볼 수 있다.
+ 첫번째
    - 두번째
        * 세번째

- 순서 없는 목록 1
    - 목록 1.1
        - 목록 1.2
- 순서 없는 목록 2

        Tab 두번 하면 코드 블럭을 만들 수 있어요.
* 순서 없는 목록 3
+ 순서 없는 목록 4
	+ `인라인 코드 가능`
    	+ 들여쓰기(tab키 이용)를 하면 다른 모양으로 표현 됩니다.

        ```　
		블럭 코드 가능
		```　



    # 수평선
*와 -로 수평선을 만들 수 있다.
개수와 상관없이 색이 검은색에서 회색으로 바뀌는 시점까지 써주면 된다.
html에서 제목을 다루는 태그는 `<hr/>`이다.
***
* * *
---
- - -



    # 강조
굵은 글, 기울여진 글, 줄이 그어진 글 등을 사용할 수 있다.
배경색을 채우려면 `(Esc아래)를 써주면 된다.

`배경색`
__굵게__
**굵게**
_기울여 쓰기_
*기울여 쓰기*
~취소선~
~~취소선~~

html 언어로 표현하기
<em>이탤릭체</em>
<del>취소선</del>
<strike>취소선</strike>
<u>밑줄</u>
윗첨자<sup>윗첨자<sup>
아랫첨자<sub>아랫첨자</sub>




    # 블록인용
이메일에서 사용하는 `>` 블럭인용문자를 이용한다.

> 인용문 작성하기
-작성자

> 인용문 작성하기
>> (>)의 갯수에 따라
>>> 중첩문 가능


> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.



    # 코드블록
코드 블록은 ```을 사용한다.
코드 블록 안에 파이썬 코드를 작성하고자 할 때는  python을 써주면 된다.

```javascript
let sumNumbers = (firstNum, lastNum) => {
  return firstNum + lastNum;
};
sumNumbers(100, 200);
```　

```python
num_list = ['one', 'two', 'three']
for num in num_list:
  print(num)
```　



    # 링크
**자동연결**
일반적인 URL 혹은 이메일주소인 경우 적절한 형식으로 링크를 형성한다.
* 외부링크: <http://example.com/>
* 이메일링크: <address@example.com>


**외부링크**
링크를 쓸 때는 그냥 주소를 넣어도 하이퍼링크가 동작하지만, 링크 주소를 설명하고 싶다면 [] 안에 링크 설명을 쓰고 ()에 링크주소를 입력하면 된다.
[]와 () 사이에 띄어쓰기가 없어야 하며, 뒤에는 링크 이름을 적을 수 있다.
```
Link: [Google](https://google.com, "google link")
```
Link: [Google](https://google.com, "google link")

**참조링크**
링크를 참조하여 쓸 때 사용하며 참조링크를 쓸 때 들여쓰기를 해야된다.
```
Link: [Google][googlelink]

[googlelink]: https://google.com "Go google"
```
Link: [Google][googlelink]

[googlelink]: https://google.com "Go google"




    # 이미지
```
![오리 꽥꽥](/path/to/img.jpg)
```
![오리 꽥꽥](duck.jfif)

```
<img src="duck.jfif" align="left" width="450px" height="300px" title="오리 꽥꽥" alt="RubberDuck">
```
<img src="duck.jfif" align="left" width="450px" height="300px" title="오리 꽥꽥" alt="RubberDuck">

# 이미지 넣을때: src에다가 주소 넣기, width와 height로 조절 가능.
<img align="left" src="http://drive.google.com/uc?export=view&id=14tJyLDklDGVn7cndniOSdciEoMLUoGII"
 width=600 height=400>




    # 수식
수식을 쓸 때는 `$`로 감싸준다.
`$$`로 감싸주면 가운데 정렬이 된다.
문자 앞에는 `\`를 붙여준다.

Name|	Symbol|	Command
---|---|---
Alpha|	αA|      \alpha A
Beta|	βB|      \beta B
Gamma|   γΓ|      \gamma \Gamma
Delta|   δΔ|	\delta \Delta
Epsilon|	ϵE|	\epsilon E
Zeta|	ζZ|	\zeta Z
Eta|	ηE|	\eta E
Theta|	θΘ|	\theta \Theta
Iota|	ιI|	\iota I
Kappa|	κK|	\kappa K
Lambda|	λΛ|	\lambda \Lambda
Mu|	μM	\mu| M
Nu|	νN	\nu| N
Omicron|	οO|	\omicron O
Pi|	πΠ|	\pi \Pi
Rho|	ρR|	\rho R
Sigma|	σΣ|	\sigma \Sigma
Tau|	τT|	\tau T
Upsilon|	υΥ|	\upsilon \Upsilon
Phi|	ϕΦ|	\phi \Phi
Chi|	χX|	\chi X
Psi|	ψΨ|	\psi \Psi
Omega|	ωΩ|	\omega \Omega

$$x=y+10$$
$$RMSE=\sqrt {\sum_{i} (Y_{i} - \hat Y_{i})^2}$$# 수학기호를 출력하고 싶으면 $   $ 사이에 넣으면 된다.
    또는 $$  $$ 사이에 넣어 출력하면 된다.
    문자는 앞에 \를 붙여준다.
Ex1. 평균이 $\mu$, 분산이 $\sigma^2$인 정규분포를 가지는 모집단에서 추출한
      $n$개의 표본의 평균을 $\bar X$, 표본분산을 $S^2$라고 하자.
      $$ T =  \frac{(\bar X - \mu)}{S/\sqrt n}$$
      $T$의 분포는 자유도가 $n-1$인 t분포이다.
Ex2. 표본평균의 표본분포
      $$ \frac{\bar X-\mu}{\sigma/\sqrt{n}} \rightarrow N \left(0,1 \right) $$
Ex3. fig, ax = plt.subplots(figsize=(6, 4))
      for k, color in zip(df_values, colors):
          dist = t(k)
          label = r'$\mathrm{t}(n=%i)$' % k
          plt.plot(x, dist.pdf(x), c=color, label=label)
      plt.xlabel('$x$')
      plt.ylabel(r'$p(x|k)$')
      plt.title("Student's $t$ Distribution")
Ex.4 - $\alpha$: <font color="blue"> 유의수준 (significance level)</font>
      - $1-\alpha$: <font color="blue">신뢰수준 (confidence level)</font>
      - $t_{\alpha/2}$: <font color="blue">자유도 (degree of freedom)</font> n – 1을 가지는 t 분포의 오른쪽 꼬리 $\alpha/2$에 해당하는 면적에 대한  t 값
      - $s$: 표본 표준편차




      # 테이블 만들기
`|`를 구분자로 데이터 프레임을 입력하면 테이블 형태의 데이터 프레임을 만들 수 있다.
두번째 줄의 표현은 표의 정렬 기준이다.
--- 정렬하지 않음
:--- 왼쪽으로 정렬
---: 오른쪽으로 정렬
:---: 가운데 정렬
```
index | 변수1 | 변수2 | 변수3
---|---|---|---
1 | v1 | v2 | v3
2 | v11 | v22 | v33
```

index | 변수1 | 변수2 | 변수3
---|---|---|---
1 | v1 | v2 | v3
2 | v11 | v22 | v33

| | one_hot_encoding      | label |
|- | --------- | --------- |
|1 | religion   | race, age_group  |
|2 | race   | religion, age_group  |
|3 | age_group   | religion, race  |
|4 | religion, race   | age_group  |
|5 | religion,age_group   | race  |
|6 | race, age_group   | religion  |
|7 | religion, race, age_group   |   |

테이블은 아래와 같이 작성합니다.
| 로 구분하며, <4. 폰트 스타일> 에서 이야기 했던 기본적인 스타일 적용이 가능합니다.
또한 -(하이픈)으로 구분된 곳 각각 왼쪽, 양쪽, 오른쪽에 :(세미콜론)을 붙일 경우
 순서대로 왼쪽 정렬, 가운데 정렬, 오른쪽 정렬이 가능합니다.

| 드라마 제목 | 주연 배우 | 방영일 |
|:----------|:----------:|----------:|
| **호텔 델루나** | 이지은, 여진구 | ~~2019.07.13. ~ 2019.09.01.~~ |
| 타인은 지옥이다 | 임시완, 이동욱, 이현욱, 이정은 | 2019.08.31. ~ |
| 멜로가 체질 | 천우희, 안재홍, 전여빈, 공명 | 2019.08.09. ~ |




    # 텍스트 색
```
<font color = 'red'> # 빨간색 글자로 출력할 수 있다.
<font color = blue> 텍스트
<font color = "#CC3D3D"> 텍스트
```
<font color=blue>텍스트
<font color=navy> # 네이비색 글자로 출력할 수 있다.
<font color=red>텍스트
<font color=green>텍스트
<font color='darkgreen'> # 녹색
<font color=pink>텍스트
<font color=yellow>텍스트



    # 체크리스트
`- [ ]` 를 입력하고 text를 쓰면 체크박스가 만들어진다.
띄어쓰기를 꼭 해줘야 되며 체크박스가 채워지려면 대괄호 안에 x를 넣어준다.

```
- [ ] 체크박스(false)
- [x] 체크박스(true)
```
- [ ] 체크박스(false)
- [x] 체크박스(true)



────────────────────────────────────────────────────────────────────────
