
--------------- PYTHON ---------------

    # 좌표, 위치, 공간 관련

# 코딩 시 사용 변수명
 (x, y), (dx, dy), (nx, ny)
 (array, map_, visit)


# 왼쪽으로 회전
def turn_left():
    global direction
    direction -= 1
    if direction == -1:
        direction = 3


# 왼쪽으로 회전 후 움직일 방향 정의
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]







--------------- SQL ---------------


    # 동시에 담긴 것 찾기
SELECT CART_ID
FROM CART_PRODUCTS
WHERE NAME IN ('Milk', 'Yogurt') # IN으로 담겨 있는 것 모두 찾음(OR 연산자도 가능)
GROUP BY CART_ID
HAVING COUNT(DISTINCT NAME) = 2 # DISTINCT로 종류 세기



    # WITH RECURSIVE로 테이블 생성하여 엮는 방법
# https://velog.io/@cyanred9/SQL-Recursive
 - with로 임시 테이블 CTE(Common Table Expression) 생성
 - 초기 설정값과 recursive할 쿼리를 union all로 엮음
WITH recursive CTE as( #재귀쿼리 세팅
    select 0 as HOUR #초기값 설정
    union all #위 쿼리와 아래 쿼리의 값을 연산
    select HOUR+1 from CTE #하나씩 불려 나감
    where HOUR < 23 #반복을 멈추는 용도
)

SELECT CTE.HOUR,COUNT(HOUR(OUTS.DATETIME)) AS COUNT
FROM CTE LEFT JOIN ANIMAL_OUTS AS OUTS ON CTE.HOUR = HOUR(OUTS.DATETIME)
GROUP BY CTE.HOUR #안해주면 COUNT가 전체 COUNT가 되어버린다!


────────────────────────────────────────────────────────────
