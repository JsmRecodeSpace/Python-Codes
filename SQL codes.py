
    # SQL 쿼리
SELECT (*), (COUNT, MIN, MAX), IFNULL(), DISTINCT, DATE_FORMAT, (LEFT, MID, RIGHT)
FROM (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN) (ON, USING) (SUB QUERY)[ALIAS]
WHERE (IS, IS NOT, =, !=), (IN, NOT IN) (조건절) (BETWEEN AND)
GROUP BY (1, 2,),
HAVING (COUNT) (DISTINCT COL) (> ,>=, <, <=)               # GROUP BY에서 조건을 줄 때는 HAVING을 사용하여 조건을 줌
ORDER BY (DESC) (COL1 - + * / COL2)
LIMIT



    # WHERE 절 함수
LOWER()
UPPER()
LIKE
 - 'JSM'으로 시작하는  데이터 검색 : LIKE 'JSM%'
   'JSM'으로 끝나는 데이터 검색: LIKE '%JSM'
   'JSM'이 들어가는 데이터 검색: LIKE '%JSM%'




    # 시간 관련 함수
# http://happycgi.com/community/bbs_detail.php?bbs_num=43&tb=board_man_story
- DAYOFWEEK(date) : 해당 날짜의 요일을 숫자로 반환한다. 일요일은 1, 토요일은 7 이다.
    select DAYOFWEEK('1998-02-03');
- WEEKDAY(date) : 해당 날짜에 대한 요일을 반환한다. 월요일은 0, 일요일은 6 이다.
    select WEEKDAY('1997-10-04 22:23:00');
- DAYOFYEAR(date) : 해당 날짜의 1월 1일부터의 날수를 반환한다. 결과값은 1에서 366 까지이다.
    select DAYOFYEAR('1998-02-03');
- YEAR(date) : 해당 날짜의 년을 반환한다.
    select YEAR('98-02-03');
- MONTH(date) : 해당 날짜의 월을 반환한다.
    select MONTH('1998-02-03');
- DAYOFMONTH(date) : 해당 날짜의 일을 반환한다. 결과값은 1 에서 31 까지이다.
    select DAYOFMONTH('1998-02-03');
- HOUR(time) : 해당날짜의 시간을 반환한다. 결과값은 0 에서 23 이다.
    select HOUR('10:05:03');
- MINUTE(time) : 해당날짜의 분을 반환한다. 결과값은 0 에서 59 이다.
    select MINUTE('98-02-03 10:05:03');
- SECOND(time) : 해당날짜의 초를 반환한다. 결과값은 0 에서 59 이다.
    select SECOND('10:05:03');
- DAYNAME(date) : 해당 날짜의 요일 이름을 반환한다. 일요일은 'Sunday' 이다.
    select DAYNAME("1998-02-05");
- MONTHNAME(date) : 해당 날짜의 월 이름을 반환한다. 2월은 'February' 이다.
    select MONTHNAME("1998-02-05");
- QUARTER(date) : 해당 날짜의 분기를 반환한다. 결과값은 1 에서 4 이다.
- WEEK(date,first) : 1월 1일부터 해당날가지의 주 수를 반환한다. 주의 시작을 일요일부터 할경우는 두번째 인자를 0, 월요일부터 시작할 경우는 1 을 넣는다. 결과값은 1 에서 52 이다.
    select WEEK('1998-02-20',1);
- PERIOD_ADD(P,N) : P (형식은 YYMM 또는 YYYYMM 이어야 한다.) 에 N 만큼의 달 수를 더한값을 반환한다. 주의할것은 두번째 인자는 숫자라는 것이다.
    select PERIOD_ADD(9801,2);
- PERIOD_DIFF(P1,P2) : 두개의 인자 사이의 달 수를 반환한다. 두개의 인자 모두 형식은 YYMM 또는 YYYYMM 이어야 한다.


- DATE_FORMAT(date,format) : 날짜를 해당 형식의 문자열로 변환하여 반환한다.
형식은 다음과 같다.
%M (달 이름), %W (요일 이름), %Y (YYYY 형식의 년도), %y (YY 형식의 년도),
%a (요일 이름의 약자), %d (DD 형식의 날짜), %e (D 형식의 날짜),
%m (MM 형식의 날짜), %c (M 형식의 날짜), %H (HH 형식의 시간, 24시간 형식),
%k (H 형식의 시간, 24시간 형식), %h (HH 형식의 시간, 12시간 형식), %i (MM 형식의 분), %p (AM 또는 PM)
예 : select DATE_FORMAT('1997-10-04 22:23:00', '%W %M %Y');
select DATE_FORMAT('1997-10-04 22:23:00', '%H:%i:%s');
select DATE_FORMAT('1997-10-04 22:23:00', '%D %y %a %d %m %b %j');
select DATE_FORMAT('1997-10-04 22:23:00', '%H %k %I %r %T %S %w');
SELECT ANIMAL_ID, NAME, DATE_FORMAT(DATETIME, '%Y-%m-%d')




    # 조건절
# IFNULL
SELECT IFNULL(NAME, 'NO NAME') FROM NAMES
 - 이름이 없는 경우 NO NAME 출력




     # CASE WHEN THEN ELSE END
# CASE - Ex 1
SELECT
    INS.ANIMAL_TYPE,
    CASE
        WHEN INS.NAME IS NULL THEN "No name"
        ELSE NAME
    END AS NAME,
    INS.SEX_UPON_INTAKE
FROM ANIMAL_INS INS



    # IF
# IF - Ex 1
SELECT animal_type, if(name is null,"No name",name), sex_upon_intake
from animal_ins
order by animal_id;



    # 문자열 부분 가져오기 (LEFT, MID, RIGHT)
# https://extbrain.tistory.com/62
 - LEFT : 문자에 왼쪽을 기준으로 일정 갯수를 가져오는 함수.
 - MID : 문자에 지정한 시작 위치를 기준으로 일정 갯수를 가져오는 함수.
 - RIGHT : 문자에 오른쪽을 기준으로 일정 갯수를 가져오는 함수.
LEFT(문자, 가져올 갯수);
MID(문자, 시작 위치, 가져올 갯수);
-- 또는 SUBSTR(문자, 시작 위치, 가져올 갯수);
-- 또는 SUBSTRING(문자, 시작 위치, 가져올 갯수);
RIGHT(문자, 가져올 갯수);



    # JOIN
FROM TABLE A LEFT JOIN TABLE B ON A.KEY = B.KEY
 - 교집합 + A에만 있는 부분

FROM TABLE A LEFT JOIN TABLE B ON A.KEY = B.KEY
WHERE B.KEY IS NULL
 - B와 공유하는 교집합을 제외하고 A에만 있는 부분

FROM  TABLE B RIGHT JOIN TALBE B ON A.KEY = B.Key
 - 교집합 + B에만 있는 부분

FROM  TABLE B RIGHT JOIN TALBE B ON A.KEY = B.Key
WHERE A.KEY IS NULL
 - A와 공유하는 교집합을 제외하고 B에만 있는 부분

FROM TABLE A FULL OUTER JOIN TABLE B ON A.KEY = B.KEY
 - 합집합

FROM TABLE A FULL OUTER JOIN TABLE B ON A.KEY = B.KEY
WHERE A.KEY IS NULL OR B.KEY IS NULL
 - 합집합 중 A와 B가 공유하는 교집합을 제외한 부분



    # USING
FROM ANIMAL_INS A RIGHT JOIN ANIMAL_OUTS B ON A.ANIMAL_ID = B.ANIMAL_ID
 = FROM ANIMAL_INS A RIGHT JOIN ANIMAL_OUTS B USING ANIMAL_ID



    # SUBQUERY(서브쿼리)
 - FROM 절에 서브쿼리 날릴 때 MYSQL에서는 ALIAS를 꼭(MUST) 적어주어야 함, 아니면 에러
SELECT HOUR, COUNT(HOUR) AS COUNT
FROM (
    SELECT ANIMAL_ID AS ANIMAL_ID, MID(DATETIME, 12, 2) AS HOUR
    FROM ANIMAL_OUTS
    ) SUB  # <- 이렇게 SUB 별명
GROUP BY HOUR
ORDER BY HOUR ASC;



    # WITH RECURSIVE (CTE)
# https://velog.io/@cyanred9/SQL-Recursive
 - with로 임시 테이블 CTE(Common Table Expression) 생성
 - 초기 설정값과 recursive할 쿼리를 union all로 엮음
WITH recursive CTE as( #재귀쿼리 세팅
    select 0 as HOUR #초기값 설정
    union all #위 쿼리와 아래 쿼리의 값을 연산
    select HOUR+1 from CTE #하나씩 불려 나감
    where HOUR < 23 #반복을 멈추는 용도
)

 ────────────────────────────────────────────────────────────────────────
