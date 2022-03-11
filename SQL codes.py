
    # SQL 쿼리
SELECT (*), (COUNT, MIN, MAX), IFNULL()
FROM
WHERE (IS, IS NOT, =, !=), (IN, )
GROUP BY (1, 2,),
HAVING (> ,>=, <, <=)               # GROUP BY에서 조건을 줄 때는 HAVING을 사용하여 조건을 줌
ORDER BY (DESC)
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








 ────────────────────────────────────────────────────────────────────────
