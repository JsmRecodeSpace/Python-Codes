
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




    # DFS/BFS
DFS - 스택, 재귀 함수 이용
BFS - 큐, 큐 자료구조 이용

    # DFS(Depth-First Search)
 - 깊이 우선 탐색, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘
 - 특정한 경로로 탐색하다가 특정한 상황에서 최대한 깊숙이 들어가서 노드를 방문한 후,
   다시 돌아가 다른 경로로 탐색하는 알고리즘
 - 데이터의 개수가 N개인 경우 O(N)의 시간이 소요
 - DFS는 스택을 이용하는 알고리즘이기 때문에 실제 구현은 재귀 함수를 이용했을 때 매우 간결하게 구현할 수 있음


# DFS 과정 1
    - 1. 탐색 시작 노드를 스택에 삽입하고 방문 처리
    - 2. 스택의 최상단 노드에 방문하지 않은 인접 노드가 있으면 그 노드를 스택에 넣고 방문 처리를 함.
      방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼냄
    - 2의 과정을 더 이상 수행할 수 없을 때까지 반복

# DFS 메서드 정의
def dfs(graph, v, visited):
    # 현재 노드를 방문 처리
    visited[v] = True
    print(v, end=' ')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
graph = [
 [],
 [2, 3, 8],
 [1, 7],
 [1, 4, 5],
 [3, 5],
 [7],
 [2, 6, 8],
 [1, 7]
]

# 각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
visited = [False] * 9

# 정의된 DFS 함수 호출
dfs(graph, 1, visited)


# DFS 과정 2
    - 1. 특정한 지점의 주변 상,하,좌,우를 살펴본 뒤에 주변 지점 중에서 값이 '0'이면서
         아직 방문하지 않은 지점이 있다면 해당 지점을 방문한다.
    - 2. 방문한 지점에서 다시 상,하,좌,우를 살펴보면서 방문을 다시 진행하면, 연결된 모든 지점을 방문할 수 있다.
    - 3. 1~2번의 과정을 모든 노드에 반복하며 방문하지 않은 지점의 수를 센다.

# DFS로 특정한 노드를 방문한 뒤에 연결된 모든 노드들도 방문
def dfs(x, y):
  # 주어진 범위를 벗어나는 경우에는 즉시 종료
  if x <= -1 or x >= n or y <= -1 or y >= m:
    return False
  # 현재 노드를 아직 방문하지 않았다면
  if graph[x][y] == 0:
    # 해당 노드 방문 처리
    graph[x][y] = 1
    # 상, 하, 좌, 우의 위치도 모두 재귀적으로 호출
    dfs(x - 1, y)
    dfs(x, y - 1)
    dfs(x + 1, y)
    dfs(x, y + 1)
    return True
  return False

# 모든 노드(위치)에 대하여 음료수 채우기
result = 0
for i in range(n):
  for j in range(m):
    # 현재 위치에서 DFS 수행
    if dfs(i, j) == True:
      result += 1

print(result) # 정답 출력





    # BFS(Breath-First Search)
 - 너비 우선 탐색, 가까운 노드부터 탐색하는 알고리즘
 - DFS는 최대한 멀리 있는 노드를 우선으로 탐색하는 방식으로 동작한다면, BFS는 반대임
 - BFS 구현은 선입선출 방식인 큐 자료구조를 이용하는 것이 정석임.
   인접한 노드를 반복적으로 큐에 넣도록 알고리즘을 작성하면 자연스럽게 먼저 들어온 것이 먼저 나가게 되어,
   가까운 노드부터 탐색을 진행하게 됨
 - 구현에 있어 deque 라이브러리를 사용하는 것이 좋으며 탐색을 수행함에 있어 O(N)의 시간이 소요된다.
 - 일반적으로 수행 시가나은 DFS보다 좋은 편이다.



# BFS 과정 1
    - 1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다.
    - 2. 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리를 한다.
    - 3. 2번의 과정을 더 이상 수행할 수 없을 때까지 방문한다.

from collections import deque
# BFS 메서드 정의
def bfs(graph, start, visited):
  # 큐(queue) 구현을 위해 deque 라이브러리 사용
  queue = deque([start])
  # 현재 노드를 방문 처리
  visited[start] = True
  # 큐가 빌 때까지 반복
  while queue:
    # 큐에서 하나의 원소를 뽑아 출력
    v = queue.popleft()
    print(v, end=' ')
    # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
    for i in graph[v]:
      if not visited[i]:
        queue.append(i)
        visited[i] = True

# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
graph = [
 [],
 [2, 3, 8],
 [1, 7],
 [1, 4, 5],
 [3, 5],
 [3, 4],
 [7],
 [2, 6, 8],
 [1, 7]
]

# 각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
visited = [False] * 9

# 정의된 DFS 함수 호출
bfs(graph, 1, visited)



# BFS 과정 2
    - 맨 처음 (1, 1)의 위치에서 시작하며, (1, 1)의 값은 항상 1이라고 문제에서 언급
    - (1, 1) 좌표에서 상,하,좌,우로 탐색을 진행하면 바로 옆 노드인 (1,2) 위치의 노드를 방문하게 되고 새롭게 방문하는 (1, 2) 노드의 값을 2로 바꾸게 됨.
    - 마찬가지로 BFS를 계속 수행하면 결과적으로 다음과 같이 최단 경로의 값들이 1씩 증가하는 형태로 변경됨.

from collections import deque

# 이동할 네 방향 정의(상, 하, 좌, 우)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# BFS 소스코드 구현
def bfs(x, y):
  # 큐(Queue) 구현을 위해 deque 라이브러리 사용
  queue = deque()
  queue.append((x, y))
  # 큐가 빌 때까지 반복
  while queue:
    x, y = queue.popleft()
    # 현재 위치에서 네 방향으로의 위치 확인
    for i in range(4):
      nx = x + dx[i]
      ny = y + dy[i]
      # 미로 찾기 공간을 벗어난 경우 무시
      if nx < 0 or ny < 0  or nx >= n or ny >= m:
        continue
      # 벽인 경우 무시
      if graph[nx][ny] == 0:
        continue
      # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
      if graph[nx][ny] == 1:
        graph[nx][ny] = graph[x][y] + 1
        queue.append((nx, ny))
  # 가장 오른쪽 아래까지의 최단 거리 반환
  return graph[n - 1][m - 1]

# BFS를 수행한 결과 출력
print(bfs(0, 0))






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
