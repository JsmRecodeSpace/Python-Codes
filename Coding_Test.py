
--------------- PYTHON ---------------



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


# DFS 과정 2 (미로탈출, start에서 끝 부분 까지의 최단 거리 찾기)
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




# DFS 과정 3 (연구소 - 바이러스 퍼짐을 DFS로 구현, 벽 세우는 것은 완전 탐색, DFS후 안전영역 카운트)
temp =[[0] * m for _ in range(n)] # 벽을 설치한 뒤의 맵 리스트

# 4가지 이동 방향에 대한 리스트
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

result = 0

# 깊이 우선 탐색(DFS)를 이용해 각 바이러스가 사방으로 퍼지도록 하기
def virus(x, y):
  for i in range(4):
    nx = x + dx[i]
    ny = y + dy[i]
    # 상, 하, 좌, 우 중에서 바이러스가 퍼질 수 있는 경우
    if nx >= 0 and nx < n and ny >= 0 and ny < m:
      if temp[nx][ny] == 0:
        # 해당 위치에 바이러스 배치하고, 다시 재귀적으로 수행
        temp[nx][ny] = 2
        virus(nx, ny)

# 현재 맵에서 안전 영역의 크기를 계산하는 메서드
def get_score():
  score = 0
  for i in range(n):
    for j in range(m):
      if temp[i][j] == 0:
        score += 1
  return score

# 깊이 우선 탐색(DFS)를 이용해 울타리를 설치하면서, 매번 안전 영역의 크기 계산
def dfs(count):
  global result
  # 울타리가 3개 설치된 경우
  if count == 3:
    for i in range(n):
      for j in range(m):
        temp[i][j] = data[i][j]
    # 각 바이러스의 위치에서 전파 진행
    for i in range(n):
      for j in range(m):
        if temp[i][j] == 2:
          virus(i, j)
    # 안전 영역의 최댓값 계산
    result = max(result, get_score())
    return
  # 빈 공간에 울타리 설치
  for i in range(n):
    for j in range(m):
      if data[i][j] == 0:
        data[i][j] = 1
        count += 1
        dfs(count)
        data[i][j] = 0
        count -= 1

dfs(0)
print(result)









    # BFS(Breath-First Search)
 - 너비 우선 탐색, 가까운 노드부터 탐색하는 알고리즘
 - DFS는 최대한 멀리 있는 노드를 우선으로 탐색하는 방식으로 동작한다면, BFS는 반대임
 - BFS 구현은 선입선출 방식인 큐 자료구조를 이용하는 것이 정석임.
   인접한 노드를 반복적으로 큐에 넣도록 알고리즘을 작성하면 자연스럽게 먼저 들어온 것이 먼저 나가게 되어,
   가까운 노드부터 탐색을 진행하게 됨
 - 구현에 있어 deque 라이브러리를 사용하는 것이 좋으며 탐색을 수행함에 있어 O(N)의 시간이 소요된다.
 - 일반적으로 수행 시가나은 DFS보다 좋은 편이다.
 - ★ '모든 도로의 거리는 1'이라는 조건, '모든 간선의 비용이 동일'할 때는 우선 너비 탐색(BFS)을 이용하여 최단 거리를 찾을 수 있다.


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



# BFS 과정 2 (음료수 얼려 먹기, 0으로 된 부분의 영역 갯수 카운트)
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



# BFS 과정 3 (특정 거리의 도시 찾기 - 최단 거리만큼에 연결되어 있는 도시 찾기)
 - 모든 도로의 거리가 1이라는 조건이 있을 때, 간선의 비용이 동일할 때는 너비 우선 탐색을 이용하여 최단 거리를 찾을 수 있다
 from collections import deque

graph = [[] for _ in range(n + 1)]
for i in range(m):
  graph[data[i][0]].append(data[i][1])

# 모든 도시에 대한 최단 거리 초기화
distance = [-1] * (n + 1)
distance[x] = 0 # 출발 도시까지의 거리는 0으로 설정

# 너비 우선 탐색(BFS) 수행
q = deque([x])
while q:
  now = q.popleft()
  # 현재 도시에서 이동할 수 있는 모든 도시를 확인
  for next_node in graph[now]:
    # 아직 방문하지 않은 도시라면
    if distance[next_node] == -1:
      # 최단 거리 갱신
      distance[next_node] = distance[now] + 1
      q.append(next_node)



# BFS 과정 4 (경쟁적 전염 - 초마다 정해진 순서의 지점별로 BFS 1칸씩 실시하여 영역 넓혀나감)
from collections import deque

n, k = map(int, input().split())

graph = [] # 전체 보드 정보를 담는 리스트
data = [] # 바이러스에 대한 정보를 담는 리스트

for i in range(n):
  # 보드 정보를 한 줄 단위로 입력
  graph.append(list(map(int, input().split())))
  for j in range(n):
    # 해당 위치에 바이러스가 존재하는 경우
    if graph[i][j] != 0:
      # (바이러스 종류, 시간, 위치 X, 위치 Y) 삽입
      data.append((graph[i][j], 0, i, j))

# 정렬 이후에 큐로 옮기기(낮은 번호의 바이러스가 먼저 증식하므로)
data.sort()
q = deque(data)

target_s, target_x, target_y = map(int, input().split())

# 바이러스가 퍼져나갈 수 있는 4가지 위치
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

# 너비 우선 탐색(BFS) 진행
while q:
  virus, s, x, y = q.popleft()
  # 정확히 s초가 지나거나, 큐가 빌 때까지 반복
  if s == target_s:
    break
  # 현재 노드에서 주변 4가지 위치를 각각 확인
  for i in range(4):
    nx = x + dx[i]
    ny = y + dy[i]
    # 해당 위로 이동할 수 있는 경우
    if 0 < nx and nx < n  and 0 <= nx and ny < n:
      # 아직 방문하지 않은 위치라면, 그 위치에 바이러스 넣기
      if graph[nx][ny] == 0:
        graph[nx][ny] = virus
        q.append((virus, s + 1, nx, ny))

print(graph[target_x -1][target_y - 1])




    # 정렬(Sort)
# 코테에서의  정렬 출제 유형
 - 문제에서 별도의 요구가 없다면 기본 정렬 라이브러리,
   데이터의 범위가 한정되어 있으며 더 빠르게 동작해야 할 때는 계수 정렬을 이용
1. 정럴 라이브러리로 풀 수 있는 문제: 단순히 정렬 기법을 알고 있는지 물어보는 문제로,
   기본 정렬 라이브러리의 사용 방법을 숙지하고 있으면 어렵지 않게 풀 수 있음
2. 정렬 알고리즘의 원리에 대해서 물어보는 문제: 선택 정렬, 삽입 정렬, 퀵 정렬 등의
   원리를 알고 있어야 문제를 풀 수 있음
3. 더 빠른 정렬이 필요한 문제: 퀵 정렬 기반의 정렬 기법으로는 풀 수 없으며 계수 정렬 등의
  다른 정렬 알고리즘을 이용하거나 문제에서 기존에 알려진 알고리즘의 구조적인 개선을 거쳐야 풀 수 있음


# 계수 정렬
 - 계수 정렬(Count Sort) 알고리즘은 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠른 정렬 알고리즘이다
 - 계수 정렬은 최악의 경우에도 수행 시간 O(N + K)를 보장한다.
 - 계수 정렬은 '데이터의 크기 범위가 제한 되어 정수 형태로 표현할 수 있을 때'만 사용할 수 있다.
 - 일반적으로 큰 데이터와 가장 작은 데이터의 차이가 1,000,000을 넘지 않을 때 효과적으로 사용가능
 - 일반적으로 별도의 리스트를 선언하고 그 안에 정렬에 대한 정보를 담는다는 특성이 있음
 - 현존하는 정렬 알고리즘 중에서 기수 정렬과 더불어 가장 빠름

# 모든 원소의 값이 0보다 크거나 같다고 가정
array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
# 모든 범위를 포함하는 리스트 선언(모든 값으로 0으로 초기화)
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)): # 리스트에 기록된 정렬 정보 확인
    for j in range(count[i]):
        print(i, end=' ') # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력


# Sorted 함수를 활용한 정렬: 키 값으로 여러 개를 두어 계산할 수 있다.
names = sorted(infos, key=lambda info : (-int(info[1]),
              int/(info[2]), -int(info[3]), info[0]))


# 거리 합을 최소로 하는 중간값(median)
 - 정렬 하면 중간값에서 array안에 있는 모든 값까지의 거리가 최소가 된다.
 - array에서 모든 값까지의 최소가 되는 값을 구할 때 중앙값을 구하는 lst[(n-1) // 2]을 활용





        # 이진 탐색 (Binary Search)
 - 내열 내부의 데이터가 정렬되어 있어야만 사용가능
 - 이진 탐색은 위치를 나타내는 변수 3개를 사용하는데 탐색하고자 하는 범위의 시작점, 끝점, 그리고 중간점이다.
 - 찾으려는 데이터와 중간점 위치에 있는 데이터를 반복적으로 비교해서 원하는 데이터를 찾음
 - 시간 복잡도가 O(log(N))
 - 처리해야 할 데이터의 개수나 값이 1,000만 단위 이상으로 넘어가면 이진 탐진 탐색과 같이 O(log(N))의
   속도를 내야 하는 알고리즘을 떠올려야 문제를 풀 수 있는 경우가 많다.


# 이진 탐색 소스코드 구현 (재귀 함수)
def binary_search(array, target, start, end):
    if start > end:
        return None
    mid  = (start + end) // 2
    # 찾은 경우 중간점 인덱스 반환
    if array[mid] == target:
        return mid
    # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
    elif array[mid] > target:
        return binary_search(array, target, start, mid - 1)
    # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
    else:
        return binary_search(array, target, mid + 1, end)


# 이진 탐색 소스코드 구현 (반복문)
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        # 찾은 경우 중간점 인덱스 반환
        if array[mid] == target:
            return mid
        # 중간점의 값보다 찾고자 하는 값이 적은 경우 왼쪽 확인
        elif array[mid] > target:
            end = mid - 1
        # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
        else:
            start = mid + 1
    return None


# 이진 탐색 과정1
# 이진 탐색을 위한 시작점과 끝점 설정
start = 0
end = max(array)

# 이진 탐색 수행(반복적)
result = 0
while(start <= end):
  total = 0
  mid = (start + end) // 2
  for x in array:
    # 잘랐을 때의 떡의 양 계산
    if x > mid:
      total += x - mid
  # 떡의 양이 부족한 경우 더 많이 자르기(왼쪽 부분 탐색)
  if total < m:
    end = mid - 1
  # 떡의 양이 충분한 경우 덜 자르기(오른쪽 부분 탐색)
  else:
    result = mid # 최대한 덜 잘랏을 때가 정답이므로, 여기에서 result에 기록
    start = mid + 1

print(result)



# 이진 탐색 과정2 (bisect 모듈 활용, 특정 원소개의 개수 세기)
 - 파이썬의 이진 탐색 라이브러리인 bisect을 적절히 활용하면
   정렬된 수열에서 특정한 값을 가지는 원소의 개수를 손쉽게 해결할 수 있다

from bisect import bisect_right, bisect_left

# 값이 [left_value, right_value]인 데이터의 개수를 반환하는 함수
def count_by_range(array, left_value, right_value):
  right_index = bisect_right(array, right_value)
  left_index = bisect_left(array, left_value)
  return right_index - left_index

# 값이 [x, x] 범위에 있는 데이터의 개수 계산
count = count_by_range(array, x, x)

# 값이 x인 원소가 존재하지 않는다면
if count == 0:
  print(-1)
# 값이 x인 원소가 존재한다면
else:
  print(count)




    # heapq 모듈
# https://www.daleseo.com/python-heapq/
 - 이진 트리(binary tree)기반의 최소 힙(min heap) 자료구조를 제공
 - 보통 리스트를 마치 힙처럼 다룰 수 있도록 도와줌
 - min heap을 사용하면 원소들이 항상 정렬된 상태로 추가되고 삭제되며,
   min heap에서 가장 작은 값은 언제나 인덱스 0, 즉, 이진 트리의 루트에 위치함.
 - 내부적으로 min heap 내의 원소(k)는 항상 자식 원소들(2k+1, 2k+2)보다 크기가 작거나 같도록 원소가 추가되고 삭제됨.


# python에서 힙 적용
 - import heapq # priority queue의 성질을 가지고 있어서 q가 붙음
 - heapq.heapify(L) # 리스트 L로부터 min heap 구성
 - m = heapq.heappop(L) # min heap L에서 최소값 삭제 (반환), 맨 앞에 껏 pop함
 - heapq.heappush(L, x) # min heap L에 원소 x 삽입
 - heap[0] # 힙에서 최소값을 삭제하지 않고 접근
 - 주의사항: 인덱스 0에 가장 작은 원소가 있다고 해서, 인덱스 1에 두번째 작은 원소,
   인덱스 2에 세번째 작은 원소가 있다는 보장은 없음. heappop() 함수를 호출하여 원소를 삭제할 때마다
   이진 트리의 재배치를 통해 매번 새로운 최소값을 인덱스0에 위치시키기 때문
   따라서 두번째 작은 원소를 얻으려면 바로 heap[1]으로 접근하면 안되고,
   반드시 heappop()을 통해 가장 작은 원소를 삭제 후에 heap[0]를 통해 새로운 최소값에 접근해야 함.


# 최대 힙
# https://www.daleseo.com/python-heapq/
 - heapq는 최소 힙(min heap)을 기능만으로 동작하기 때문에 최대 힙(max heap)으로 활용하기 위해서는 약간의 요령이 필요함
 - 바로 힙에 튜플(tuple)를 원소로 추가하거나 삭제하면, 튜플내에서 맨 앞에 있는 값을 기준으로 최소 힙의 구성되는 원리를 이용하는 것
 - 따라서 최대 힙을 만들려면 각 값에 대한 우선 순위를 구한 후, (우선 순위, 값) 구조의 튜플(tuple)을 힙에 추가하거나 삭제하면 됨.
 - 그리고 힙에서 값을 읽어올 때는 각 튜플에서 인덱스 1에 있는 값을 취하면 됨. (우선 순위에는 관심이 없으므로)

```python
import heapq

nums = [4, 1, 7, 3, 8, 5]
heap = []

for num in nums:
  heapq.heappush(heap, (-num, num))  # (우선 순위, 값)

while heap:
  print(heapq.heappop(heap)[1])  # index 1
```



    # 최단 경로
 - 가장 짧은 경로는 찾는 알고리즘
 - 상황에 맞는 효율적인 알고리즘이 이미 정립되어 있음
    - '한 지점에서 다른 특정 지점까지의 최단 경로를 구해야 하는 경우'
    - '모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야 하는 경우'


#### 1. 다익스트라(Dijkstra) 최단 경로 알고리즘
 - 그래프에서 여러 개의 노드가 있을 때,
   특정한 노드에서 출발하여 다른 노드로 가는 각각의 최단 경로를 구해주는 알고리즘
 - 매번 '가장 비용이 적은 노드'를 선택해서 임의의 과정을 반복하여 그리디 알고리즘으로 분류됨
 - 원리
     1. 출발 노드를 설정한다.
     2. 최단 거리 테이블을 초기화한다.
     3. 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택한다.
     4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신한다.
     5. 위 과정에서 3번과 4번을 반복한다.
 - '각 노드에 대한 현재까지의 최단 거리' 정보를 항상 1차원 리스트에 저장하여 리스트를 계속 갱신한다는 특징이 있다.
 - 구현 방법
    - 방법1. 구현하기 쉽지만 느리게 동작하는 코드
    - 방법2. 구현하기 조금 까다롭지만 빠르게 동작하는 코드★


** 방법 1. 간단한 다익스트라 알고리즘 소스코드 **
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기★
graph = [[] for i in range(n + 1)]
# 방문한 적이 있는지 체크하는 목적의 리스트 만들기★
visited = [False] * (n + 1)
# 최단 거리 테이블을 모두 무한으로 초기화★
distance = [INF] * (n + 1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))


# 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
def get_smallest_node():
    min_value = INF
    index = 0 # 가장 최단 거리가 짧은 노드(인덱스)
    for i in range(1, n + 1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index


def dijkstra(start):
    # 시작 노드에 대해서 초기화
    distance[start] = 0 # 자기 자신에게 가는 거리 = 0
    visited[start] = True # 자기 자신은 방문처리
    for j in graph[start]: # start노드에서 연결된 다른 노드까지 가는 거리가 담긴 리스트
        distance[j[0]] = j[1] # j = [(2, 2), (3, 5), (4, 1)] -> j[0]은 노드번호, j[1]은 노드까지의 거리
			        # start에서 연결된 노드들까지의 거리를 distance에 초기화 해준 것
    # 시작 노드를 제외한 전체 n - 1개의 노드에 대해 반복
    for i in range(n - 1):
        # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
        now = get_smallest_node()
        visited[now] = True
        # 현재 노드와 연결된 다른 노드를 확인
        for j in graph[now]: # j = [(3, 3), (5, 1)]
            cost = distance[now] + j[1] # 1에서 now까지의 거리 + now에서 연결된 다른 노드까지의 거리
	    # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[j[0]]: # 새로 구한 거리 cost가 distance에 저장된 거리보다 작은 경우
                distance[j[0]] = cost # 새로 구한 거리가 더 짧으므로 갱신

# 다익스트라 알고리즘 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n + 1):
    # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
    if distance[i] == INF:
        print("INFINITY")
    # 도달할 수 있는 경우 거리를 출력
    else:
        print(distance[i])



** 방법 2. 개선된 다익스트라 알고리즘 소스코드 **
 - 시간 복잡도 O(Elog(V))를 보장하여 해결, V는 노드의 개수, E는 간선의 개수를 의미
 - 힙 자료구조를 이용
 - '현재 가장 가까운 노드를 저장하기 위한 목적'으로 우선순위 큐를 추가로 이용
 - '최단 거리가 가장 짧은 노드'를 선택하는 과정을 다익스트라 최단 경로 함수 안에서 우선순위 큐를 이용하여 대체
 - 튜플의 첫 번째 원소인 '거리' 정보를 기준으로 해서 우선순위 큐를 구성


import heapq
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기★
graph = [[] for i in range(n + 1)]
# 최단 거리 테이블을 모두 무한으로 초기화★
distance = [INF] * (n + 1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))


def dijkstra(start):
  q = []
  # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
  heapq.heappush(q, (0, start))
  distance[start] = 0
  while q: # 큐가 비어있지 않다면
    # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
    dist, now = heapq.heappop(q)
    # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
    if distance[now] < dist:  # 이미 방문하여 distance[now]가 최솟값을 가져서 후에 재방문 할 필요가 없음
      continue		       # 힙에 넣어지면서, 거리에 따라 순위가 재조정되는 과정에서 이미 앞에서 방문할 수 있기 때문
    # 현재 노드와 연결된 다른 인접한 노드들을 확인
    for i in graph[now]:
      cost = dist + i[1]
      # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
      if cost < distance[i[0]]:
        distance[i[0]] = cost
        heapq.heappush(q, (cost, i[0]))

# 다익스트라 알고리즘을 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n + 1):
  # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
  if distance[i] == INF:
    print("INFINITY")
  # 도달할 수 있는 경우 거리를 출력
  else:
    print(distance[i])



# 다익스트라 과정 - 01
import heapq
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수 및 간선의 개수, 시작지점을 입력받기
n, m, start = map(int, input().split())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담고 있는 리스트를 만들기
graph = [[] for i in range(n + 1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n + 1)

# 모든 간선 정보를 입력받기
for _ in range(m):
  x, y, z = map(int, input().split())
  # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
  graph[x].append((y, z))


def dijkstra(start):
  q = []
  # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
  heapq.heappush(q, (0, start))
  distance[start] = 0
  while q: # 큐가 비어 있지 않다면
    # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
    dist, now = heapq.heappop(q)
    # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
    if distance[now] < dist:
      continue
    # 현재 노드와 연결된 다른 인접한 노드들을 확인
    for i in graph[now]:
      cost = dist + i[1]
      # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
      if cost < distance[i[0]]:
        distance[i[0]] = cost
        heapq.heappush(q, (cost, i[0]))

dijkstra(start)

# 도달할 수 있는 노드의 개수
count = 0
# 도달할 수 있는 노드 중에서, 가장 멀리 있는 노드와의 최단 거리
max_distance = 0

for d in distance:
  # 도달할 수 있는 노드의 경우
  if d != INF:
    count += 1
    max_distance = max(max_distance, d)

# 시작 노드는 제외해야 하므로 count -1을 출력
print(count - 1, max_distance)



#### 2. 플로이드 워셜 알고리즘
 - '모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야 하는 경우'
 - 총 복잡도는 O(N^3)이다.
 - 플로이드 워셜 알고리즘은 2차원 리스트에 '최단 거리' 정보를 저장한다는 특징.
   모든 노드에 대하여 다른 모든 노드로 가는 최단 거리 정보를 담아야 하기 때문.
 - 다이나믹 프로그래밍이라는 특징을 가지고 있음.
   노드의 개수가 N이라고 할 때, N번 만큼의 단계를 반복하며 '점화식에 맞게' 2차원 리스트를 갱신하기 때문
 - K번에 단계에 대한 점화식은 다음과 같음
    D_ab = min(D_ab, D_ak + D_kb)
     - 'A에서 B로 가는 최소비용'과 'A에서 K를 거쳐 B로 가는 비용'을 비교하여 더 작은 값으로 갱신
     - 즉, '바로 이동하는 거리'가 '특정한 노드를 거쳐서 이동하는 거리'보다 더 많은 비용을 가진다면
       이를 더 짧은 것으로 갱신한다는 것

** 플로이드 워셜 알고리즘 소스코드 **
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수 및 간선의 개수를 입력받기
n = int(input())
m = int(input())
# 2차원 리스트(그래프 표현)을 만들고, 모든 값을 무한으로 초기화
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n + 1):
    for b in range(1, n + 1):
        if a==b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력받아, 그 값으로 초기화
for _ in range(m):
    # A에서 B로 가는 비용은 C라고 설정
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n+1):
    for a in range(1, n+1):
        for b in range(1, n+1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행된 결과를 출력
for a in range(1, n+1):
    for b in range(1, n+1):
        # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
        if graph[a][b] == INF:
            print("INFINITY", end=" ")
        else:
            print(graph[a][b], end=" ")
    print()



# 플로이드 워셜 과정 - 01
# 미래 도시
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수 및 간선의 개수를 입력받기
n = int(input())
m = int(input())

# 2차원 리스트(그래프 표현)을 만들고, 모든 값을 무한으로 초기화
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n + 1):
    for b in range(1, n + 1):
        if a==b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력받아, 그 값으로 초기화
for _ in range(m):
    # A에서 B로 가는 비용은 1이라고 설정
    a, b = map(int, input().split())
    graph[a][b] = 1
    graph[b][a] = 1 # 양방향으로 이동할 수 있다고 했기 때문

# 거쳐 갈 노드 X와 최종 목적지 노드 K를 입력받기
x = int(input())
k = int(input())

# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n+1):
    for a in range(1, n+1):
        for b in range(1, n+1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행된 결과를 출력
if graph[1][k] == INF: # A회사에서 K까지 가는 최소 거리
  print(-1)
else:
  print(graph[1][k] + graph[k][x]) # A회사에서 K를 거친 후 X까지 가는 최소거리



    # 그래프 이론
 - '서로 다른 개체(혹은 객체)가 연결되어 있다'는 이야기를 들으면 가장 먼저 그래프 알고리즘을 떠올려야 함
   EX: '여러 개의 도시가 연결되어 있다'
 - 그래프의 구현 방법은 2가지 방식이 존재
    1. 인접 행렬(Adjacency Matrix): 2차원 배열을 사용하는 방식
	- 플로이드 워셜 알고리즘
    2. 인접 리스트(Adjacency List): 리스트를 사용하는 방식
	- 다익스트라 최단 경로 알고리즘
 - 어떤 문제든 메모리와 시간을 염두에 두고 알고리즘을 선택해서 구현해야 함
	- 노드의 개수가 적은 경우 -> 플로이드 워셜
	- 노드, 간선의 개수가 많은 경우 -> 다익스트라


#### 1. 서로소 집합
 - 서로소 집합: 공통 원소가 없는 두 집합
 - 서로소 집합 자료구조: 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
 - 서로소 집합 자료구조는 union과 find 2개의 연산으로 조작 가능
	- union: 2개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산
	- find: 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산
 - 트리를 이용해 서로소 집합을 계산하는 알고리즘
    1. union 연산을 확인하여, 서로 연결된 두 노드 A, B를 확인한다.
    2. 모든 union 연산을 처리할 때까지 1번 과정을 반복한다.
 - union 연산들은 간선으로 표현됨.
 - 일반적으로 서로소 집합을 그림으로 표현할 때는 번호가 큰 노드가 번호가 작은 노드를 간선으로 가리키도록 트리 구조를 이용해 그림을 그림


** 개선된 서로소 집합 알고리즘 소스코드 **
 - 경로 압축 기법을 적용해서 시간 복잡도를 개선
 - 경로 압축은 find 함수를 재귀적으로 호출한 뒤에 부모 테이블값을 갱신하는 기법

# 특정 원소가 속한 집합을 찾기(특정 원소의 루트노드 찾기)
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(union 연산)의 개수 입력받기
v, e  = map(int, input().split())
parent = [0] * (v + 1) # 부모 테이블 초기화

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i

# union 연산을 각각 수행
for i in range(e):
    a, b, = map(int, input().split())
    union_parent(parent, a, b)

# 각 원소가 속한 집합 출력
print('각 원소가 속한 집합: ', end=' ')
for i in range(1, v + 1):
    print(find_parent(parent, i), end=' ')

print()

# 부모 테이블 내용 출력
print('부모 테이블: ', end=' ')
for i in range(1, v + 1):
    print(parent[i], end=' ')



# 서로소 집합 과정 - 01
# '같은 팀 여부 확인' 연산에 대하여 한 줄에 하나씩 YES 혹은 NO로 결과를 출력
# 각 연산을 하나씩 확인
for i in range(m):
    oper, a, b = map(int, input().split())
    # 합집합(union)의 연산의 경우
    if oper == 0:
        union_parent(parent, a, b)
    # 찾기(find) 연산인 경우
    elif oper == 1:
        if find_parent(parent, a) == find_parent(parent, b):
            print('YES')
        else:
            print('NO')





#### 1 - 2.서로소 집합을 활용한 사이클 판별
 - 간선을 하나씩 확인하면서 두 노드가 포함되어 있는 집합을 합치는 과정을 반복하는 것만으로도 사이클을 판별할 수 있음
    1. 각 간선을 확인하며 두 노드의 루트 노드를 확인함
	- 루트 노드가 서로 다르다면 두 노드에 대하여 union 연산을 수행
	- 루트 노드가 서로 같다면 사이클(Cycle)이 발생한 것
    2. 그래프에 포함되어 있는 모든 간선에 대하여 1번 과정을 반복함
 - 사이클 판별 알고리즘은 그래프에 포함되어 있는 간선의 개수가 E개일 때, 모든 간선을 하나씩 확인하여,
   매 간선에 대하여 union 및 find 함수를 호출하는 방식으로 동작함.
 - 해당 알고리즘은 간선에 방향성이 없는 무향 그래프에서만 적용 가능함.

** 서로소 집합을 활용한 사이클 판별 소스코드 **

# 특정 원소가 속한 집합을 찾기(부모 노드 찾기)
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(union 연산)의 개수 입력받기
v, e  = map(int, input().split())
parent = [0] * (v + 1) # 부모 테이블 초기화

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i


cycle = False # 사이클 발생 여부

for i in range(e):
    a, b = map(int, input().split())
    # 사이클이 발생한 경우 종료
    if find_parent(parent, a) == find_parent(parent, b):
        cycle = True
        break
    # 사이클이 발생하지 않았다면 합집합(union) 수행
    else:
        union_parent(parent, a, b)

if cycle:
    print("사이클이 발생했습니다.")
else:
    print("사이클이 발생하지 않았습니다.")



#### 신장 트리(Spanning Tree)
 - 신장 트리란 하나의 그래프가 있을 때 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프를 의미한다.
 - 다양한 문제 상황에서 가능한 한 최소한의 비요응로 신장 트리를 찾아야 할 때가 있음
    Ex: 모든 도시를 '연결'할 때, 최소한의 비용으로 연결하려는 경우
 - 신장 트리 중에서 최소 비용으로 만들 수 있는 신장 트리를 찾는 알고리즘을 '최소 신장 트리 알고리즘'이라 한다.
 - 대표적인 최소 신장 틜 알고리즘으로는 '크루스칼 알고리즘'이 있다

#### 2. 크루스칼 알고리즘(Kruskal Algorithm)
 - 크루스칼 알고리즘을 사용하면 가장 적은 비용으로 모든 노드를 연결할 수 있는데
   크루스칼 알고리즘음 그리디 알고리즘으로 분류된다.
 - 먼저 모든 간선에 대하여 정렬을 수행한 뒤에, 가장 거리가 짧은 간선부터 집합에 포함시키면 된다.
 - 이때 사이클을 발생시킬 수 있는 간선의 경우, 집합에 포함시키지 않는다.
    1. 간선 데이터를 비용에 따라 오름차순으로 정렬한다.
    2. 간선을 하나씩 확인하며 현재의 간선이 사이클을 발생시키는지 확인한다.
         - 사이클이 발생하지 않는 경우 최소 신장 트리에 포함시킨다.
	- 사이클이 발생하는 경우 최소 신장 트리에 포함시키지 않는다.
    3. 모든 간선에 대하여 2번의 과정을 반복한다.
 - 최종적으로 신장 트리에 포함되는 간선의 개수가 '노드의 개수 - 1'과 같다는 특징이 있다.
 - 시간복잡도는 O(Elog(E))이다. 시간이 가장 오래 걸리는 부분은 간선을 정렬하는 작업이며,
   E개의 데이터를 정렬했을 때의 시간 복잡도는 O(Elog(E))이기 때문


** 크루스칼 알고리즘 소스코드(최초 신장 트리를 만드는데 필요한 비용을 계산하는 크루스칼 알고리즘)**

# 특정 원소가 속한 집합을 찾기(부모 노드 찾기)
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(union 연산)의 개수 입력받기
v, e  = map(int, input().split())
parent = [0] * (v + 1) # 부모 테이블 초기화

# 모든 간선을 담을 리스트와 최종 비용을 담을 변수
edges = []
result = 0

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i

# 모든 간선에 대한 정보를 입력받기
for _ in range(e):
    a, b, cost = map(int, input().split())
    # 비용순으로 정렬하기 위해서 튜플의 첫 번째 원소를 비용으로 설정
    edges.append((cost, a, b))

# 간선을 비용순으로 정렬
edges.sort()

# 간선을 하나씩 확인하며
for edge in edges:
    cost, a, b = edge
    # 사이클이 발생하지 않는 경우에만 집합에 포함
    if find_parent(parent, a) != find_parent(parent, b):
        union_parent(parent, a, b)
        result += cost

print(result)



#### 3. 위상 정렬(Topology Sort)
 - 위상 정렬은 정렬 알고리즘의 일종임.
 - 위상 정렬은 '순서가 정해져 있는 일련의 작업'을 차례대로 수행해야 할 때 사용할 수 있는 알고리즘임
 - 위상 정렬이란 방향 그래프이 모든 노드를 '방향성에 거스르지 않도록 순서대로 나열하는 것'
    Ex. "선수과목을 고려한 학습 순서 설정"
 - 그래프상에서 선후관계가 있다면, 위상정렬을 수행하여 모든 선후 관계를 지키는 전체 순서를 계산할 수 있음
 - 진입차수란 특정한 노드로 '들어오는' 간선의 개수를 의미함.
 - 위상 정렬 알고리즘
    1. 진입차수가 0인 노드를 큐에 넣는다
    2. 큐가 빌 때까지 다음의 과정을 반복한다.
        - 큐에서 원소를 꺼내 해당 노드에서 출발하는 간선을 그래프에서 제거한다.
	- 새롭게 진입차수가 0이 된 노드를 큐에 넣는다
 - 모든 원소를 방문하기 전에 큐가 빈다면 사이클이 존재한다고 판단할 수 있음
 - 위상 정렬의 답안은 여러 가지가 될 수 있음
 - 위상 정렬의 시간 복잡도는 O(V + E)임.


** 위상 정렬 소스코드**

from collections import deque

# 노드의 개수와 간선의 개수를 입력받기
v, e = map(int, input().split())
# 모든 노드에 대한 진입차수는 0으로 초기화
indegree = [0] * (v + 1)
# 각 노드에 연결된 간선 정보를 다믹 위한 연결 리스트(그래프) 초기화
graph = [[] for i in range(v + 1)]

# 방향 그래프의 모든 간선 정보를 입력받기
for _ in range(e):
    a, b = map(int, input().split())
    graph[a].append(b) # 정점 A에서 B로 이동 가능
    # 진입차수를 1 증가
    indegree[b] += 1

# 위상 정렬 함수
def topology_sort():
    result = [] # 알고리즘 수행 결과를 담을 리스트
    q = deque() # 큐 기능을 위한 deque 라이브러리 사용

    # 처음 시작할 때는 진입차수가 0인 노드를 큐에 삽입
    for i in range(1, v + 1):
        if indegree[i] == 0:
            q.append(i)

    # 큐가 빌 때까지 반복
    while q:
        # 큐에서 원소 꺼내기
        now = q.popleft()
        result.append(now)
        # 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
        for i in graph[now]:
            indegree[i] -= 1
            # 새롭게 진입차수가 0이 되는 노드를 큐에 삽입
            if indegree[i] == 0:
                q.append(i)

    # 위상 정렬을 수행한 결과 출력
    for i in result:
        print(i, end=' ')

topology_sort()



    # 다이나믹 프로그래밍
- 1. 최적 부분 구조: 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있다.
- 2. 중복되는 부분 문제: 동일한 작은 문제를 반복적으로 해결해야 한다.

** BottomUp 소스코드 예시 **
# 앞서 계산된 결과를 저장하기 위한 DP테이블 초기화
d = [0] * 100

# 첫 번째 피보나치 수와 두 번째 피보나치 수는 1
d[1] = 1
d[2] = 1
n = 99

# 피보나치 함수(Fibonacci Function) 반복문으로 구현(보텀업 다이나믹 프로그래밍)
for i in range(3, n+1):
    d[i] = d[i - 1] + d[i -2]

print(d[n])









    # greedy 문제들
# 문제 01 - 만들 수 없는 금액
n = int(input())
data = list(map(int, input().split()))

data.sort()
target = 1

for x in data:
  # 만들 수 없는 금액을 찾았을 때 반복 종료
  if target < x:
    break
  target += x
  print(x, target)

print(target)













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


# 왼쪽으로 회전 후 움직일 방향 정의(동서남북)
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


# 2차원 리스트 90도 회전(시계방향)
def rotate_a_matrix_by_90_degree(a):
  n = len(a) # 행 길이 계산
  m = len(a[0]) # 열 길이 계산
  result = [[0] * n for _ in range(m)] # 결과 리스트
  for i in range(n):
    for j in range(m):
      result[j][n - i - 1] = a[i][j]
  return result







    # stack, queue, priority queue
스택(Stack) - 가장 나중에 삽입된 데이터
큐(Queue) - 가장 먼저 삽입된 데이터
우선순위 큐(Priority Queue) - 가장 우선순위가 높은 데이터





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
