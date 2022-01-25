

    # 전체 과정, 흐름
* ML, Machine Learning
* 머신러닝 여러 설명
* 머신러닝 혼합 아이디어
* Import codes
* 변수 생성

---------- ML, Machine Learning ----------

    # 전체 과정, 흐름
* 데이터수집
* 데이터 정제
* 피처 엔지니어링(피처 생성 -> 피처 변환 -> 피처 선택)
    (1). 피처 생성 (Feature Generation)
    (2). 피처 변환 (Feature Preprocessing)
    (3). 피처 선택 (Feature Selection)
* 모델링(Modeling, Hyperparameter 최적화 with CV)
* 앙상블(Ensemble)
* 성능검증
* 모형 해석


    # ML- pipeline [Information Flow]
Stage 1: Domain Understaing
Stage 2: Data Collection and Understanding
Stage 3: Data Processing
    Data Cleansing: remove noise, outliers, missing values
    Data Integration: combine multiple datasources
    [ Feature Engineering ] Feature Construction / extraction : derive features from raw data
    [ Feature Engineering ] Feature Selection choose most relevant features remove redundant
    [ Feature Engineering ] Normalization: normalise contribution of each feature
    Sampling: subsample data to create training, cross validation, and test datasets
Stage 4: Data Mining / ML
Stage 5: Evaluation
Stage 6: Consolidation and Deployment



──────── 데이터수집 ────────







──────── 데이터 정제 ────────

    # Split data
from sklearn.model_selection import train_test_split
# train_test_split(data1, data2, test_size, train_size, random_state, shuffle, stratify)
# - train_test_split() 데이터를 학습용 데이터와 검증용 데이터로 분리한다.
# - test_size : 전체 데이터에서 테스트 데이터 세트 크기를 얼마로 샘플링 할 것인가를 결정. 디폴트는 0.25
# - train_size : 전체 데이터에서 학습용 데이터 세트 크기를 얼마로 샘플링할 것인가를 결정.
# - shuffle : 데이터를 분리하기 전에 데이터를 미리 섞을지를 결정합니다. 디폴트는 True.
#	  데이터를 분산시켜서 좀 더 효율적인 학습 및 테스트 데이터 세트를 만드는 데 사용됩니다.
# - random_state : 호출할때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수값입니다.
#		지정하지 않으면 수행할 때마다 다른 학습/테스트 용 데이터를 생성합니다.
# - stratify = y : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify= y(소문자)로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
#	       분류예측하는데 사용하는 것을 추천
X_train, X_test, y_train, y_test = train_test_split(train_features, y_train, test_size=0.25, random_state=42)
print(f'X-train shape: {X_train.shape}', \
      f'X_test.shape: {X_test.shape}', \
      f'y_train shape: {y_train.shape}',
      f'y_test shape: {y_test.shape}')










──────── 피처 엔지니어링(피처 생성 -> 피처 변환 -> 피처 선택) ────────




──────── (1). 피처 생성 (Feature Generation) ────────

    # 전처리를 동일하게 적용하기 위해 두 데이터를 합한다.
train_test = pd.concat([train, test])


    # 학습용 정답 데이터를 읽는다.
# 전처리 후 학습용과 제출용 데이터를 분리하기 위해 ID를 보관한다.
y_train = pd.read_csv('y_train.csv', encoding='cp949').GENDER.map({'남':0, '여':1})
IDtest = np.sort(test['CUS_ID'].unique())


    # 범주형 변수와 수치형 변수를 분리
cat_features = list(train_test.select_dtypes(include=['object']).columns)
num_features = [c for c in train_test.columns.tolist() if c not in cat_features]




──────── (2). 피처 변환 (Feature Preprocessing) ────────

    # 결측값 처리 - Ex 1
train_test[cat_features] = train_test[cat_features].fillna('None')
train_test[num_features] = train_test[num_features].fillna(0)

    # 결측값 처리 - Ex 2
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True')
    df['Fare'].fillna(0, inplace=True)
    return df

    # 결측치 처리 - Ex 3 (SimpleImputer를 사용하는 방법)
from sklearn.impute import SimpleImputer
 # strategy='mean' (평균 대체),
 # strategy='mdian' (중위수 대체)
 # strategy='most_frequent' (최빈값 대체)
imputer_con = SimpleImputer(strategy='median')
imputer_con.fit(data[con])
data_imp = data.copy()
# imputer를 fit훈련시킨후 transform으로 실제 적용하는 것
data_imp[con] = imputer_con.transform(data[con])



    # 이상치(Outlier) 제거
# 이상치 데이터(Outlier)는 전체 데이터의 패턴에서 벗어난 이상 값을 가진 데이터이며, 아웃라이어라고도 불림.
# 이상치로 인해 머신러닝 모델의 성능에 영향을 받는 경우가 발생하기 쉬움.

    # clip을 통한 이상치 제거 방식
# 비율에서 0.05% 보다 작은 값들을 정확히 0.05%에 해당하는 값으로 대체
# 비율에서 0.95% 보다 큰 값들을 정확히 0.95%에 해당하는 값으로 대체
train_test.iloc[:,1:] = train_test.iloc[:,1:].apply(lambda x: x.clip(x.quantile(.05), x.quantile(.95)), axis=0)

    # IQR(Inter Quantile Range) 이상치 제거 방식
# - 25% 구간인 Q1부터 75%구간인 Q3의 범위를 IQR이라고 함.
# - IQR은 사분위(Quantile) 값의 편차를 이용하는 기법으로 흔히 박스 플롯(Box Plot)방식으로 시각화할 수 있음: 박스플롯은 사분위의 편차와 IQR, 그리고 이상치를 나타냄
#   [이상치, 최솟값(Q1에서 IQR*1.5 값을 뺀 지점), Q1, Q2, Q3, 최댓값(Q3에서 IQR*1.5를 더한 지점), 이상치]
# - IQR을 이용해 이상치 데이터를 검출하는 방식은 보통 IQR에 1.5를 곱해서 생성된 범위를 이용해 최댓값과 최솟값을 경정한 뒤 최댓값을 초과하거나 최솟값에 미달하는 데이터를 이상치로 간주함
#   Q3에 IQR * 1.5를 더해서 일반적인 데이터가 가질 수 있는 최댓값으로 가정하고, Q1에 IQR * 1.5를 빼서 일반적인 데이터가 가질 수 있는 최솟값을 가정함. 경우에 따라서 1.5가 아닌 다른 값을 적용할 수도 있지만 보통은 1.5를 적용함.
# - 매우 많은 피처가 있을 경우 이들 중 결정값(즉 레이블)과 가장 상관성이 높은 피처들을 위주로 이상치를 검출하는 것이 좋음.
#   모든 피처들의 이상치를 검출하는 것은 시간이 많이 소모되며, 결정값과 상관값이 높지 않은 피처들의 경우는 이상치를 제거하더라고 크게 향상에 기여하지 않기 때문.

    # IQR 이상치 제거 - Ex 1
def get_oulier(df=None, column=None, weight=1.5):
# fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 구함.
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
# IQR을 구하고 IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    hightest_val = quantile_75 + iqr_weight
# 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정하고 DataFrame index 반환.
    outlier_index = fraud[(fraud < lowest_val) | (fraud > hightest_val)].index
    return outlier_index
outlier_index = get_outlier(df=card_df, column='V14', weight=1.5)
print('이상치 데이터 인덱스: ', outlier_index)

# get_processed_df()를 로그 변환 후 V14 피처의 이상치 데이터를 삭제하는 로직으로 변경
def get_processed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    # 이상치 데이터 삭제하는 로직 추가
    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy

    # IQR 이상치 제거 - Ex 2
# 이상치 제거 함수
def removeOutliers(x, column):
# Q1, Q3구하기
    q1 = x[column].quantile(0.25)
    q3 = x[column].quantile(0.75)

# 1.5 * IQR(Q3 - Q1)
    iqt = 1.5 * (q3 - q1)

# 원래 데이터 복제
    y = x

# 이상치를 NA로 변환
    y["tip"][(tips["tip"] > (q3 + iqt)) | (tips["tip"] < (q1 - iqt))] = None
    y["tip"]

# y 반환
    return(y)

    # IQR 이상치 제거 - Ex 3
from collections import Counter
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers
Outliers_to_drop = detect_outliers(df_train, 2, ["변수명"])


    # Isolation Forest 패키지 이용
# - Isolation Forest와 Local Outier Factor가 많은 경우에 대체로 우수하다고 하여,
#  여기서는 이 Isolation Foreset를 이용하여 Outlier detection을 해 보도록 하겠다.
# 이 방법을 이용해서 outlier를 제거하는 코드를 작성하는 것은 어렵지 않았다.
# 1. fit - 데이터를 학습시킵니다.
# 2. predict - 학습한 정보를 이용해 outlier를 판별한다.

# - Isolation Forest 방법을 사용하기 위해, 변수로 선언을 해 준다.
if_clf = IsolationForest(max_samples=1000, random_state=42)
# - fit 함수를 이용하여, 데이터셋을 학습시킨다. race_for_out은 dataframe의 이름이다.
if_clf.fit(race_for_out)
# - 다음으로 predict 함수를 이용하여, outlier를 판별해 준다. 0과 1로 이루어진 Series형태의 데이터가 나온다.
y_pred_outliers = clf.predict(race_for_out)
# - 원래의 dataframe에 붙여서 사용하기 위해서, 데이터 형태를 dataframe으로 바꾸고 명령어를 이용해서 붙여준다.
# 데이터가 0인 것이 outlier이기 때문에, 0인 것을 제거하면 outlier가 제거된  dataframe을 얻을 수 있다.
out = pd.DataFrame(y_pred_outliers)
out = out.rename(columns={0: "out"})
race_an1 = pd.concat([race_for_out, out], 1)


    # 표준화 값(z 점수)에 근거에한 이상치
# • 표준화 값(z 점수)에 근거하여 데이터값을 다음과 같이 분류할 수 있다.
# • 이상값(Unusual): if |zi| › 2 (µ ± 2σ이상)
# • 특이값(Outlier): if |zi| › 3 (µ ± 3σ이상)
#  - IQR을 이용한 방법 외에 이런식으로도 이상값과 특이값을 제거할 수도 있겠다.



    # 인코딩(Encoding)
    # 레이블 인코딩(label Encoding) / 라벨 인코딩
# 카테고리 피처를 코드형 숫자 값으로 변환하는 것(하나의 라벨을 하나의 정수와 대응시켜 Encoding 한다)
# LabelEncoder 객체를 생성한 후, fit()과 transform()으로 레이블 인코딩 수행
# - 숫자 값의 경우 크고 작음에 대한 특성이 작용하기 때문에 특정 ML 알고리즘에서는 가중치가 더 부여되거나 더 중요하게 인식할 가능성이 있다.
# 이러한 특성때문에 레이블 인코딩은 선형회귀와 같은 ML알고리즘에는 적용되지 않아야함.
# 트리계열 ML알고리즘은 숫자의 이러한 특성을 반영하지 않으므로 레이블 인코딩도 별 문제가 없음.
from sklearn.preprocessing import LabelEncoder
items = ['TV', '냉장고',' '전자레인지', '컴퓨터', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값: ', labels)
print('인코딩 클래스: ', encoder.classes_)

# inverse_transform()을 통해 인코딩된 값을 다시 디코딩할 수 있음
print('디코딩 원본 값: ', encoder.inverse_transform([4,5,2,0,1,1,3,3]))

    # 레이블 인코딩 - Ex 1
def labelEncode_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

    # 원핫인코딩(One-Hot Encoding)
# - 피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 칼럼에만 1을 표시하고 나머지 칼럼에는 0을 표시하는 방식
# - 사이킷런에도 있지만 판다스에서 더 쉽게 지원하므로 판다스의 API인 get_dummies()을 기록
import pandas as pd
df = pd.DataFrame({'item':['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
pd.get_dummies(df)

    # 원핫인코딩(One-Hot Encoding) - Ex 2
# 0과 1의 배열(벡터)을 하나의 라벨(카테고리 값)에 대응하여 Encoding한다.(해당하는 범주만 1로 하고 나머지는 다 0으로 만들어진다.)
# 선형모델에서는 꼭 원핫인코딩을 해야하는데 트리계열은 원핫인코딩을 해도 좋고 안해도 성능저하가 별로 없다.
# 원핫인코딩을 하게 되면 해당 칼럼이 사라지고 인코딩시킨 칼럼이 추가된다.
data_imp_ohe = pd.get_dummies(data_imp, columns['day'])

# 범주형 변수에 One-Hot-Encoding 후 수치형 변수와 병합
if len(cat_features) > 0:
    train_test = pd.concat([train_test[num_features], pd.get_dummies(train_test[cat_features])], axis=1)

    # Mean Encoding
# Mean Encoding은 특히 Gradient Boosting Tree 계열에 많이 쓰이고 있다.
# 원핫인코딩이나 라벨인코딩은 인코딩된 값 자체가 별로 의미가 없다. 하지만 Mean Encoding은 구분을 넘어 좀 더 의미있는 Encoding을 하기 위해서
# 내가 Encoding하는 feature와 예측하려하는 target간의 어떤 수치적인 관계를 catgorical에서도 찾으려는 것이다.
# Regression이든 Classification이든 이런 feature는 예측 값에 좀 더 가깝게 학습되게 한다. -> 즉, Less bias를 가진다. -> 오버피팅의 문제가 있다.
target = 'Survived' # target 설정
sex_mean = df.groupby('Sex')[target].mean() # target에 대한 sex 내 각 변수의 mean을 구함
df['Sex_mean'] = df['Sex'].map(sex_mean) # 가존 변수에 encoded된 값을 매핑

    # Smoothing, CV loop, Expanding mean
# Mean Encoding은 예측 값에 대한 정보가 포함되기 때문에 오버피팅이 된다. 따라서 Mean Encoding에는 Data Leakage와 Overfitting을 최소화하려는 기법들이 존재한다.

    # Smoothing
# Smoothing은 위에 단점의 마지막상황(Trainset에는 남자가 100명, 여자가 5명이고, Testset에는 50명, 50명인 경우를 고려한 기법)
# 저 5명의 평균이 여자 전체의 평균을 대표한다고 보기엔 힘드니, 그 평균을 남녀 무관한 전체 평균에 좀더 가깝게 만다는 것이다.
# 즉, 치우쳐진 평균을 전체 평균에 가깝도록, 기존 값을 스무스하게 만든다.
# alpha 는 하이퍼 파라미터로, 유저가 임의로 주는 것이다.
# 0을 주면, 기존과 달라지는게 없다. (저 식에 0을 넣으면 그냥 원래 encoding 된 값이다.)
# 반대로 alpha 값이 커질수록 더 스무스하게 만든다, 보통 알파는 카테고리 사이즈와 동일할 때, 신뢰할 수 있다함.
df['Sex_n_rows'] = df['Sex'].map(df.groupby('Sex').size())
global_mean = df[target].mean()
alpha = 0.7

def smoothing(n_rows, target_mean):
    return (target_mean*n_rows + global_mean*alpha) / (n_rows + alpha)
df['Sex_mean_smoothing'] = df.apply(lambda x:smoothing(x['Sex_n_rows'], x['Sex_mean']), axis=1)

    # CV loop
# CV Loop은 Trainset 내에서 cross validation을 통한 Mean Encoding을 통해 Data Leakage를 줄이고,
# 이전보다 Label값에 따른 Encoding값을 다양하게 만다는 시도를 한다.
# Encoding값을 다양하게 만들면, 트리가 만들어질 때, 더 세분화되어 나누어져, 더 좋은 훈련효과를 볼 수 있다.
    # CV loop - Ex 1
from sklearn.model_selection import train_test_split
# trainset 과 testset 분리.
# encoding은 무조건 trainset 만 사용해야 한다.
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
# train -> train_new 로 될 예정. 미리 데이터프레임 만들어주기.
train_new = train.copy()
train_new[:] = np.nan
train_new['Sex_mean'] = np.nan
from sklearn.model_selection import StratifiedKFold
# Kfold 만들어 주기.
X_train = train.drop(target, axis=1)
Y_train = train[target]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 각 Fold iteration.
for tr_idx, val_idx in skf.split(X_train, Y_train):
    X_train, X_val = train.iloc[tr_idx], train.iloc[val_idx]
# train set 에서 구한 mean encoded 값을 validation set 에 매핑해줌.
    means = X_val['Sex'].map(X_train.groupby('Sex')[target].mean())
    X_val['Sex_mean'] = means
    train_new.iloc[val_idx] = X_val
# 폴드에 속하지못한 데이터들은 글로벌 평균으로 채워주기.
global_mean = train[target].mean()
train_new['Sex'] = train_new['Sex'].fillna(global_mean)
train_new[['Sex', 'Sex_mean']].head()
# 위에서 Fold의 수는 5개 였으니, 각각의 train fold마다 Mean encoding 값을 만들어 낼 것이고, 그러면 한 label값당 5개의 encoding값이 나오는 것을 알 수 있다.
# 보통 4-5fold를 진행하면 괜찮은 결과, 카테고리 수가 너무 작은 경우엔 오버피팅

    # Expanding mean
# Expanding mean은 Label당 encoded되는 값을 좀 더 많이 만들어보자는 시도이다. 즉 위 CV Loop 기법에서는 encoded 값이 Fold 수 만큼 나올 수 밖에 없었다.
# Expandin mean은 cumsum()과 cumcount()를 이용하여, encoded 된 값의 특성은 지니면서, 값을 좀 더 잘게 나누는 테크닉이다.
# -> 하지만 이렇게 만들어낸 값이 유용한 값일지, noise 인지 확신할 수는 없기에 경우에 따라서 잘 써야한다.
# -> CatBoost모델에서 이 기법이 Built-in 되어 기본적인 성능 향상을 시켰다고 한다.
cumsum = train.groupby('Sex')[target].cumsum() - train[target]
cumcnt = train.groupby('Sex').cumcount()
train_new['Sex_mean'] = cumsum / cumcnt





    # 스케일링(feature scaling)
# - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
# 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고 독립 변수의 공분산 행렬의 조건수(condition number)를
# 감소시켜 최적화 과정에서의 안정성 및 수렴속도를 향상시킨다.
# 변수들의 단위 차이로 인해 숫자의 스케일이 크게 달라지는 경우, 스케일링으로 해결하는 것이다.
# 일반적으로
#	회귀 분석 문제의 더미 변수는 스케일링 하지 않고 0, 1값을 사용합니다.
#	분류 문제에서는 스케일링해도 결과에 큰 영향을 미치지 않습니다
#	특히 k-means 등 거리 기반의 모델에서는 스케일링이 매우 중요하다.
# 모든 스케일러 처리 전에는 아웃라이어 제거가 선행되어야 한다. 또한 데이터의 분포 특징에 따라 적절한 스케일러를 적용해주는 것이 좋다.
# 보통 회귀에서 하고 분류에서는 안한다 함

    # scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.
# - StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
# - RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환. 아웃라이어의 영향을 최소화
# - MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
# - MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환, 최대절대값과 0이 각각 1,0이 되도록 스케일링

    # 사용방법
# (1) 학습용 데이터의 분포 추정: 학습용 데이터를 입력으로 하여 fit 메서드를 실행하면 분포 모수를 객체내에 저장
# (2) 학습용 데이터 변환: 학습용 데이터를 입력으로 하여 transform 메서드를 실행하면 학습용 데이터를 변환
# (3) 검증용 데이터 변환: 검증용 데이터를 입력으로 하여 transform 메서드를 실행하면 검증용 데이터를 변환
# (1)번과 (2)번 과정을 합쳐서 fit_transform 메서드를 사용할 수도 있다.

# 전체 스케일링 결과는 비슷하지만 아웃라이어를 제거한 나머지 데이터의 분포는 로버스트 스케일링을 사용했을 때가 더 좋다.

    # StandardScaler
# 개별 피처를 평균이 0이고 분산이 1인 값으로 변환
# preprocessing using zero mean and nuit variance scaling
# 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
# 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.
# (SVM, 선형회귀(Linear Regression), 로지스틱 회귀에서는 데이터가 가우시안 분포를 가지고 있다고 가정하고 구현돼어서 사전에 표준화를 적용하는 것은 예측 성능 향상에 중요한 요소가 될 수 있음
from sklearn.preprocessing import StandardScaler
# StandardScaler 객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환. fit()과 transform() 호출.
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
#transform()시 스케일 변환된 데이터 세트가 Numpy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)
print('feature들의 평균 값')
print(iris_df_scaled.mean())
print('feature들의 분산 값')
print(iris_df_scaled.var())

    # MinMaxScaler
# - MinMaxScaler는 데이터값을 0과 1사이의 범위로 변환합니다(음수 값이 있으면 -1에서 1값으로 변환합니다.)
# - 데이터의 분포가 가우시안 분포가 아닐 경우 Min, Max Scale을 적용해 볼 수 있음.
# 모든 feature 값이 0~1 사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축 될 수 있다.
# 즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.
from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler 객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환. fit()과 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
# transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)
print('feature들의 최솟값')
print(iris_df_scaled.min())
print('feature들의 최댓값')
print(iris_df_scaled.max())

    # MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며,
# 큰 이상치에 민감할 수 있다.
from sklearn.preprocessing import MaxAbsScaler
maxAbsScaler = MaxAbsScaler()
maxAbsScaler.fit(train_data).transform(train_data)

    # RobustScaler
from sklearn.preprocessing import RobustScaler
# 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에
# StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포시키고 있음을 확인할 수 있다.
# IQR = Q3 - Q1 : 즉, 25퍼센트와 75퍼센트의 값들을 다룬다.
robustScaler = RobustScaler()
robustScaler.fit(train_data)
train_data_robustScaled = robustScaler.transform(train_data)



    # log화, log1p
# 왼쪽으로 치우진 분포를 정규분포로 바꾸기 위해 로그 변환을 수행한다.
# 로그 함수 혹은 제곱근 함수 등을 사용하여 변환된 변수를 사용하면 회귀 성눙이 향상 될 수 있다.
#       독립 변수나 종속 변수가 심하게 한쪽으로 치우친 분포를 보이는 경우
#       독립 변수와 종속 변수간의 관계가 곱셈 혹은 나눗셈으로 연결된 경우
#       종속 변수와 예측치가 비선형 관계를 보이는 경우
#  -> 데이터 중 금액처럼 큰 수치 데이터에 로그를 취하게 되는 이유이기도 하다.
#  -> 보통 이런 데이터는 선별적으로 로그를 취한 후 모델링 전 전반적으로 스케일링을 적용한다.(in my opinion)
train_test.iloc[:,1:] = np.log1p(train_test.iloc[:,1:])



    # 차원 축소
# 특성 차원이 너무 많을 경우 과적합이 발생하기 때문에 차원 축소를 실행한다.

    # 주성분 분석(PCA)
from sklearn.decomposition import PCA
# 아래와 같이 하는 방법도 있지만 num_d에 0.99로 실수 값을 넣어주어서 간단하게 할 수도 있다.
max_d = num_d = train_test.shape[1] - 1
pca = PCA(n_components=max_d, random_state=0).fit(train_test.iloc[:,1:])
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.99) + 1             # 분산의 설명량이 99%이상 되는 차원의 수
if num_d == 1: num_d = max_d
pca = PCA(n_components=num_d, random_state=0).fit_transform(train_test.iloc[:,1:])
train_test = pd.concat([train_test.iloc[:,0], pd.DataFrame(pca)], axis=1)






    # 오버샘플링(Oversampling)과 언더샘플링(Undersampling)
# - 오버 샘플링 방식이 예측 성능상 더 유리한 경우가 많아 주로 사용됨

# 언더샘플링
# - 언더샘플링은 많은 데이터 세트를 적은 데이터 세트 수준으로 감소시키는 방식입니다.
# - 즉, 정상 레이블을 가진 데이터가 10000건, 이상 레이블을 가진 데이터가 100건이 있으면 정상 레이블 데이터를 100건으로 줄여버리는 방식입니다.
# - 이렇게 정상 레이블 데이터를 이상 레이블 데이터 수준으로 줄여 버린 상태에서 학습을 수행하면 과도하게 정상 레이블로 학습/예측하는 부작용을 개선할 수 있지만,
# 너무 많은 정상 레이블 데이터를 감소시키기 때문에 정상 레이블의 경우 오히려 제대로 된 학습을 수행할 숭 없다는 단점이 있어 잘 적용하지 않는 방법입니다.

# 오버샘플링
# - 오버 샘플링은 이상 데이터와 같이 적은 데이터 세트를 증식하여 학습을 위한 충분한 데이터를 확보하는 방법입니다.
# - 동일한 데이터를 단순히 증식하는 방법은 과적합(Overfitting)이 되기 때문에 의미가 없으므로 원본 데이터의 피처 값들을 아주 약간만 변경하여 증식합니다.
# - 대표적은 SMOTE(Synthetic Minority Over-sampling Tehnique) 방법이 있습니다.
# SMOTE는 적은 데이터 세트에 있는 개별 데이터들의 K 최근접 이웃(K Nearest Neighbor)을 찾아서
# 이 데이터와 K개 이웃들의 차이를 일정 값으로 만들어서 기존 데이터와 약간 차이가 나는 새로운 데이터들을 생성하는 방식입니다.

    # SMOTE 객체의 fit_sample()
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n, pd.Series(y_train_over).value_counts())
# SMOTE를 적용하면 재현율은 높아지나, 정밀도는 낮아지는 것이 일반적임 (오버샘플링으로 인해 실제 원본 데이터의 유형보다 너무나 많은 Class=1 데이터를 학습하면서 실제 테스트 데이터 세트에서 예측을 지나치게 Class=1로 적용해 정밀도가 급격하게 떨어지게 된 것.)
# 재현율이 높더라도 정밀도가 지나치게 저조할 경우 현실 업무에 적용할 수 없음.
# 분류 결정 임곗값에 따른 정밀도와 재현율 곡선을 통해 SMOTE로 학습된 로지스틱 회귀 모델에 어떠한 문제가 발생하고 있는지 시각적으로 확인
precision_recall_curve_plot(y_test, lr_clf.predic_proba(X_test)[:,1])



# 전처리 후 학습용과 제출용 데이터로 분리한다.
train_x = train_test.query('CUS_ID not in @IDtest').drop('CUS_ID', axis=1)
test_x = train_test.query('CUS_ID in @IDtest').drop('CUS_ID', axis=1)



# 불필요한 칼럼 제거
def drop_features(df):
    df.drop(['PassnegerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df



# 피처 전처리 함수 한꺼번에 호출
def tranform_features(df):
    df = fillna(df)
    df = labelEncode_features(df)
    df = drop_featuers(df)
    return df
X_titanic_df = transform_features(X_titanic_df)



──────── (3). 피처 선택 (Feature Selection) ────────


    # Feature Selection
# 모델에 적용할 input 변수를 잘 선정하는 것은 모델의 성능에 직접적인 영향을 미치는 과정이다.
# 아래의 방법을 이용해 모델의 성는을 높이는데 유용한 변수를 선택할 필요가 있다.
# - Univariate Selection (T-test, ANOVA, Coefficient and so on)
# - Feature Importance (from Tree-based model)
# - RFE (recursive feature elimination)

    # Univariate feature selection
# ※ Univariate Selection은 그룹내 분산이 작고 그룹간 분산이 클 경우 값이 커지는 F-value를 이용하여 분수를 선택한다.
# 각 변수마다 F값을 구해 F 값이 큰 변수를 기준으로 변수를 선택하는 방법이다.
# -> [SelectKBest] removes all but the  highest scoring features
# -> [SelectPercentile] removes all but a user-specified highest scoring percentage of features using common univariate statistical tests for each feature: false positive rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
# -> [GenericUnivariateSelect] allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.

# These objects take as input a scoring function that returns univariate scores and p-values (or only scores for SelectKBest and SelectPercentile):
# For regression: f_regression, mutual_info_regression
# For classification: chi2, f_classif, mutual_info_classif
#                     chi2: 카이제곱 검정 통계값
#                     f_classif: 분산분석(ANOVA) F검정 통계값
#                     mutual_info_classif: 상호정보량(mutual information)

    # Univariate feature selection - Ex 1 (SelectKBest)
from sklearn.feature_selection import SelectKBest, f_classif
selectK = SelectKBest(score_func=f_classif, k=8)
x = selectK.fit_transform(X, y)

    # Univariate feature selection - Ex 2 (SelectKBest)
from sklearn.feature_selection import SelectKBest, chi2
selector1 = SelectKBest(chi2, k=14330)
X_train1 = selector1.fit_transform(X_train, y_train)
X_test1 = selector1.transform(X_test)

    # Univariate feature selection - Ex 3 (SelectPercentile)
    # SelectPercentile
model = LogisticRegression(random_state=0)
# 각 특성과 타깃(class) 사이에 유의한 통계적 관계가 있는지 계산하여 특성을 선택하는 방법
cv_scores = []
for p in tqdm(range(5,100,1)):
    X_new = SelectPercentile(percentile=p).fit_transform(train_x, y_train)
    cv_score = cross_val_score(model, X_new, y_train, scoring='roc_auc', cv=5).mean()
    cv_scores.append((p,cv_score))

# Print the best percentile
best_score = cv_scores[np.argmax([score for _, score in cv_scores])]
print(best_score)

# Plot the performance change with p
plt.plot([k for k, _ in cv_scores], [score for _, score in cv_scores])
plt.xlabel('Percent of features')
plt.grid()
plt.show()

# 과적합을 피하기 위해 최적의 p값 주변의 값을 선택하는게 더 나은 결과를 얻을 수 있다.
selectp = SelectPercentile(percentile=best_score[0]).fit(train_x, y_train)
X_train_sel = selectp.transform(train_x)
X_test_sel = selectp.transform(test_x)


    # Feature Selection based on Feature Importance
# ※ ExtraTreesClassifier와 같은 트리 기반 모델은 Feature Importance 를 제공한다.
# 이 Feature Importance는 불확실도를 많이 낮출수록 증가하므로 이를 기준으로 변수를 선택할 수 있다.
from sklearn.ensemble import ExtraTreesClassifier
etc_model = ExtraTreeClassifier()
etc_model.fit(X, y)
print(etc_model.feature_importances_)
feature_list = pd.concat([pd.Series(X.columns), pd.Series(etc_model.feature_importances_)], axis=1)
feature_list.columns = ['features_name', 'importance']
feature_list.sort_values("importance", ascending=False)[:8]


    # Feature Selection based on RFE
# ※ 마지막으로 RFE (recursive feature elimination)는 Backward 방식 중 하나로,
# 모든 변수를 우선 다 포함시킨 후 반복해서 학습을 진행하면서 중요도가 낮은 변수를 하나씩 제거하는 방식이다.
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


    # Model based feature selection
# 다른 모형을 이용한 특성 중요도 계산
# 특성 중요도(feature importance)를 계산할 수 있는 랜덤포레스트 등의 다른 모형을 사용하여 일단 특성을 선택하고 최종 분류는 다른 모형을 사용할 수도 있다,
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
selectfromModel = SelectFromModel(ExtraTreesClassifier(random_state=0), threshold=None)
X_train_sc2_fs1 = selectfromModel.fit(X_train_sc2, y_train).transform(X_train_sc2)
X_test_sc2_fs1 = selectfromModel.transform(X_test_sc2)
svm.fit(X_train_sc2_fs1, y_train).score(X_test_sc2_fs1, y_test)

    # Model based feature selection - Ex 2
n_sample = 10000
idx = np.random.choice(range(len(y_train)), n_sample)
model_sel = ExtraTreesClassifier(n_estimators=50).fit(X_train[idx, :], y_train[idx])
selector = SelectFromModel(model_sel, prefit=True, max_features=14330)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)



    # 다중공선성
# 상관관계가 큰 독립 변수들이 있는 경우, 이 경우에는 변수 선택이나 PCA를 사용한 차원 축소 등으로 해결한다.


──────── 모델링(Modeling, Hyperparameter 최적화 with CV) ────────

    # 모델 이름 출력하기
model.__class__.__name__



    # Cross Validation
# - 교차 검증은 데이터 편중을 막기 위해서 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행하는 것입니다.
#   대부분의 ML 모델의 성능 평가는 교차검증 기반으로 1차 평가를 한 뒤에 최종적으로 테스트 데이터 세트에 적용해 평가하는 프로세스 입니다.
#   ML에 사용되는 데이터 세트를 세분화해서 학습, 검증, 테스트 데이터 세트로 나눌 수 있습니다. 테스트 데이터 세트 외에 별도의 검증 데이터 세트를 둬서 최정 평가 이전에 학습된 모델을 다양하게 평가하는 데 사용합니다.
#  학습 데이터 세트[학습 데이터를 다시 분할하여 학습 데이터와 학습된 모델의 성능을 일차 평가하는 검증 데이터로 나눔] ->  테스트 데이터 세트[모든 학습/검증 과정이 완료된 후 최종적으로 성능을 평가하기 위한 데이터 세트]

    # KFold - Process
# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
from sklearn.model_selection import KFold
kfold = KFold()
cv_accuracy = []

n_iter = 0
#KFold 객체의 split()를 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측
    df_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'#{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'#{n_iter} 검증 세트 인덱스:{test_index}')
    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))


    # KFold - Function
def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold 객체를 생성, 폴드 수만큼 예측결과 저장을 위한 리스트 객체 생성.
    kfold = KFold(n_splits=folds)
    scores =[]

    # KFold 교차 검증 수행
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_Df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        #Classifier 학습, 예측, 정확도 계산
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print(f'교차 검증 {iter_count} 정확도: {accuracy:.4f}')
        # 5개의 fold에서의 평균 정확도 계산
        mean_score = np.mean(scores)
        print(f'평균 정확도: {mean_score:.4f}')
# exec_kfold 호출
exec_kfold(df_clf, folds=5)



    # Stratified KFold
# - Stratified K 폴드는 불균형한(imbalanced) 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K 폴드 방식입니다.
#   불균형한 분포도를 가진 레이블 데이터 집합은 특정 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한쪽으로 치우지는 것을 말합니다.
# - Stratified K 폴드는 K 폴드가 레이블 데이터 집합이 원본 데이터 집합의 레이블 분포를 학습 및 테스트 세트에 제대로 분배하지 못하는 경우의 문제를 해결해 줍니다.
#   Stratified K 폴드는 원본 데이터의 레이블 분포를 먼저 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트를 분배합니다.
# - 일반적으로 분류에서의 교차 검증은 K 폴드가 아니라 Stratified K 폴드로 분할돼야 합니다.
# - 회귀에서는 Stratified K 폴드가 지원되지 않습니다. 회귀의 결정값은 이산값 형태의 레이블이 아니라 연속된 숫자값이기 때문에 결정값 별로 분포를 정하는 의미가 없기 때문입니다.
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

# StratifiedKFold의 split() 호출시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index in skf.split(features, label):
    # split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측
    df_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    # 반복 시마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'#{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'#{n_iter} 검증 세트 인덱스:{test_index}')
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n ## 교차 검증별 정확도: ', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy))



    # cross_val_score()
# - 폴드 세트를 설정하고, for 루트에서 반복적으로 학습 및 테스트 데이터의 인덱스를 추출한 뒤, 반복적으로 학습과 예측을 수행하고 예측 성능을 반환하는, 이런 일련의 과정을 한꺼번에 수행해주는 API가 cross_val_score() 입니다.
# - 내부에서 Estimator를 학습(fit), 예측(predict), 평가(evaluation) 시켜주므로 간단하게 교차검증을 수행할 수 있습니다.
# - cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
#   이 중 estimator, X, y, scoring, cv가 주요 파라미터입니다.
# - cross_val_score()는 classifier가 입력되면 Stratified K 폴드 방식으로 레이블값의 분포에 따라 학습/테스트 세트를 분할합니다.
#   (회귀인 경우는 Stratified K 폴드 방식으로 분할할 수 없으므로 K 폴드 방식으로 분할합니다)
from sklearn.model_selection import cross_val_score
# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(clf, data, label, scoring='accuracy', cv=3)
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))

scores = cross_val_score(logreg, iris.data, iris.target, cv = 5)
cross_val_score(tree, x_train, y_train, , scoring='roc_auc', cv=5)
print('cross-val-score \n{}'.format(scores))
print('cross-val-score.mean \n{:.3f}'.format(scores.mean())) # 교차 검증 평균값으로 우리는 이 모델의 정확도가 대략 몇%일 것으로 기대할 수 있음
print(f'Mean: {scores.mean():.3f} \
    \nStd:{scores.std():.3f} \
    \nMin: {scores.min():.3f} \
    \nMax: {scores.max():.3f}')

for n in [3, 5]:

    kfold = KFold(n_splits=n, shuffle=True, random_state=0)
    scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    print('n_splits={}, cross validation score: {}'.format(n, scores))

scores = cross_val_score(dt_clf, X_titanic, y_titanic, cv=5)
for iter_count, accuracy in enumerate(scores):
    print(f'교차 검증 {iter_count} 정확도: {accuracy:.4f}')
print(f'평균 정확도: {np.mean(scores):.4f}')


    # LOOCV(Leave-One-Out Cross-Validation)
# LOOCV is very time-consuming => useful in small data
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X_train, y_train, cv=LeaveOneOut())
scores.mean()


    # Shuffle-Split Cross-Validation
# n_splits: 몇번 실험 할건지 = 몇번 교차검증할 것인지
# 학습데이터 40프로, 테스트데이터 50프로, 10프로는 아에 안씀
# 주로 5-fold나 10-fold를 보편적으로 사용. 5번 또는 10번 교차검증한다는 것.
# defualt: ShuffleSplit(n_splits=10, test_size=None, train_size=None, random_state=None)
# random_state=None, 일시 할때마다 출력결과값이 달라짐.
from sklearn.model_selection import ShuffleSplit, cross_val_score
sscv = ShuffleSplit(test_size=.5, train_size=.4, n_splits=10)
scores = cross_val_score(model, X_train, y_train, cv=sscv)
scores.mean()




    # Desicion Tree, 의사결정나무
# - 데이터 균일도에 따른 규칙 기반의 결정트리
# - 정보의 균일도를 측정하는 대표적인 두 가지 방법

    # 1.정보 이득은 엔트로피라는 개념을 기반으로 합니다.
# 엔트로피는 주어진 데이터 혼잡도를 의미하는데,
# 서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞여 있으면 엔트로피가 낮습니다.
# 정보 이득 지수는 1에서 엔트로피 지수를 뺀 값입니다. 즉, 1-엔트로피 지수입니다.
# 결정트리는 이 정보 이득 지수로 분할 기준을 정합니다. 즉, 정보 이득이 높은 속성을 기준으로 분할합니다.

    # 2.지니 계수는 원래 경제학에서 불평등 지수를 나타낼 때 사용하는 계수입니다. 0이 가장 평등하고 1로 갈수록 불평등합니다.
# 머신러닝에 적용될 때는 의미론적으로 재해석돼 데이터가 다양한 값을 가질수록 평등하며 특정 값으로 쏠릴 경우에는 불평등한 값이 됩니다.
# 즉, 다양성이 낮을수록 균일도가 높다는 의미로서, 1로 갈수록 균일도가 높으므로 지니 계수가 높은 속성을 기준으로 분할하는 것입니다.

from sklearn.tree import DeicisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42, n_jobs=-1)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print(f'DecisionTreeClassifier 정확도 : {accuracy_score(y_test, dt_pred):.4f}')

    # y_test 있을 때 성능 보기
dtree_score = dtree.score(X_test, y_test)

    # DecisionTreeClassifier 하이퍼 파라미터
print('DecisionTreeClassifier 기본 하이퍼 파라미터: \n', dt_clf.get_params())
# max_depth[default=None]	 : 트리의 최대 깊이를 규정. 디폴트 None으로 설정하면 완벽하게 클래스 결정 값이 될 때까지 깊이를 계속 분할하거나 노드가 가지는 데이터 개수가 min_samples_split보다 작아질 때까지 계속 깊이를 증가. 적절한 값으로 제어 필요
# max_features[default=None] : 최적의 분할을 위해 고려할 최대 피처 개수. 디폴트는 None으로 데이터 세트의 모든 피처를 사용해 분할 수행.
#		int로 지정 -> 피처의 개수 / float로 지정 -> 전체 피처 중 대상 피처의 퍼센트,
#		'sqrt'는 전체 피처 중 sqrt(전체 피처 개수), 'auto'로 지정하면 sqrt와 동일, 'log'는 전체 피처 중 log2(전체 피처 개수), 'None'은 전체 피처 선정
# min_samples_split[default=2] : 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는데 사용됨. 작게 설정할수록(1로 설정할 경우도) 분할되는 노드가 많아져서 과적합 가능성 증가
# min_samples_leaf		  	: 말단 노드(Leaf)가 되기 위한 최소한의 샘플 데이터 수, 과적합 제어용도. 비대칭적(imbalanced) 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 이 경우는 작게 설정 필요.
# max_leaf_nodes		 	: 말단 노드(Leaf)의 최대 개수

    # 결정나무 시각화
import graphviz
from sklearn.tree import export_graphviz

    # 피쳐 중요도 시각화
ftr_importances_values = best_df_clf.feature_importances_
# Top 중요도로 정렬을 쉽게 하고, 시본의 막대그래프로 쉽게 표현하기 위해 Series 변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()



    # Random Forest
# - 랜덤포레스트의 개별적인 분류기의 기반 알고리즘은 결정 트리이지만
#   개별 트리가 학습하는 데이터 세트는 전체 데이터에서 일부가 중첩되게 샘플링된 데이터 세트입니다.
# - 이렇게 여러 개의 데이터 세트를 중첩되게 분리하는 것을 부트스트래핑(bootstrapping) 분할 방식이라고 합니다.(그래서 배깅(Bagging)이 bootstrap aggregating의 줄임말입니다.)
# - 랜포의 서브세트(Subset) 데이터는 이러한 부트스트래핑으로 데이터가 임의로 만들어집니다. 세브세트의 데이터 건수는 전체 데이터 건수와 동일하지만, 개별 데이터가 중첩되어 만들어 집니다.
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print(f'RandomForestClassifier 정확도:{accuracy_score(y_test, rf_pred):.4f}')
rf_score = rf.score(X_test, y_test)

    # RF 파라미터
# criterion [default = gini] : 다른거는 entropy (설명 결정트리에 존재)
# n_estimators [default = 10]	:나무개수
# max_depth	: 나무 깊이
# max_features [default='auto'='sqrt']: bootstrap sampling 무작위 추출할 변수 개수, 데이터의 feature를 참조할 비율,개수
# min_samples_split : 내부 노드를 분할하는데 필요한 최소 샘플 수 (default : 2)
# min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)
# min_weight_fraction_leaf : min_sample_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율
# max_leaf_nodes : 리프 노드의 최대수
# min_impurity_decrease : 최소 불순도
# min_impurity_split : 나무 성장을 멈추기 위한 임계치
# bootstrap : 부트스트랩(중복허용 샘플링) 사용 여부
# oob_score : 일반화 정확도를 줄이기 위해 밖의 샘플 사용 여부
# n_jobs :적합성과 예측성을 위해 병렬로 실행할 작업 수
# random_state : 난수 seed 설정
# verbose : 실행 과정 출력 여부
# warm_start : 이전 호출의 솔루션을 재사용하여 합계에 더 많은 견적가를 추가
# class_weight : 클래스 가중치

    # RF tuning - Ex 1
rf_params = {'n_estimators':[10,20,50,100],
	     'criterion': ['gini', 'entropy'],
 	     'max_depth': [3,6,9,12],
	     'max_features':[5,10,15,20],
	   }
#  모델 나머지 파라미터 설정 및 범위 적용
rf = RandomForestClassifier(**rf_params, oob_score=False, random_state=42,  n_jobs=-1)



    # ExtraTreesClassifier
# 랜덤 포레스트보다 특성선택과 샘플링에 있어 무작위성을 많이 갖는 앙상블 모델이다.
# 랜덤 포레스트에서 최적의 임계값을 찾아 그를 특성으로 선택했던 것과 달리, 엑스트라 트리는 후보 특성을 기반으로 무작위로 분할을 하고 그 중 최적치를 선택하게 된다. 즉, 무작위성이 더 크다.
# 또한 부트스트랩 샘플링을 적용하지 않고 전체 원본 샘플을 사용한다. 부트스트랩 샘플링에 따른 다양성으로 발생할 분산을 조금 줄이는 것이다.
# 계산 비용 및 실행 시간 측면에서 Extra Trees 알고리즘이 더 빠르다.
from sklearn.ensemble import ExtraTreesClassifier



    # GBM (GradientBoostingClassifier)
# - 부스팅 알고리즘은 여러 개의 약한 학습기(weak learner)를
#   순차적으로 학습-예측하면서
#   잘못 예측한 데이터에 가중치를 부여를 통해
#   오류를 개선해 나가면서 학습하는 방식임.
from sklearn.ensemble import GradientBoostingClassifier
start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
pred = gb_clf.predict(X_test)
gd_accuracy = accuracy_score(y_test, pred)
print(f'GBM 정확도: {gb_accuracy:.4f}')
print(f'GBM 수행시간: {time.time() - start_time:.1f} 초')
gb_score = gb_clf.score(X_test, y_test)
print(f'gbm_score: {gbm_score}')

    # GBM 파라미터
# n_estimator [default=100]: weak learner의 개수. weak learner가 순차적으로 오류를 보정하므로 개수가 많을수록 예측 성능이 일정 수준까지는 좋아질 수 있음.
# max_depth
# max_features
# loss : 경사하강법에서 사용할 비용 함수를 지정함. 특별한 이유가 없으면 기본값인 'deviance'를 그대로 적용함
# learning_rate [default=0.1]: GBM이 학습을 진행할 때마다 적용하는 학습률임. Weak learner가 순차적으로 오류 값을 보정해 나가는데 적용하는 계수. 0~1사이의 값을 지정할 수 있으면 기본값은 0.1임.
#	learning_rate은 n_estimator와 상호 보완적으로 조합해 사용해야함. learning_rate을 작게하고 n_estimator를 크게 하면 더 이상 성능이 좋아지지 않는 한계점까지는 예측 성능이 조금씩 좋아질 수 있음.
# subsample : weak learner가 학습에 사용하는 데이터 샘플링 비율임. 기본값은 1이며, 이는 전체 학습 데이터를 기반으로 학습한다는 의미임. 과적합이 염려되는 경우 subsample을 1보다 작은 값으로 설정함.

    # GBM tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
gb_clf = GradientBoostingClassifier(random_state=42, n_jobs=-1)
gb_params = {'learning_rate' : [0.1,0.3,0.5],
	         'n_estimators' : [100,500,1500],
             'max_depth' : [3,6,9],
             'min_samples_split' : [5,10,15],
	         'n_jobs' : -1}
gb_rs = RandomizedSearchCV(gb_clf, param_distributions = gb_params, n_iter = 10, cv=5, scoring="accuracy", verbose = 1)
gb_rs.fit(X_train,y_train)
gb_rs_best = gb_rs.best_estimator_
# Best score
display(gb_rs.best_score_, gb_rs_best)



    # XGBClassifier
# - 일반적으로 GBM은 순차적으로 weak learner가 가중치를 증감하는 방법으로 학습하기 때문에 전반적으로 속도가 느림
# - 하지만 XGB는 병렬 수행 및 다양한 기능으로 GBM에 비해 빠른 수행 성능을 보장함.
# - 표준 GBM의 경우 과적합 규제 기능이 없으나 XGB는 자체에 과적합 규제 기능으로 과적합에 좀 더 강한 내구성을 가질 수 있습니다.
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=3, random_state=42, n_jobs=-1)
evals = [(X_valid, y_valid)]
 # eval_set = [(X_train, y_train), (X_valid, y_valid)]
xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='loglosss', eval_set=evals, verbose=True)
 # xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)
pred = xgb_clf.predict(X_test)
get_clf_eval(y_test, pred)

xgb_score = xgb_clf.score(X_test, y_test)
print(f'xgb_score: {xgb_score}')
 # xgb_roc_score = roc_auc_score(y_test, pred)
 # print(f'ROC AUC: {xgb_roc_score:.4f}')

	# 튜닝의 순서
# 높은 learning rate를 선택하고, 이에 맞는 최적의 트리 수를 찾는다. cv를 사용
# 트리의 파라미터(max_depth, min_child_weight, gmma, subsample, solsample_bytree)를 결정한다.
# 과적합을 낮추도록 계속 신경쓴다
# 정규화 파라미터(lambda, alpha)를 조정해서 모델의 복잡도를 감소시키고 성능을 향상 시키도록 한다.
# Learning rate를 더 낮추고 최적의 파라미터를 결정한다

    # 과적합 문제가 있을시 다음과 같은 방법을 고려
# - learning_rate 낮추기, n_estimator 높이기
# - max_depth 낮추기
# - min_child_weight 높이기
# - gamma 높이기
# - subsample과 colsample_bytree 조정

    # 조기중단(early_stopping_rounds)
# - 조기중단은 xgb가 수행 성능을 개선하기 위해서 더 이상 지표 개선이 없을 경우에 n_estimators 횟수를 모두 채우지 않고 중간에 반복을 빠져 나올 수 있도록 하는 것임.
# - early_stopping_rounds 파라미터를 설정해 조기중단을 수행하기 위해서는 반드시 eval_set과 eval_metric이 함께 설정돼야 함.
# - xgb는 반복마다 eval_set으로 지정된 데이터 세트에서 eval_metric의 지정된 평가 지표로 예측 오류를 측정함.

    # XGB 파라미터
# - 일반 파라미터: 일반적으로 실행 시 스레드의 개수나 silent 모드 등의 선택을 위한 파라미터로서 디폴트 파라미터 값을 바꾸는 경우는 거의 없음
# - 부스터 파라미터: 트리 최적화, 부스팅, regularization 등과 관련 파라미터 등을 지칭
# - 학습 태스크 파라미터: 학습 수행 시의 객체 함수, 평가를 위한 지표 등을 설정하는 파라미터입니다.

# 주요 일반 파라미터
# booster(string, default: 'gbtree') : Booster가 사용할 모드(gbtree, gblinear, dart)
#                                      트리, 회귀(gblinear) 트리가 항상 더 좋은 성능을 내기 때문에 수정할 필요없다고 한다.
# slient: Boosting을 실행하는 동안 메세지를 print 할지 여부
#         silent 대신 verbosity를 사용하라고 되어 있다함
# verbosity(int, default: None) : Flag to print out detailed breakdown of runtime -> Valid values are 0 (silent) - 3 (debug)
# nthread(int, default: 1): xgboost를 실행하는데 사용할 병렬 스레드 수
#         nthread 대신 n_jobs 사용하기
# missing(float, default np.nan): 누락된 값(Missing Value)으로 존재하는 데이터를 처리할 값
# n_jobs(int, default: 1): Number of parallel threads used to run xgboost.
#                           제일 효율적으로 하고 싶으면 -1 로 모두 사용하기
# random_state(int) : seed와 동일


# 부스터 파라미터
# learning_rate [default=0.1] : 0에서 1 사이의 값을 지정하며 부스팅 스텝을 반복적으로 수행할 때 업데이트 되는 학습률 값. 보통은 0.01~0.2 사이의 값을 선호함
# n_estimators [default=100] : fit하기 위한 Boosted tree의 수
# max_depth[default=6] : 0을 지정하면 깊이에 제한이 없음. 보통은 3~10의 값을 적용함.
# min_child_weight[default=1] : 트리에서 추가적으로 가지를 나눌지 결정하기 위해 필요한 데이터들의 가중치 총합(과적합 조절 용도), 범위: 0~inf
#    - GBM의 min_child_leaf과 유사(똑같지는 않음). 과적합(overfitting)을 방지할 목적으로 사용되는데, 너무 높은 값은 과소적합(underfitting)을 야기하기 때문에 CV를 사용해서 적절한 값이 제시되어야 한다.
# gamma[default=0] : 트리의 리프 노드를 추가적으로 나눌지를 결정할 최소 손실 감소 값입니다. 해당 값보다 큰 손실(loss)이 감소된 경우에 리프 노드를 분리합니다. 값이 클수록 과적합 감소 효과가 있음.
#                  노드가 split 되기 위한 loss function의 값이 감소하는 최소값을 정의한다.
#                  gamma 값이 높아질수록 알고리즘은 보수적으로 변하고, loss function의 정의에 따라 적정값이 달라지기 때문에 반드시 튜닝
# sub_sample[default=1] : GBM의 subsample과 동일. 트리가 커져서 과적합되는 것을 제어하기 위해 데이터를 샘플링하는 비율을 지정함. 일반적으로 0.5 ~ 1사이의 값을 사용함.
#                         학습(Training) Instance의 subsample 비율
# col_sample_bytree[default=1] : GBM의 max_features와 유사. 트리 생성에 필요한 피처(칼럼)을 임의로 샘플링 하는데 사용됨. 매우 많은 피처가 있는 경우 과적합을 조정하는데 적용함.
#                                각 Tree를 구성할 때 column의 Subsample 비율, 보통 0.6~0.9
# reg_alpha : L1 Regularization 적용 값임. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과가 있음
#             Weight에 대한 L1 정규화(regularization)
# reg_lambda[default=1] : L2 Regularization 적용 값임. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과가 있음
#                         Weight에 대한 L2 정규화(regularization)
# scale_pos_weight[default=1] : 특정 값으로 치우친 비대한 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터임.
#                              양의 클래스와 음의 클래스에 대한 균형
# max_delta_step(int) : 각 Tree의 가중치(weight) 추정을 허용하는 최대 Delta 단계
# colsample_bylevel(float) : 각 Tree의 Level에서 분할(split)에 대한 column의 Subsample 비율
#                           트리의 레벨별로 훈련 데이터의 변수를 샘플링해주는 비율. 보통 0.6~0.9
# early_stopping_rounds : 조기 중단을 위한 반복 횟수. N번 반복하는 동안 성능 평가 지표가 향상되지 않으면 반복이 멈춤
#                         조기종료를 위해 검증데이터와 같이 써야한다.
#                         eval_set = [(X_train,Y_train),(X_vld, Y_vld)]
#                         ex: early_stopping_rounds = 20 (20번 반복동안 최대화 되지 않으면 stop)


# 학습 태스크 파라미터
# objective : 학습하여 최솟값을 가져야할 손실 함수를 정의(4가지 설정 가능). XGB는 많은 유형의 손실함수를 사용할 수 있음. 주로 사용되는 손실함수는 이진 분류인지 다중 분류인지에 따라 달라짐.
#      'reg:linear' : 회귀의 경우
#      'binary:logistic' : 이진 분류일 때 적용함
#      'multi:softmax' : 다중 분류일 때 적용함. 손실함수가 multi:softmax일 경우에는 레이블 클래스의 개수인 num_class 파라미터를 지정해야 함.
#      'multi:softprob' : multi:softmax와 유사하나 개별 레이블 클래스의 해당되는 예측 확률을 반환합니다.
# eval_metric : 검증에 사용되는 함수를 정의합니다. 기본값은 회귀는 경우는 rmse, 분류일 경우에는 error입니다. 다음은 eval_metric의 값 유형입니다.
#       rmse: Root Mean Square Error
#       mae: Mean Absolute Error
#       logloss: Negative log-likehood
#       error: Binary classification error rate (0.5 threshold)
#       merror: Multiclass classification error rate
#       mlogloss: Multiclass logloss
#       auc: Area under the curve

    # 손실 함수(Loss function)
# - 손실함수는 알고리즘이 얼마나 잘못하고 있는지를 표현하는 지표이다.
#   값이 낮을수록 학습이 잘된 것. 정답과 알고리즘 출력을 비교하는 데에 사용한다.


    # XGB tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
xgb_clf = XGBClassifier(random_state=0, n_jobs=-1)
xgb_param_grid = { 'learniing_rate' : [0.05,0.1],
	               'n_estimators' : [100,200],
                   'max_depth' : [6,8],
                   'min_child_weight' : [6,8],
                   'gamma' : [0,5],
                   'colsample_bytree' : [0.5,1],
                   'subsample' : [0.5,1],
	     }

xgb_rs = RandomizedSearchCV(xgb,param_distributions = xgb_param_grid, n_iter = 10, cv=3 , scoring="accuracy", verbose = 1)
xgb_rs.fit(X_train,y_train)
 # xgb_rs.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)])
xgb_rs_best = xgb_rs.best_estimator_
print('RandomSearhCV 최적 파라미터:, xgb_rs.best_params_)
 # xgb_roc_score = roc_auc_score(y_test, pred)
 # print(f'ROC AUC: {xgb_roc_score:.4f}')

    # Best score
display(xgb_rs.best_score_, xgb_rs_best)

    # 피처 중요도 시각화
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_clf, ax=ax)



    # LGBMClassifier, LGBM
from lightgbm import LGBMClassifier(LGBMClassifier)
lgb_clf = LGBMClassifier(random_state=42, n_jobs=-1)
# lgb_clf = LGBMClassifier(n_estimators=1000, num_leaves=32, sub_sample=0.8, min_child_samples=100, max_depth=128, random_state=0, n_jobs=-1)
# eval = [(X_valid, y_valid)]
# lgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)
# lgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc', eval_set=evals, verbose=True)
pred = lgb_clf.predict(X_test)
get_clf_eval(y_test, pred)

lgb_score = lgb_clf.score(X_test, y_test)
print(f'lgb_score: {lgb_score}')
# lgbm_roc_score = roc_auc_score(y, test, lgb_clf.predict_proba(X_test)[:,1], average='macro')
# print(f'ROC AUC: {lgbm_roc_score:.4f}')

    # LGBM 파라미터
# - LightGBM 하이퍼 파라미터는 Xgb와 많은 부분이 유사
# 하지만 주의해야 할 점은 LightGBM은 xgb와 다르게 리프 노드가 계속 분할되면서 트리의 깊이가 깊어지므로 이러한 트리 특성에 맞는 하이퍼 파라미터 설정이 필요하다는 점임(예: max_depth를 매우 크게 가짐)

    # 주요 파라미터
# learning_rate[default=0.1] : 0에서 1사이의 값을 지정하며 부스팅 스텝을 반복적으로 수행할 때 업데이트되는 학습률값입니다.
#               일반적으로 n_estimators를 크게 하고 learning_rate를 작게 해서 예측 성능을 향상시킬 수 있으나, 마찬가지로 과적합 이슈와 학습 시간이 길어지는 부정적인 영향도 고려해야함.
# n_estimators[default=100]	 : 트리개수
# max_depth[default=-1]		 : 0보다 작게 지정하면 깊이에 제한이 없음. 지금까지 소개한 Depth wise 방식의 트리와 다르게 LightGBM은 Leaf wise 기반으로 깊이가 상대적으로 더 깊습니다.
# min_child_samples[default=20]	: 결정트리의 min_samples_leaf와 같은 파라미터. 최종 결정 클래스인 리프 노드가 되기 위해서 최소한으로 필요한 레코드 수. 과적합을 제어하기 위한 파라미터임.
# num_leaves[default=31]    	: 하나의 트리가 가질 수 있는 최대 리프 개수임.
# boosting[default=gbdt]		: 부스팅의 트리를 생성하는 알고리즘을 기술 (gbdt: 일반적인 그래디언트 부스팅 결정 트리, rf: 랜덤포레스트)
# sub_sample[default=1.0]		: 트리가 커져서 과적합 되는 것을 제어하기 위해서 데이터를 샘플링하는 비율을 지정함.
# colsample_bytree[default=1.0]	: 개별 트리를 학습할 때마다 무작위로 선택하는 피처의 비율임. 과적합을 막기 위해 사용됨. GBM의 max_features와 유사, xgb의 colsample_bytree와 똑같음.
# reg_lambda[default=0.0]		: L2 regulation 제어를 위한 값임. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과가 있음
# reg_alpha[default=0.0]		: L1 regulation 제어를 위한 값임. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과가 있음

    # Learning Task 파라미터
# objective: 최솟값을 가져야 할 손실함수를 정의. xgb의 objective 파라미터와 동일. 애플리케이션 유형, 즉 회귀, 다중 클래스 분류, 이진 분류인지에 따라서 objective인 손실함수가 지정됨.
# - 극도로 불균형한 레이블 값 분포도를 가지는 데이터에서 lgbm 객체 생성 시 boost_from_average=False로 파라미터를 설정해야 함
# - LightGBM이 버전업되면서 boost_from_average 파라미터의 디폴트 값이 False에서 True로 변경되었음. 레이블 값이 극도로 불균형한 분포를이루는 경우 boost_from_average=True 설정은 재현율 및 ROC-AUC 성능을 매우 크게 저하시킴.
#   2.1.0 이상의 버전이 설치되어 있거나 불균형한 데이터 세트에서 예측 성능이 매우 저조할 경우 LGBM 객체 생성 시 boost_from_average=False로 파라미터를 설정해야 함.

    # LGBM tuning
# - 랜덤서치를 통해 하이퍼 파라미터를 탐색하는 과정
# - 짧게 걸릴경우 단일모델 몇개로 스스로 범위 및 값을 조정할 수도 있음
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
lgbm = LGBMClassifier()
lgb_param_grid = {'num_leaves' : [70,80],
                  'min_data_in_leaf' : [100,200],
                  'max_depth' : [7,8],
	              'n_jobs' : -1}

LGB = RandomizedSearchCV(lgbm,param_distributions = lgb_param_grid, n_iter = 10, cv=10, scoring="accuracy", verbose = 1)
LGB.fit(X_train,y_train)
LGB_best = LGB.best_estimator_

    # plot_importance()를 이용해 피처 중요도 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(lgb+clf, ax=ax)



    # Logistic Regression
# - 독립변수와 종속변수의 선형 관계성에 기반한 로지스틱 회귀
# - 로지스틱 회귀의 경우 일반적으로 숫자 데이터에 스케일링을 적용하는 것이 좋습니다.
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_pred.predict(X_test)
print(f'LogisticRegression 정확도: {accuracy_score(y_test, lr_pred):.4f}')



    # LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
    # 모형 검토
print(model.coef_)
print(mode.intercept_)
    # 모형 평가
model.score(X_test, y_test) # R_square(r2_score)
y_pred = model.predict(X_test); y_pred
    # 분석 결과 시각화
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel(Measured')
plt.ylabel('Predicted')



    # KNN
# - 근접 거리를 기준으로 하는 최소 근접 알고리즘
# n_neighbors가 k임. k는 나랑 피쳐가 거리가 가까운 애를 몇개 선택할 건지
# weights 파라미터에서 uniform은 거리에 따른 가중치두지 않고 다 균등하게 voting하겠다는 것이고,
#                	          distance는 가까운애한테 더 가중치를 두겠다는 것임.
from sklearn.neignbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, n_jobs=-1)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

param_grid = {'n_neighbors': range(1, 10),
	          'p': range(1,5),
	          'metric': ['euclidean', 'manhattan', 'mahalanobis','minkowski'],
	          'weights': ['uniform', 'distance']}
knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', cv=5, n_jobs=-1)

neighbors = np.arange(1,9)
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k, n_jobs=-1)
    knn.fit(X_train, y_train)
    print(f'{i}th score: ', knn.score(X_test, y_test))



    # SVM
# - 개별 클래스 간의 최대 분류 마진을 효과적으로 찾아주는 서포트 벡터 머신
from sklearn.svm import SVC
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)



    # MLP
from sklearn.neural_network import MLPClassifier



    # GridSearch
# gridsearch는 왜 사용할까? - 교차검증과 하이퍼 파라미터튜닝을 동시에 할 수 있기 때문에.
# GridSearchCV 의 파라미터는 아래와 같다.
# estimator: 모델
# param_grid: key: list 구조를 갖는 딕셔너리, estimator 튜닝에 사용하는 파라미터 값
# scoring: 평가방법
# cv: 학습/테스트 세트 개수
# refit: 디폴트가 True,  True로 하면 가장 최적의 하이퍼 파라미터를 찾은 뒤 이걸로 재학습시킴.
# 즉, fit 이후 모델에 자동적으로 실험이 가장 성능이 우수한 모델이 저장된다.
from sklearn.model_selection import GridSearchCV
    # 파라미터 범위
rf_params = {'n_estimators':[10,20,50,100],
	         'criterion': ['gini', 'entropy'],
	         'max_depth': [3,6,9,12],
	         'max_features':[5,10,15,20]
}
    # 모델 적용 / 학습
rf_gs = GridSearchCV(rf_clf, param_grid = rf_params, scoring = 'accuracy', cv = 5, n_jobs=-1)
rf_gs.fit(X_train, y_train)

    # GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df = pd.DataFrame(rf_gs.cv_results_) # 훈련에 대한 결과들이 cv_results_아래에 있다. 데이터 프레임으로 만들어본것.
scores_df[['params', 'mean_test_score', 'rank_test_score',
	 'split0_test_score', 'split1_test_score', 'split2_test_score']]
# - params 칼럼에는 수행할 때마다 적용된 개별 하이퍼 파라미터값을 나타냄
# - rank_test_score는 하이퍼 파라미터별로 성능이 좋은 score 순위를 나타냄. 1이 가장 뛰어난 순위
# - mean_test_score는 개별 하이퍼 파라미터별로 cv의 폴딩 테스트 세트에 대해 총 수행한 평가 평균값

print('GridSearchCV 최적 파라미터: ', rf_gs.best_params_)
print(f'GridSearchCV 최고 정확도: {rf_gs.best_score_:.4f}')

    # GridSearchCV의 refit으로 이미 학습된 estimator  반환
best_rf = rf_gs.best_estimator_

    # best_rf는 이미 최적 학습이 됐으므로 별도 학습이 별요 없음
pred = best_rf.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, pred):.4f})



    # Random Search
from sklearn.model_selection import RandomizedSearchCV
params = {'criterion': ['gini', 'entropy'],
       	  'max_depth': [5,6,8,10],
          'min_samples_split': [2,3,5,8]}
rs = RandomizedSearchCV(DT_clf,
			param_distributions =params,
			scoring = 'accuracy',
			cv = 5,
			n_iter=8,
			n_jobs= -1,
			random_state=0)

    # 모델훈련 및 점수확인
rs.fit(cancer.data, cancer.target)
best_parameters = rs.best_params_
best_result = rs.best_score_
print(best_parameters)
print(best_result)
print("Best estimator:\n{}".format(rand_search.best_estimator_))

    # RandomSearhCV - Ex 2
clfs = [
   (
        KNeighborsClassifier(),              # 사용하려는 모델
        {'n_neighbors': [3,5,7,9,11],        # 최적화하려는 하이퍼파라미터
         'weights': ['uniform','distance']}
    ),
    (
        MLPClassifier(random_state=0),
        {'batch_size': [32, 64, 128],
         'learning_rate' : ['constant', 'adaptive'],
         'activation': ['tanh', 'relu'],
         'solver': ['sgd', 'adam']}
    ),
    (
        LogisticRegression(random_state=0),
        {'C': np.arange(0.1, 1.1, 0.1),
         'penalty': ['l1','l2']}
    ),
    (
        RandomForestClassifier(random_state=0),
        {'n_estimators': [100,200,300],
         'max_depth': [3,4,5],}
        # 'max_features': (np.arange(0.5, 1.0, 0.1)*X_train.shape[1]).astype(int)}
    ),
    (
        GradientBoostingClassifier(random_state=0),
        {'n_estimators': [100, 200, 300],
         'learning_rate': [1, 0.1, 0.01],}
        # 'max_features': (np.arange(0.5, 1.0, 0.1)*X_train.shape[1]).astype(int)}
    ),
    (
        XGBClassifier(tree_method = 'hist', random_state=0),
        {'min_child_weight': range(0, 121, 20),
         'learning_rate': np.arange(0.1, 0.6, 0.1),
         'subsample': np.arange(0.5, 1.0, 0.1)}
    ),
    (
        LGBMClassifier(random_state=0),
        {'min_child_weight': range(0, 121, 20),
         'learning_rate': np.arange(0.1, 0.6, 0.1),
         'subsample': np.arange(0.5, 1.0, 0.1)}
    ),
]

clfs_tuned = []  # 튜닝된 모델을 저장
for clf, param_grid in tqdm(clfs):
    start = time.time()
    rand_search = RandomizedSearchCV(clf, param_grid, n_iter=5, scoring='roc_auc',
                                     cv=3, random_state=0, n_jobs=-1)
    rand_search.fit(X_train, y_train)
    clf_name = type(clf).__name__
    clf_score = rand_search.score(X_valid, y_valid)
    print('{:30s} {:30f} {:.1f}'.format(clf_name, clf_score, time.time() - start))
    clfs_tuned.append((clf_name, rand_search, clf_score))
                                # rand_search안의 베스트 꺼내서 반환하도록 해야함



    # Bayesian Optimization, 베이지안 최적화
# 매 회 새로운 hyperparameter 값에 대한 조사를 수행할 시 '사전 지식'을 충분히 반영하면서, 동시에 전체적인 탐색 과정을 체계적으로 수행할 수 있는 방법이 Bayesian Optimization임.
# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

    # BO
# https://github.com/fmfn/BayesianOptimization

    # How does it work?
Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the
function you want to optimize. As the number of observations grows, the posterior distribution improves,
and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not,
as seen in the picture below.
 -> init_points 값을 많이 주면 더 능률이 오른다
 -> init_points 값 이후에 Gaussian Process and Utility Function

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account
what it knows about the target function. At each step a Gaussian Process is fitted to the known samples
(points previously explored), and the posterior distribution, combined with a exploration strategy
(such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that
 should be explored (see the gif below).
 ->  n_iter 값 많이 줄수록 최적값을 찾아간다


    # Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
)

    # The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is maximize, which does exactly what you think it does.
# There are many parameters you can pass to maximize, nonetheless, the most important ones are:
# init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
# n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
optimizer.maximize(
    init_points=2,
    n_iter=3,
)

    # The best combination of parameters and target value found can be accessed via the property `optimizer.max`
print(optimizer.max)
# >>> {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}

    # While the list of all parameters probed and their corresponding target values is available via the property `optimizer.res`
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
# >>> Iteration 0:
# >>>     {'target': -7.135455292718879, 'params': {'y': 1.3219469606529488, 'x': 2.8340440094051482}}
# >>> Iteration 1:
# >>>     {'target': -7.779531005607566, 'params': {'y': -1.1860045642089614, 'x': 2.0002287496346898}}
# >>> Iteration 2:
# >>>     {'target': -19.0, 'params': {'y': 3.0, 'x': 4.0}}
# >>> Iteration 3:
# >>>     {'target': -16.29839645063864, 'params': {'y': -2.412527795983739, 'x': 2.3776144540856503}}
# >>> Iteration 4:
# >>>     {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}


    # Changing bounds
During the optimization process you may realize the bounds chosen for some parameters are not adequate.
For these situations you can invoke the method set_bounds to alter them.
You can pass any combination of existing parameters and their associated new bounds.

optimizer.set_bounds(
                      new_bounds={"x": (-2, 3)}
                    )
optimizer.maximize(
    init_points=0,
    n_iter=5,
)

    # Saving, loading and restarting
By default you can follow the progress of your optimization by setting verbose>0 when instantiating the BayesianOptimization object.
If you need more control over logging/alerting you will need to use an observer.
For more information about observers checkout the advanced tour notebook.
Here we will only see how to use the native JSONLogger object to save to and load progress from files.

    # Saving progress
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

The observer paradigm works by:

Instantiating an observer object.
Tying the observer object to a particular event fired by an optimizer.
The BayesianOptimization object fires a number of internal events during optimization, in particular, everytime it probes the function and obtains a new parameter-target combination it will fire an Events.OPTIMIZATION_STEP event, which our logger will listen to.

Caveat: The logger will not look back at previously probed points.

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# Results will be saved in ./logs.json
optimizer.maximize(
    init_points=2,
    n_iter=3,
)


    # Loading progress
# Naturally, if you stored progress you will be able to load that onto a new instance of BayesianOptimization. The easiest way to do it is by invoking the load_logs function, from the util submodule.
from bayes_opt.util import load_logs
new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)

# New optimizer is loaded with previously seen points
load_logs(new_optimizer, logs=["./logs.json"]);































    # Correlation between models - Ex 1
clfs = []
LR=LogisticRegression(random_state=0); clfs.append(LR)
DT=DecisionTreeClassifier(random_state=0); clfs.append(DT)
MLP=MLPClassifier(random_state=0); clfs.append(MLP)
KNN=KNeighborsClassifier(); clfs.append(KNN)
RF=RandomForestClassifier(random_state=0); clfs.append(RF)
GBM=GradientBoostingClassifier(random_state=0); clfs.append(GBM)

pred_results = []
for clf in clfs :
    pred = clf.fit(X_train, y_train).predict(X_test)
    name = type(clf).__name__
    pred_results.append(pd.Series(pred, name=name))
    print(f'{name:30s}  {accuracy_score(y_test, pred):.5f}')

ensemble_results = pd.concat(pred_results, axis=1)
# 모형의 예측값 간의 상관관계를 보기 위해 hitmap을 도식한다.
# correlation 값이 떨어질수록 서로 독립이라는 것이다. 앙상블할때 봐야한다.
plt.figure(figsize=(8,6))
corr_graph = sns.heatmap(ensemble_results.corr(), annot=True, cmap='Blues')
corr_graph.set_title('Correlation between models')
plt.show()

    # Correlation between models - Ex 2
pred_results = []
for name, clf, clf_score in clfs_tuned:
    pred = clf.predict_proba(X_valid)[:,1]
    name = f'{name}\n({clf_score:.4f})'
    pred_results.append(pd.Series(pred, name=name))
ensemble_results = pd.concat(pred_results, axis=1)

plt.figure(figsize = (8,6))
g = sns.heatmap(ensemble_results.corr(), annot=True, cmap='Blues')
g.set_title("Correlation between models")
plt.show()





    # 회귀
# 머신러닝 회귀 예측 핵심: 주어진 피처와 결정 값 데이터 기반에서 학습을 통해 "최적의 회귀 계수"를 찾아내는 것
# - 독립 변수 1개: 단일 회귀 -> 선형 회귀
# - 독립 변수 여러 개: 다중 회귀 -> 비선형 회귀

    # 규제(Regularization)과 모델
# 규제는 일반적인 선형 회귀의 과적합 문제를 해결하기 위해서 회귀 계수에 패널티 값을 적용하는 것을 말함

    # 모델
# - 일반 선형 회귀: 예측값과 실제 값의 RSS(Residual Sum of Squares)를 최소화 할 수 있도록 회귀 계수를 최적화하며,
#                  규제(Regularzation)를 적용하지 않은 모델
# - 릿지(Ridge): 릿지 회귀는 선형 회귀에 L2 규제를 추가한 회귀 모델입니다. 릿지 회귀는 L2 규제를 적용하는데,
#   		     L2 규제는 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델입니다.
# - 라쏘(Lasso): 라쏘 회귀는 선형 회귀에 L1 규제를 적용한 방식입니다. L2규제가 회귀 계수 값의 크기를 줄이는데 반해,
#                L1 규제는 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 피처가 선택되지 않도록 하는 것입니다.
#                이러한 특성 때문에 L1 규제는 피처 선택 기능으로도 불립니다.
# - 엘라스틱넷(ElasticNet): L2, L1 규제를 함께 결합한 모델입니다. 주로 피처가 많은 데이터 세트에서 적용되며,
#                          L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정합니다.
# - 로지스틱 회귀(Logistic Regression): 로지스틱 회귀는 회귀라는 이름이 붙어 있지만, 사실은 분류에 사용되는 선형 모델입니다.
#                           로지스틱 회귀는 매우 강력한 분류 알고리즘입니다.
#                          일반적으로 이진 분류뿐만 아니라 희소 영역의 분류, 예를 들어 텍스트 분류와 같은 영역에서 뛰어난 예측 성능을 보입니다.



    # LinearRegression
# - LinearRegression 클래슨 예측값과 실제 값의 RSS(Residual Sum of Squares)를 최소화해 OLS(Ordinary Least Squares) 추정 방식으로 구현한 클래스입니다.
# - fit() 메서드로 X, y 배열을 입력 받으면 회귀 계수 (Coefficients)인 W를 coef_ 속성에 저장합니다.
# fit_intercept: 불린 값, 디폴트는 True. intercept(절편) 값을 계산할 것인지 말지를 지정함.
#	     만일 False로 지정하면 intercept가 사용되지 않고 0으로 지정됩니다.
# normalize: 불린 값으로 디폴트는 False입니다. fit_intercept가 False인 경우에는 이 파라미터가 무시됩니다.
#        만일 True이면 회귀를 수행하기 전에 입력 데이터 세트를 정규화합니다.

    # 속성
# coef_: fit()메서드를 수행했을 때 회귀 계수가 배열 형태로 저장하는 속성. Shape는 (Target 값 개수, 피처 개수)
# intercept_: intercept 값
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
from sklearn.linear_model import LinearRegression


    # 규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷
# 1. 릿지
# - Ridge 클래스의 주요 생성 파라미터는 alpha이며, 이는 릿지 회귀의 alpha L2 규제 계수에 해당합니다.


# 2. 라쏘
# - W의 절댓값에 페널티를 부여하는 L1 규제를 선형 회귀에 적용한 것이 라쏘(Lasso) 회귀입니다.
# - 즉 L1규제는 alpha * |W|를 의미하며, 라쏘 회귀 비용함수의 목표는 RSS(W) + alpha*|W| 식을 최소화하는 W를 찾는 것입니다.
# - L2 규제가 회귀 계수의 크기를 감소시키는데 반해, L1 규제는 불필요한 회귀 계수를 급격하게 감소히켜 0으로 만들고 제거합니다.
# - 이러한 측면에서 L1규제는 적절한 피쳐만 회귀에 포함시키는 피처 선택의 특성을 가지고 있습니다.


# 3. 엘라스틱넷
# - 엘라스틱넷(Elastic Net) 회귀는 L2규제와 L1규제를 결합한 회귀입니다.
# - 따라서 엘라스틱넷 회귀 비용함수의 목표는 RSS(W) + alpha2 * |W|^2 + alpha1 * |W|식을 최소화하는 W를 찾는 것입니다.
# - 엘라스틱넷은 라쏘 회귀가 서로 상관관계가 높은 피처들의 경우에 이들 중에서 중요 피처만들 셀럭션하고 다른 피처들은 모두 회귀 계수를 0으로 만드는 성향이 강합니다.
# - 특히 이러한 성향으로 인해 alpha값에 따라 회귀 계수의 값이 급격히 변동할 수도 있는데, 엘라스틱넷 회귀는 이를 완화하기 위해 L2규제를 라쏘 회귀에 추가한 것입니다.
# - 반대로 엘라스틱넷 회귀의 단점은 L1과 L2 규제가 결합된 규제로 인해 수행시간이 상대적으로 오래 걸린다는 것입니다.


    # 회귀 평가 지표
# 회귀의 평가를 위한 지표는 실제 값과 회귀 예측값의 차이 값을 기반으로 한 지표가 중심입니다.
# MAE: Mean Absolute Error(MAE)이며 실제 값과 예측값의 차이를 절댓값으로 변환해 평균한 것입니다.
# MSE: Mean Squared Error(MSE)이며 실제 값과 예측값의 차이를 제곱해 평균한 것입니다.
# RMSE: MSE 값은 오류의 제곱을 구하므로 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 씌운 것이 RMSE(Root Mean Squared Error)입니다.
# MSLE: MSE에 로그를 적용 (Mean Squared Log Error)
# RMSLE: RMSE에 로그를 적용 (Root Mean Squared Log Error)


    # 회귀 트리
# - 회귀 트리는 분류 트리와 크게 다르지 않습니다. 다만 리프 노드에서 예측 결정 값을 만드는 과정에 차이가 있는데,
# - 분류 트리가 특정 클래스 레이블을 결정하는 것과 달리 회귀 트리는 리프 노드에 속한 데이터 값의 평균값을 구해 회귀 예측값을 계산합니다.





    # 비지도학습
# - 사이킷런에서 비지도학습인 차원축소, 클러스터링, 피처추출(feature extraction) 등을 구현한 클래스 역시 대부분 fit()과 transfrom()을 적용합니다.
#   비지도학습과 피쳐 추출에서 fit()은 지도학습의 fit()과 같이 학습을 의미하는 것이 아니라, 입력 데이터의 형태에 맞춰 데이터를 변환하기 위한 사전 구조를 맞추는 작업입니다.
#   fit()으로 변환을 위한 사전 구조를 맞추면 이후 입력 데이터의 차원 변환, 클러스터링, 피쳐 추출 등의 실제 작업은 transform()으로 수행합니다.
#   사이킷런은 fit()과 transform()을 하나로 결합한 fit_transform()도 함께 제공합니다.



    # 군집 분석 - Kmeans
# 군집분석 알고리즘 원리 : 흩뿌려진 데이터에 대해서 랜덤으로 중심을 선택, 주변의 것들을 묶음,
# 묶은 것들 사이에서 다시 중심을 찍음 + 중심확인해서 새로 묶음, 다시 묶인 것들 사이에서 다시 중심을 찍음 + 새중심 확인해서 다시 묶음(반복)
from sklearn.cluster import KMeans
kmeans = Kmeans(n_clusters=10, n_jobs=-1, random_state=seed)
# n_clusters: 군집의 갯수
# init: 초기화 방법. "random"이면 무작위, "k-means++"이면 K-평균++방법, 또는 각 데이터의 군집 라벨, 디폴트는 k-means++
# n_init: 초기 중심위치 시도 횟수. 디폴트는 10이고 10개의 무작위 중심위치 목록 중 가장 좋은 값을 선택한다.
# max_iter: 최대 반복 횟수, 디폴트는 300
kmeans.fit(X)
kmeans.labels_



    # word2vec embedding (w2v)
from gensim.models import Word2Vec
embedding_model = Word2Vec(tokenized_contents, size=100, window = 2, min_count=50, workers=4, iter=100, sg=1)

# check embedding result
print(embedding_model.most_similar(positive=["디자인"], topn=100))



    # 빈발항목집합 추출, 연관규칙탐사, Apriori
# 연관규칙탐사 (Association Rule Learning)
# - 데이터 안에 존재하는 항목간의 종속 관계를 찾아내는 작업
# - 장바구니 분석(market basket analysis)
#   -> 고객의 장바구니에 들어있는 품목 간의 관계를 발견
# - 규칙의 표현: 항목 A와 품목 B를 구매한 고객은 품목 C를 구매한다.
# - 연관규칙의 활용: 제품이나 서비스의 교차판매, 매장진열, 첨부우편, 사기적발
# - 지지도: 그리고 이러한 경향을 가지는 사람들은 전체의 --% 정도이다.
from mlxtend.frequent_patterns import apriori, association_rules

# 지지도(support)가 5% 이상인 빈발항목집합(itemsets)만 추출하고 지지도 기준 내림차순으로 출력
freq_items = apriori(transactions, min_support=0.05, use_colnames=True)

# min_support 최소 지지도
freq_items.sort_values(by='support', ascending=False)

# apriori
apriori(
    df,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    low_memory=False,
)

# 신뢰도(confidence)가 85% 이상인 연관규칙만 출력: 지지도가 5프로 넘는 애들중에 신뢰도가 85프로 넘는 애들 출력하는 것
rules = association_rules(freq_items, metric='confidence')
rules.query('confidence >= 0.85')



──────── 앙상블(Ensemble) ────────


    # Ensemble
# - 앙상블 학습의 유형은 전통적으로 보팅(Voting), 배깅(Bagging), 부스팅(Boosting)의 세 가지로 나눌 수 있으며, 이외에도 스태킹을 포함한 다양한 앙상블 방법이 있습니다.
# - 보팅과 배깅은 여러 개의 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식입니다.
# - 보팅과 배깅의 다른점은 보팅의 경우 일반적으로 서로 다른 알고리즘을 가진 분류기를 결합하는 것이고, 배깅의 경우 각각의 분류기가 모두 같은 유형의 알고리즘 기반이지만, 데이터 샘플링을 서로 다르게 가져가면서 학습을 수행해 보팅을 수행하는 것입니다.
# - 배깅은 학습 데이터를 중복을 허용하면서 다수의 세트로 샘플링하여 이를 다수의 약한 학습기가 학습한 뒤 최종 결과를 결합해 예측하는 방식입니다.
# - 부스팅은 여러 개의 분류기가 순차적으로 학습을 수행하되, 앞에서 학습한 분류기가 예측이 틀린 데이터에 대해서는 올바르게 예측할 수 있도록 다음 분류기에게는 가중치(weight)을 부여하면서 학습과 예측을 진행하는 것입니다. 계속해서 분류기에게 가중치를 부스팅하면서 학습을 진행하기에 부스팅 방식으로 불립니다.
# - 스태킹은 여러 가지 다른 모델의 예측 결괏값을 다시 학습 데이터로 만들어서 다른 모델(메타 모델)로 재학습시켜 결과를 예측하는 방법입니다.
# -  Model의 결과값을 앙상블할 때, 고려해야하는 가장 큰 2가지: 모델의 성능, 모델 간 이질성

    # Voting Ensemble
# 서로 다른 모델을 연결하여 Voting해주는 방법
# - 보팅 방법에는 두 가지 방법이 있습니다. 하드 보팅과 소프트 보팅입니다.
# - 하드 보팅을 이용한 분류는 다수결 원칙과 비슷합니다. 예측한 결괏값들중 다수의 분류기가 결정한 예측값을 최종 보팅 결괏값으로 선정하는 것입니다.
# - 소프트 보팅은 분류기들의 레이블 값 결정 확률을 모두 더하고 이를 평균해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값으로 선정합니다. 일반적으로 소프트 보팅이 보팅방법으로 적용됩니다.
from sklearn.ensemble import VotingClassifier
    #1. hardVoting
vo_clf = VotingClassifier(estimators = [('xgb', xgb), ('tree', tree), ('knn', knn)], voting = 'hard')

# 리스트 컴프리핸션 이용
vo_clf = VotingClassifier(
    estimators = [(type(clf).__name__, clf) for clf in clfs], voting='hard')

vo_clf.fit(X_train, y_train).score(X_test, y_test)
pred = vo_clf.predict(X_test)
print(f'Voting 분류기 정확도: {accuracy_score(y_teset, pred):.4f}')


    # 2. SoftVoting: Averaging predictions
# - 평가지표가 roc-auc, logloss 등일 경우 사용
# - 산술평균으로 계산됨
vo_clf = VotingClassifier(
    estimators = [('xgb', xgb), ('tree', tree), ('knn', knn)], voting = 'soft')
vo_clf.fit(X_train, y_train)
print('AUC =', roc_auc_score(y_test, vo_clf.predict_proba(X_test)[:,1]))

# 개별 모델의 학습 / 예측 평가
for clf in (xgb, tree, knn, vo_clf):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    class_name = clf.__class__.__name__
    print(f'{class_name} 정확도: {accuracy_score(y_test, pred):.4f}')




    # Stacking
# 서로 다른 모델의 결과값을 바탕으로 다시 한 번 모델링을 해주는 방법이다.
# 모델별로 도출된 예측 레이블 값을 다시 합해서(스태킹) 새로운 데이터 세트를 만들고 이렇게 스태킹된 데이터 세트에 대해 최종 모델을 적용해 최종 예측을 하는 것이 스태킹 앙상블 모델이다.
# 따라서 Meta-Classifier가 필요하다

#!pip install mlxtend
from vecstack import stacking

    # vecstack - stacking, 2-layer stacking
models = clfs
S_train, S_test = stacking(models,
		        X_train, y_train, X_test,  # data
		        regression=False,	  # classification task (if you need regression - set to True) , 분류문제를 하겠다
		        needs_proba=False,	  # predict class labels (if you need probabilities - set to True), 확률값을 낼 것이냐
		        metric=accuracy_score, # metric: callable
		        n_folds=5,		  # number of folds
		        stratified=True,  # stratified split for folds
		        shuffle=True,	  # shuffle the data
		        random_state=0,	  # ensure reproducibility
		        verbose=2)		  # verbose: int, default 0 -> Level of verbosity
		        		        		# 0 - show no messages
		        		        		# 1 - for each model show mean score
		        		        		# 2 - for each model show score for each fold and mean score
meta_model = GBM.fit(S_train, y_train)
accuracy_score(y_test, meta_model.predict(S_test))

    # 3-layer stacking
# level-1: LR, DT, MLP, KNN, RF, GBM
models = clfs
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,
                           regression=False,
                           needs_proba=True,
                           metric=accuracy_score,
                           n_folds=3,
                           stratified=True,
                           shuffle=True,
                           random_state=0,
                           verbose=0)
# level-2: LR, DT, KNN
# level-3: Voting
voting = VotingClassifier(estimators = [('lr', LR), ('dt', DT), ('knn', KNN)], voting='hard')
voting.fit(S_train, y_train).score(S_test, y_test)

    # 모형을 다양하게 조합하여 stacking한 후 최적의 조합 찾기
# 모델도 받도록 수정하면 좋음
from itertools import combinations

results = []

for num_of_clfs in range(3, 6):
    for models in combinations(clfs, num_of_clfs):
        S_train, S_test = stacking(models,
			    X_train, y_train, X_test,
			    regression=False,
			    needs_proba=True,
			    metric=None,
			    n_folds=4,
			    stratified=True,
			    shuffle=True,
			    random_state=0,
			    verbose=0)
        meta_model = LR.fit(S_train, y_train)
        combi = [type(i).__name__ for i in models]
        score = roc_auc_score(y_test, meta_model.predict_proba(S_test)[:,1])
        results.append(score, combi))

print(max(results))



    # StackingTransformer
from vecstack import StackingTransformer
selected = [
    'KNeighborsClassifier',
    'MLPClassifier',
    'LogisticRegression',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
]
estimators = [(name, clf) for name, clf, _ in clfs_tuned if name in selected]

# Initialize StackingTransformer
stack = StackingTransformer(estimators,
                            regression=False,
                            needs_proba=True,
                            metric=None,
                            n_folds=3,
                            stratified=True,
                            shuffle=True,
                            random_state=42)
# Fit
stack = stack.fit(X_train, y_train)

# Get your stacked features
S_train = stack.transform(X_train)
S_valid = stack.transform(X_valid)
S_test = stack.transform(X_test_sel)

# Use 2nd level estimator with stacked features
meta_model = LogisticRegression().fit(S_train, y_train)
print(roc_auc_score(y_valid, meta_model.predict_proba(S_valid)[:,1]))



    # StackingClassifier
    # 조정가능 파라미터
# (classifiers, meta_classifier, use_probas=False,
#  drop_last_proba=False, average_probas=False, verbose=0,
#  use_features_in_secondary=False, store_train_meta_features=False,
#  use_clones=True,
# )

from mlxtend.classifier import StackingClassifier
    # StackingClassifier - Ex 1
stacking_1 = StackingClassifier(classifiers=[tree, rf, knn],
                              meta_classifier= logreg, # blender or meta-learner
                              use_probas=False,
                              average_probas=False)

for clf in (tree, xgb, knn, stacking_1) :
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, accuracy_score(
        y_test, clf.predict(X_test)))

    # StackingClassifier - Ex 2
stacking_2 = StackingClassifier(classifiers=[tree, rf, xgb],
                              meta_classifier= logreg, # blender or meta-learner
                              use_probas=False,
                              average_probas=False)

for clf in (tree, xgb, rf, stacking_2) :
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, accuracy_score(
        y_test, clf.predict(X_test)))



    # Ensemble 성능평가
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz = []
accuracy = []
std = []
classifiers = ['Voting','Averaging','Stacking']
models = [voting, averaging, stacking_1]

for model in models:
    cv_result = cross_val_score(model, X_train, y_train, cv = kfold, scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)

models_dataframe = pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)
plt.subplots(figsize=(12,6))
box = pd.DataFrame(accuracy,index=classifiers)
box.T.boxplot()
plt.show()



    # 튜닝된 모델들로 평균내서 제일 좋은 조합 찾기
selected = [
    'KNeighborsClassifier',
    'MLPClassifier',
    'LogisticRegression',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
]
models_for_ensemble = [clf for clf in clfs_tuned if clf[0] in selected]
max_score = 0
for p in tqdm([0, 1, 2.56]):  # p==1:산술평균, p=0:기하평균, 그 외:멱평균(주의:멱평균은 과적합 가능성이 높음)
    for i in range(2, len(models_for_ensemble)+1):
        for models in combinations(models_for_ensemble, i):
            if p == 0:
                pred_mean = gmean([clf.predict_proba(X_valid)[:,1] for name, clf, _ in models], axis=0)
            else:
                preds = [clf.predict_proba(X_valid)[:,1] for name, clf, _ in models]
                pred_mean = (np.sum(np.array(preds)**p, axis=0) / len(models))**(1/p)
            score = roc_auc_score(y_valid, pred_mean)
            if max_score < score:
                best_avg_ensemble = (p, models, score)
                max_score = score

p, models, score = best_avg_ensemble
print('p={}\n{}\n{}'.format(p, '●'.join([clf_name for clf_name, _, _ in models]), score))



    # 평균의 종류
# 산술평균 = (a+b) / 2
# 기하평균 = √(ab)
# 조화평균 = 2ab / (a+b)

    # 기하평균 - Ex 1
from scipy.stats.mstats import gmean
pred_logreg = LR.fit(X_train, y_train).predict_proba(X_test)[:,1]
pred_tree = DT.fit(X_train, y_train).predict_proba(X_test)[:,1]
pred_knn = KNN.fit(X_train, y_train).predict_proba(X_test)[:,1]
print('AUC = ', roc_auc_score(y_test, gmean([pred_logreg, pred_tree, pred_knn], axis=0)))



    # save model, 모델 저장
import joblib
joblib.dump(Extra_clf, 'Extra_clf_bayesian.pkl')







──────── 성능검증 ────────

    # 모형평가 - Ex 0
    # 정예실모과 재실예데
1. Precision : 정밀도
 - 모델이 True라고 분류한 것 중에서 실제 True인 것의 비율
 - 모델이 예측한 것 중에서 실제 True인 것의 비율
 - 정 + 예 + 실 = 정예실

2. Recall : 재현율
 - 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
 - 재 + 실 + 예 = 재실예

3. 예시와 관점
 - 정예실(Precision): 날씨 예측을 모델이 맑다로 예측했는데, 실제 날씨가 맑았는지 살펴볼때 사용
 - 재실예(Recall): 실제로 날씨가 맑은 날 중에서 모델이 맑다고 예측한 비율
 - 모두 실제 True인 정답을 모델이 True라고 예측한 경우데 관심이 있으나,
   바라보고자 하는 관점이 다름. Precision은 모델의 입장에서, 그리고 Recall은 실제 정답(data)의 입장에서 정답을 맞춘 경우임.

4. 결론
 - 정예실 + 모(델의 입장), 재실예 + 데(이터)의 입장
 - (정예실모)와 (재실예데)를 기억하자!
 - 참고 url : https://sumniya.tistory.com/26




    # predict_proba() 메서드
# - predict_proba() 는 학습이 완료된 사이킷런 Classifier 객체에서 호출이 가능하며 테스트 피처 데이터 세트를 파라미터로 입력해주면 테스트 피처 레코드의 개별 클래스 예측 확률을 반환합니다.
# - predict() 메서드와 유사하지만 단지 반환 결과가 예측 결과 클래스값이 아닌 예측 확률 결과입니다.
# - 반환값:
#  개별 클래스의 예측 확률을 ndarraay m x n (m: 입력 값의 레코드 수, n: 클래스 값 유형) 형태로 반환
#  각 열은 개별 클래스의 예측 확률입니다. 이진 분류에서 첫번째 칼럼은 0 Negative의 확률, 두번째 칼럼은 1 Positive의 확률입니다.
#  즉 이진분류에서 predict_proba()를 수행해 반환되는 ndarray는 첫번째 칼럼이 클래스 값 0에대한 예측확률, 두번째 칼럼이 클래스 1에 대한 예측 확률입니다.


    # Binarizer 클래스
# - threshold 변수를 특정 값으로 설정하고 Binarizer 클래스를 객체로 생성
# - 생성된 Binarizer객체의 fit_transform() 메서드를 이용해 넘파이 ndarray를 입력하면 입력된 ndarray의 값을 지정된 threshold보다 같거나 작으면 0값으로, 크면 1값으로 변환해 반환합니다.
from sklearn.preprocessing import Binarizer
X = [[1, -1, 2],
      [2, 0, 0],
      [0, 1.1, 1.2]]
# X의 개별 원소들이 threshold값보다 같거나 작으면 0을, 크면 1을 반환
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

    # predict_proba()로 추출한 예측 결과 확률값을 변환해 변경된 임곗값에 따른 예측 클래스 값을 구하기
# 임곗값을 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold=0.48)
# 앞에서 구한 lr_clf의 predict_proba() 예측 확률 array에서 1에 해당하는 칼럼값을 Binarizer 변환
pred_th_048 = binarizer.fit_transform(pred_proba[:,1].reshape(-1, 1))
get_clf_eval(y_test, pred_th_048)

# 임계값을 조절하며 평가 지표를 조사
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
def get_eval_by_threshold(y_test, pred_proba_c1, threshold):
# thresholds list객체 내의 값을 차례로 iteration하면서 Evaluation 수행
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(pred_proba_c1)
        predict = binarizer.transform(pred_proba_c1)
        print('임곗값:', threshold)
        get_clf_eval(y_test, predict)

get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)



    # 모형평가 - Ex 1
    # 정확도(Accuracy)
# - 정확도는 실제 데이터에서 예측 데이터가 얼마나 같은지를 판단하는 지표입니다.
# - 정확도(accuracy) = 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수
# - 정확도는 불균형한(imbalanced) 레이블 값 분포에서 ML 모델의 성능을 판단한 경우, 적합한 평가 지표가 아닙니다.
from sklearn.metrics import accuracy_score

    # 오차행렬 (Confusion Matrix)
# - 성능지표로 잘 활용되는 오차행렬은 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고(confused) 있는지도 함께 보여주는 지표입니다.
#  즉, 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표입니다.
# - 오차행렬은 4분면 행렬(TN, FP, FN, TP)에서 실제 레이블 클래스 값과 예측 레이블 클래스 값이 어떠한 유형을 가지고 매핑되는지를 나타냅니다.
#   TN, FP, FN, TP 기호가 의미하는 것은 앞 문자 True/False는 예측값과 실제값이 '같은가/틀린가'를 의미합니다.
#   뒤 문자 Negative/Positive는 예측 결과 값이 부정(0)/긍정(1)을 의미합니다.
# TN는 예측값을 Negative 값 0으로 예측했고 실제 값이 역시 Negative 값 0
# FP는 예측값을 Positive 값 1로 예측했는데 실제 값은 Negative 값 0
# FN은 예측값을 Negative 값 0으로 예측했는데 실제 값은 Positive 값 1
# TP는 예측값을 Positive 값 1로 예측했는데 실제 값 역시 Positive 값 1
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_tree))
confusion_matrix(y_test, best_rf.predict(X_test))

    # 정밀도(Precision)
# - 정밀도 = TP / (FP + TP)
# - 정밀도는 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치 데이터의 비율을 뜻합니다.
# - 분모인 FP + TP는 예측을 Positive로 한 모든 데이터 건수이며, 분자인 TP는 예측과 실제값이 Positive로 일치한 데이터 건수입니다.
# - 정밀도가 중요 지표인 경우는 실제 Negative 음성인 데이터 예측을 Positive 양성으로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우
#	EX: 스팸메일 여부를 판단하는 모델

    # 제현율(Recall)
# - 재현율 = TP / (FN + TP)
# - 재현율은 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율을 뜻합니다.
# - 분모인 FN + TP는 실제 값이 Positive인 모든 데이터 건수이며, 분자인 TP는 예측값이 실제 값이 Positive로 일치한 데이터 건수입니다.
# - 민감도(Sensitivity) 또는 TRP(True Positive Rate)라고도 불립니다.
# - ★재현율이 중요 지표인 경우는 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우입니다. (보통 재현율이 정밀도보다 상대적으로 중요한 업무가 많음)
#	EX: 암 판단 모델, 보험사기와 같은 금융 사기 적발 모델
# - 가장 좋은 성능평가는 재현율과 정밀도 모두 높은 수치를 얻는 것임. 둘 중 어느 한평가만 매우 높고, 다른 수치는 매우 낮은 결과를 나타내는 경우는 바람직하지 않음.
# - 예측의 임계값을 변경함에 따라 정밀도와 재현율의 수치가 변경됨.
# - 임계값의 이러한 변경은 업무 환경에 맞게 두 개의 수치를 상호 보완할 수 있는 수준에서 적용돼야 함.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precison = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    AUC = roc_auc_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {AUC:.4f}')
EX)
 lr_clf.fit(X_train, y_train)
 pred = lr_clf.predict(X_test)
 get_clf_eval(y_test, pred)


    # precision_recall_curve()
# - precision_recall_curve() API는 정밀도와 재현율의 임계값에 따른 값 변화를 곡선 형태의 그래프로 시각화하는 데 이용할 수 있습니다.
from sklearn.metrics import precision_recall_curve
def precision_recall_curve_plot(y_test, pred_proba_c1):
# threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

# X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8, 6))
    threshold_boundary = threshold.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

# threshold 값 X축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

# X축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:,1])


    # F1 스코어
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
print(f'F1 스코아: {f1:.4f}')


    # ROC 곡선과 AUC
# - ROC 곡선은 FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선입니다. FPR을 X축으로, TPR을 Y축으로 잡으면 FPR의 변화에 따른 TPR의 변화가 곡선 형태로 나타납니다.
# - TPR은 재현율을 나타냅니다. 민감도로도 불립니다.
# - 민감도에 대응하는 지표로 TNR이라고 불리는 특이성이 있습니다. FPR은 1 - FNP 또는 1 - 특이성으로 표현됩니다.
# - ROC 곡선이 가운데 직선에 가까울수록 성능이 떨어지는 것이며, 멀어질수록 성능이 뛰어난 것입니다.
# - roc_curve()
#	입력파라미터: y_true, y_score
#	반환 값: fpr, tpr, threshold (모두 arraty로 반환)
from sklearn.metrics import roc_curve
def roc_curve_plot(y_test, pred_proba_c1):
#  임곗값에 따른 FPR, TPR 값을 반환받음
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
# ROC 곡선을 그래프 곡선으로 그림
    plt.plot(fprs, tprs, label='ROC')
# 가운데 대각선 직선을 그림
    plt.plot([0,1], [0,1], 'k--', label='Random')

# FPR X축의 Scale을 0.1 단위로 변경, X, Y축 명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel('FPR(1-Sensitivity)'); plt.ylabel('TRP(Recall)')
    plt.legend()

roc_curve_plot(y_test, pred_proba[:, 1])

    # roc_auc_score **
# - 일반적으로 ROC 곡선 자체는 FPR과 TPR의 변화 값을 보는데 이용하며 분류의 성능지표로 사용되는 것은 ROC곡선 면적에 기반한 AUC 값으로 결정합니다.
# - ROC-AUC는 일반적으로 이진 분류의 성능 평가를 위해 가장 많이 사용되는 지표임.
from sklearn.metrics import roc_auc_score
pred = lf_clf.predict(X_test)
roc_score = roc_auc_score(y_test, pred)
print('ROC AUC 값: {roc_score:.4f}')


    # Plot multiple ROC curves
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, dummy.predict_proba(x_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree.predict_proba(x_test,)[:,1])
plot_roc_curve(tpr_tree, tpr_tree, 'decision tree', 'darkgreen')
EX2)
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, dummy.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy')

fpr_dummy, tpr_dummy, _ = roc_curve(y_test, logit.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'logit')

fpr_tree, tpr_tree, _ = roc_curve(y_test, tree.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'tree')

fpr_tree, tpr_tree, _ = roc_curve(y_test, knn.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'k-NN')


    # classification_report
from sklearn.metrics import classification_report
print('\nDecision tree:')
print(classification_report(y_test, pred_tree, target_names=['not 9', '9']))



    # 모형평가 - Ex 2
# Accuracy vs Precision vs Recall
# - Accuracy: (TP + TN) / (TP + TN + FP + FN)
#     정확도 : 예측이 정답과 얼마나 정확한가?

# - Precision: TP / (TP + FP)
#     정밀도-Precision: 예측한 것중에 정답의 비율은?
#     FP 를 줄이는 것이 목표일 때는 precison을 주로 사용

# - Recall(=Sensitivity): TP / (TP + FN)
#     재현율-Recall: 찾아야 할 것중에 실제로 찾은 비율은?
#     FN 을 줄이는 것이 목표일 때는 recall을 주로 사용

# precison과 recall은 trad-off 관계이기 때문에, 전반적으로 두 성능지표를 고려하거나
# - F1-score:
#     F1 Score : 정밀도와 재현율의 평균
#     y의 클래스가 불균형인 경우에는 이 둘을 조화 평균한 값인 F1-score을 많이 사용


    # ROC & AUC
# ROC curve: Receiver Operating Characteristic curve
# - false positive rate(1-specificity)를 x축으로, true positive rate(recall)를 y축으로 하여
#   둘 간의 관계를 표현한 그래프

# AUC
# - ROC curve의 밑부분 면적(area uder the ROC curve; AUC)이 넓을수록 모형 성능이 높아짐
# - Thumb rules: Poor model(0.5~0.7), Fair model(0.7~0.8), Good model(0.8~0.9), Excellent model(0.9~1.0)





──────── 모형 해석 ────────






──────── 기타 ────────

    # 구글 코랩 런타임 끊김 방지 - Ex1
# - 구글 코랩에서 F12로 개발자 도구창을 열고 Console 선택 후 아래의 코드를 입력한 뒤 엔터를 누르면됩니다.
function ClickConnect() {var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); buttons.forEach(function(btn) { btn.click(); }); console.log("1분마다 자동 재연결"); document.querySelector("colab-toolbar-button#connect").click(); } setInterval(ClickConnect,1000*60);

    # 구글 코랩 런타임 끊김 방지 - Ex2
function ClickConnect(){
    console.log("코랩 연결 끊김 방지");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60 * 1000)





---------- 머신러닝 여러 설명 ----------


    # 머신러닝이 무엇인가?
# - 머신러닝을 전통적인 프로그래밍과 비교하여 간단히 설명하여 보겠습니다.
#   이전의 전통적인 프로그래밍은 컴퓨터에 어떠한 규칙과 그에 해당하는 데이터를 대입하여 정답이 나오는 형식이었습니다.
#   이와 비교해서 머신러닝은 거꾸로 이미 나와있는 정답을 입력하여 어떤 일련의 규칙이 출력되는 것이라 설명할 수 있을 것 같습니다.



    # 과대적합(overfitting)
    # 과적합 설명 - Ex 1
# - 모델이 데이터에 필요이상으로 적합한 모델
# - 데이터 내에 존재하는 규칙 뿐만 아니라 불완전한 레코드도 학습

    # 과적합 설명 - Ex 2
# - 과적합은 모델이 학습 데이터에만 과도하게 최적화되어,
#   실제 예측을 다른 데이터로 수행할 경우에는 예측 성능이 과도하게 떨어지는 것을 말합니다.

    # 과소적합(underfitting)
# - 모델이 데이터에 제대로 적합하지 못한 모델
# - 데이터 내에 존재하는 규칙도 제대로 학습하지 못함



    # 머신러닝의 한계
# - 과적합(overfitting) 또는 과도한 일반화(overgeneralitzation) 문제
# - 정답이 있는 대량 데이터 필요
# - 도출 결과의 설명력 부족(Explainability problem)
# - 기존 학습 모델의 재사용 어려움(Domain complexity)
#    Ex) 과거 학습된 내용(금융분야)을 다른 영역(법률분야)에 적용하지 못함

# - 샘플링 잡음 및 편향 문제(Sampling noise or Bias)
# - 모델 성능 약화(Model performance deterioration)
#    Ex) 과거 현상들이 더 이상 미래에 유사하게 일어나지 않을 때 발생

# - 데이터 유출(Data leakage)
#    Ex) 부동산 가격을 예측할 때 매입 이후 지불하는 인지세 및 부동산 수수료와 같은 데이터를 입력 데이터에 포함시키는 경우



    # Hyperparameter Optimization
# - 학습을 수행하기 위해 사전에 설정해야 하는 값이 hyperparameter의 최저값을 탐색하는 문제.
#   여기서, 최적값이란 학습이 완료된 러닝 모델의 일반화 성능을 최고 수준으로 발휘하도록 하는 hyperparameter 값을 의미

    # 하이퍼파라미터 최적화 기법 종류
# - Manual Search

# - Grid Search
#   탐색의 대상이 되는 특정 구간 내의 후보 hyperparameter 값들을 일정한 간격(grid)을 두고 선정하여,
#   이들 각각에 대하여 성능 결과를 측정한 후 가장 높은 성능을 발휘했던 hyperparameter 값을 최적값으로 선정하는 방법

# - Random Search
#   탐색 대상 구간 내의 후보 hyperparameter 값들을 랜덤 샘플링을 통해 선정.
#   Grid Search에 비해 불필요한 반복 수행 횟수를 대폭 줄이면서 동시에 정해진 간격 사이에 위치한 값들에 대해서도
#   확률적으로 탐색이 가능하므로, 최적값을 더 빨리 찾을 수 있는 것으로 알려져 있음

# - Bayesian Optimization
#   매 회 새로운 hyperparameter 값에 대한 조사를 수행할 때에 "사전 지식"을 충분히 반영하면서
#   동시에 전체적인 탐색 과정을 좀 더 체계적으로 수행하는 방법



    # 모형평가의 개념 및 고려사항
    # 모형평가란
# - 고려된 서로 다른 모형들 중 어느 것이 가장 우수한 예측력을 보유하고 있는지,
#   선택된 모형이 '임의의 모형(random model)'보다 우수한지 등을 비교하고 분석하는 과정을 말한다.
# - 이 때 다양한 평가지표와 도식을 활용하는데, 머신러닝 애플리케이션의 목적이나 데이터 특성에 따라
#   적절한 성능지표(performance measure)를 선택해야 한다.

    # 모형 선택 시 고려사항
# - (일반화 가능성): 같은 모집단 내의 다른 데이터에 적용하는 경우 얼마나 안정적인 결과를 제공해 주는가?
# - (효율성) 얼마나 적은 feature를 사용하여 모형을 구축했는가?
# - (정확성) 모형이 실제 문제에 적용될 수 있을 만큼 충분한 성능이 나오는가?



    # 변동계수(coefficient of variation)
# • 측정단위가 서로 다를 때(예: 킬로그램과온스), 또는 평균이 서로 다를 때 데이터의 퍼진 정도를 비교하기 위한 것이 변동계수(CV)이다.
# • 즉, 변동계수는 단위와 무관한 측도이다.
# • 공식에서 보는 것처럼 CV는 표준편차가 평균의 몇 퍼센트인지를 나타낸다.
# • 어떤 데이터세트의 경우 표준편차가 평균보다 큰 경우도 있기 때문에 CV가 100%를 넘을 수도 있다.
# • 서로 다른 단위로 측정된 변수들을 비교할때 CV가 유용하다.
# • 단, 평균이 0, 또는 음수(-)이면 정의되지 않으므로 양수(+)의 데이터에만 가능하다.



    # 표본데이터의 표준편차 추정(Estimating Sigma)
# • 정규분포는 거의 모든 관측치가 μ ± 3σ 이내에 있다. 즉, 관측치의 범위는 대략 6σ 이내이다.
# • 따라서 표본데이터의 범위 ‘Xmax − Xmin’을 안다면 표준편차는 ‘s = (Xmax − Xmin)/6’ 으로 추정된다.
# • 이 규칙은 표본데이터의 범위만을 알고 있을 때 표준편차를 대략 손쉽게 구할 수 있는 방법이다.
# • 물론 이 표준편차 추정치는 표본데이터가 정규분포를 따른다는 가정을 바탕으로 하고 있다.



    # < 기초통계 >
평균: 산술평균, 기하평균, 중앙값(중위수), 최빈값, 조화평균, 멱평균
변동성: 데이터 값들이 중심에서 퍼져있는 정도
 -> 변동계수: s / x_bar * 100% : 표준편차가 평균의 몇퍼센트인지 보는 것
표준화(standardized value): Z값 : 관측치와 평균과의 거리가 표준편차의 몇 배인지를 나타낸다.
 		-> z값은 특정한 관측치가 평균으로부터 얼마나 떨어져 있는지를 파악하는 방법이 된다. (Zi = (xi - mu) / sigma, 표본인경우 Zi = (xi - x_bar) / s)
백분위수(percentile): 데이터를 100개의 그룹으로 나눈 것이다
공분산(covariance): 두 변수 X와 Y의 공분산은 X와 Y가 같은 방향으로 변하는 정도를 측정
상관계수(correlation coefficient): X와 Y의 공분산을 각각의 표준편차의 곱으로 나눈 것.
                                  이들간의 선형적인 관련성을 측정하는 대표적인 통계량임




---------- 머신러닝 혼합 아이디어  ----------



    # 지도학습 비지도학습 혼합 아이디어(실제 해본 적 있는 것들)
 - PCA 사용시 보통 99퍼로 가져가지만, PCA 과정을 하나의 피처 생성과정으로 보고 PCA 90%퍼로 추출해서 원본 피처에 부착해서 사용
 - 비지도학습(Ex: Kmeans)로 이용고객 여러 개의 집단으로 묶어서(150~170여개 정도) predict_label값을 피처로써 사용



    # 머신러닝 딥러닝 혼합 아이디어(실제 해본 적 있는 것들)
 - AE(혹은 DAE)로 train 피처를 학습시킴, 학습시킨 AE의 Encoder로 train 피처와 test 피처를 predict하여 compressed_representation을 각각 뽑음
   compressed_representation된 X_train, X_test를 DNN으로 학습시켜 submission까지 추출

 - 피처의 사진화: 평범한 피처의 각 행을 각각 하나의 사진으로써 보고 사진을 쌓아올리는 과정으로 사진묶음(3D Data)을 만들어서 CNN을 활용해 예측

 - 'Featurizer'란 단어로 검색해서 다양한 모델, 기법을 활용하여 피처를 만드는 방법 써보는 것도 정말 색다른 피처를 만드는 좋은 방법이었음
    EX: KMeansFeaturizer, EmbeddingVectorizer






---------- Import Codes ----------

    # basic import codes
# Data Handling
import pandas as pd
import numpy as np
from tqdm import tqdm


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)


# Data Engineering
from sklearn.preprocessing import LabelEncoder


# Modeling & Tuning
from sklearn.model_selection import KFold
n_splits=5; seed = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


# Ensemble
from itertools import combinations
from sklearn.ensemble import VotingRegressor
from mlxtend.classifier import StackingRegressor


# Evaluation
from sklearn.metrics import mean_absolute_error


####################################################


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classfication_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import gmean

    # Cross Validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro') # model, train, target, cross validation
print('cross-val-score \n{}'.format(scores))
print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))


    # DT
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DeicisionTreeRegressor
    # RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
    # Extra
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
    # GBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
    # XGB
from xgboost import XGBClassifier
from xgboost import XGBRegressor
    # LGBM
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
    # Cat
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
    # Logistic Regression
from sklearn.linear_model import LogisticRegression
    # KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
    # SVM
from sklearn.svm import SVC
from sklearn.svm import SVR
    # Ridge, Lasso, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    # GridSearch
from sklearn.model_selection import GridSearchCV
    # RandomSearch
from sklearn.model_selection import RandomizedSearchCV
    # Ensemble
from itertools import combinations
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingRegressor


    # word2vec embedding (w2v)
from gensim.models import Word2Vec
embedding_model = Word2Vec(tokenized_contents, size=100, window = 2, min_count=50, workers=4, iter=100, sg=1)
# check embedding result
print(embedding_model.most_similar(positive=["������"], topn=100))


    # Save Model
import joblib


    # XGB feature Importance
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_gs.best_estimator_, ax=ax)
plt.show()

    # 의사결정 나무 시각화
import graphviz
from sklearn.tree import export_graphviz



# pip install -U gensim
# !pip install mlxtend
# !pip install vecstack
# !pip install bayesian-optimization
# !pip install catboost
# !pip install selenium
# pip install scikit_optimize
!pip3 install jsonpath --user
!pip install pyarrow --user


---------- 변수 생성 ----------

# 피처를 담을 리스트
features = []
    # 데이터 정제(handle missing value)가 이루어진 후 feature 제작해야 한다
    # feature 제작은 EDA를 바탕으로 만들어져야 한다
    # 생성된 feature들은 필요에 따라 전처리 작업(binning, scaling, 차원축소)을 거친다



# Merge & Save Features
- nan값 존재 시 정제 및 가공 과정 다시보기, 적절한 값으로 대체
- 아래 코드를 수행하면 생성한 모든 파생변수가 병합되고 CSV 화일로 만들어진다.
data = pd.DataFrame({'cust_id': tr.cust_id.unique()})
for f in features :
    data = pd.merge(data, f, how='left').fillna(0)
display(data)
data.to_csv('features.csv', index=False, encoding='cp949')
data.info()


### Feature EDA 예시────────────────────────────────────────


    # 남여별 비율, 구매양상
    # 파이차트(pie), 막대그래프(countplot)
f,ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
tr.groupby('gender')['cust_id'].nunique().plot(kind='pie', autopct='%.2f%%', explode=(0,0.02), ax=ax[0])
ax[0].set_title('데이터 내의 남/녀 비율')
sns.countplot(data=tr, x='amount_range', hue='gender', ax=ax[1])
ax[1].set_title('구매 양상: 모든 구매금액대에서 여성의 수가 더 많음')
for item in ax[1].get_xticklabels():
       item.set_rotation(90)
# plt.axis('equal')
plt.tight_layout()
plt.show()



    # 칼럼의 남여별 수와 비율의 양상
    # 막대그래프(countplot), 밀도함수(kdeplot)
figure, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(data=f, x='class', hue='gender', ax=ax[0])
sns.kdeplot(f.query('gender==0')['class'], ax=ax[1], label='여성')
sns.kdeplot(f.query('gender==1')['class'], ax=ax[1], label='남성')
plt.tight_layout()
plt.show()



    # 월별 남여별 비율
figure, ax = plt.subplots(2, 6, figsize=(15, 7))

for month in range(1,7):
    sns.kdeplot(f.query('gender==0')[month], label = '여성', ax=ax[0,month-1])
    sns.kdeplot(f.query('gender==1')[month], label = '남성', ax=ax[0,month-1])
    ax[0,month-1].set_title(f'{month}월')

temp = 0
for month in range(7,13):

    sns.kdeplot(f.query('gender==0')[month], label = '여성', ax=ax[1,temp])
    sns.kdeplot(f.query('gender==1')[month], label = '남성', ax=ax[1,temp])
    ax[1,temp].set_title(f'{month}월')
    temp += 1

plt.tight_layout()
plt.show()






### 금액, 가격 관련 features────────────────────────────────────────

    # 총 구매액, 총 구매건수, 평균구매액, 최대구매액, 최소구매액
f = tr.groupby('cust_id')['amount'].agg([
    ('총구매액', np.sum),
    ('총구매건수', np.size),
    ('평균구매액', lambda x: np.round(np.mean(x))),
    ('최대구매액', np.max),
    ('최소구매액', np.min),
]).reset_index()
features.append(f); f


    # 구매 가격대
    # 정의하기 나름임. 구매 가격대를 다음과 같이 학생때 했던 것처럼 할 수도 있지만
    # 좀 더 테크닉하고 통계적인 접근의 구매 가격대로 고객을 나눌 수 있겠음
    # 살 때마다 몇 백만원씩 사는 사람들 vs 그게 아닌 사람들
    # 부자인 사람들 사이에서도 얼마씩 어떻게 쓰는 사람들을 분 류 할 수 있을 것
    # 서민들 중에서도 얼마씩 쓰는 사람들을 분류 할 수 있을 것
    # 어떤 데이터 분석과 통계적인 접근으로 나눌수도 있겠지만 비지도학습으로도 나눌 수 있을 것이라 생각됨
tr['amount_range'] = pd.cut(tr['amount'],
       bins=[-300000000,0, 99999, 199999, 299990, 399990, 499990, 699990,999990, 1999990, 4999990, 800000000 ],
      right = False,
      labels = ['환불금액', '10만원이하', '10만원이상 20만원미만', '20만원이상 30만원미만', '30만원이상 40만원미만', '40만원이상 50만원미만', '50만원이상 70만원미만', '70만원이상 100만원미만', '100만원이상 200만원미만', '200만원이상 500만원미만', '500만원이상'])


    # 총 내점일수
visits = tr.groupby('cust_id')['tran_date'].agg(lambda x: x.nunique())
    # 내점 당 구매액 = 총 구매액 / 총 내점일수
f = (tr.groupby('cust_id')['amount'].sum() / visits).reset_index(name = '내점당구매액')
features.append(f); display(f)
    # 내점당 구매건수 = 총 구매건수 / 총 내점일수
f = (tr.groupby('cust_id')['amount'].size() / visits).reset_index(name = '내점당구매건수')
features.append(f); f


    # 최근 3개월, 6개월, 12개월 구매금액, 구매건수
    # 사용시 아래를 응용해서 만들기
for m in [3,6,12]:
    start = str(pd.to_datetime(tr.tran_date.max()) - pd.offsets.MonthBegin(m))
    f = tr.query('tran_date >= @start').groupby('cust_id')['amount'].agg([
        (f'최근{m}개월_구매금액', np.sum),
        (f'최근{m}개월_구매건수', np.size)
    ]).reset_index()
    f = pd.merge(merge_tempdf, f, how='outer').fillna(0)
    features.append(f)
    display(f)


    # 환불금액, 환불건수
f = tr[tr.amount < 0].groupby('cust_id')['amount'].agg([
    ('환불금액', lambda x: x.sum() * -1),
    ('환불건수', np.size)
]).reset_index()
features.append(f); f



    # 년 총 구매금액에 따른 등급 (실제 롯데백화점 등급 기준 적용)
    #ex 롯데백화점 MVG 조건
      # - 해당하지 않는 고객들은 silver로 설정 -> 10
      # - 400만원 이상: Vip -> 20
      # - 800만원 이상: Vip+ -> 30
      # - 1800~2000만원이상: MVG-Ace  -> 40
      # - 4000만원이상: MVG-Crown -> 50
      # - 6000만원이상: MVG-Prestige -> 60
      # - 1억이상: LENITH -> 70
temp_amountdf = tr.groupby('cust_id')['amount'].sum().reset_index()
def divide_class(x):
    if 4000000<= x < 8000000:
        return 20
    elif 8000000<= x < 18000000:
        return 30
    elif 18000000<= x <40000000:
        return 40
    elif 40000000<= x <60000000:
        return 50
    elif 60000000<= x <100000000:
        return 60
    elif x>= 100000000:
        return 70
    else:
        return 10
temp_amountdf['class'] = temp_amountdf['amount'].agg(divide_class)
f = temp_amountdf[['cust_id', 'class']]
features.append(f); f



### 시간, 날짜 관련 features────────────────────────────────────────


    # tran_date가 2007-01-19 00:00:00 형식일때
    # 년, 월, 일, 요일, 계절, (월초, 월중, 월말) 추가
tr['year'] = tr['tran_date'].agg(lambda x: pd.to_datetime(x).year) # 년 칼럼 추가
tr['month'] = tr['tran_date'].agg(lambda x: pd.to_datetime(x).month) # 월 칼럼 추가
tr['day'] = tr['tran_date'].agg(lambda x: pd.to_datetime(x).day) # 일 칼럼 추가
tr['date'] = tr['tran_date'].agg(lambda x: ('월', '화', '수', '목', '금', '토', '일')[pd.to_datetime(x).weekday()] # 요일 칼럼 추가
tr['season'] = tr['month'].agg(lambda x: 'spring' if x in [3,4,5] else 'summer' if x in [6,7,8], else 'fall' if x in [10,11,12] else 'winter') # 계절 칼럼 추가
tr['month_when'] = tr['day'].agg(lambda x: '0to10' if x in list(range(1,11)) else '11to20' if x in list(range(11,21)) else '21to31') # 월초, 월중, 월말 칼럼 추가



    # 생년월일의 데이터 타입 분리 (주소 같은 것도 분리할 때 쉽게 가능)
date_lists = stock_data['Date'].str.split('-')
    # .str.get(인덱스) 를 이용하여 데이터프레임에 연, 월, 일 정보를 따로 분리한 열을 추가한다.
stock_data['연'] = date_lists.str.get(0)
stock_data['월'] = date_lists.str.get(1)
stock_data['일'] = date_lists.str.get(2)



    # 내점일수, 구매주기
f = tr.groupby('cust_id')['tran_date'].agg([
    ('내점일수',lambda x: x.str[:10].nunique()),
    ('구매주기', lambda x: int((x.astype('datetime64').max() - x.astype('datetime64').min()).days / x.str[:10].nunique()))
]).reset_index()
features.append(f); f



    # 선호방문계절,
f = tr.groupby('cust_id')['season'].agg([
    ('perfer_season', lambda x: x.value_counts().index[0])
]).reset_index()
    # spring은 0, summer는 1, fall은 2, winter는 3으로 바꿔주기
f.perfer_season = f.perfer_season.map({'spring': 0, 'summer':1, 'fall':2, 'winter':3})
features.append(f); f

    # 계절별 구매비율
    # 계절별 구매비율은 다른 방식으로도 구현할 수 있겠음. 하기 나름
    # ex: 피벗테이블 ~ 칼럼을 계절로, value를 구매금액으로
f = tr.groupby('cust_id')['tran_date'].agg([
    ('봄-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([3,4,5]))),
    ('여름-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([6,7,8]))),
    ('가을-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([9,10,11]))),
    ('겨울-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([1,2,12])))
]).reset_index()
features.append(f); f


    # 구매일자 월(month)추출하여 새로운 칼럼 판매월 만들기
# 판매월, 구매월 구분 잘하기
tr['판매월'] = tr.구매일자//100%100
tr['구매월'] = tr['구매일자'].apply(lambda x:''.join(list(str(x))[4:6]))



    # 선호방문월
f = tr.groupby('cust_id')['month'].agg([
    ('prefer_month', lambda x: x.value_counts(1).index[0])
]).reset_index()
features.append(f);f

    # 월별 구매
# 고객별 구매금액 총합
total_amount =  tr.groupby('cust_id')['amount'].sum()

    # 고객별 월별 구매금액
temp_monthpivot = pd.pivot_table(tr, index='cust_id', columns='month', values='amount', aggfunc=sum, fill_value=0)
    # 칼럼명 변경화
oldname = temp_monthpivot.columns.tolist()
newname = [str(i) + '_month_amount' for i in oldname]
temp_monthpivot = temp_monthpivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 월별 구매금액비율
for i in range(len(temp_monthpivot.columns)):
    temp_monthpivot.iloc[:,i] = temp_monthpivot.iloc[:,i] / total_amount
f = temp_monthpivot.reset_index()
features.append(f); f

    # 월별 구매건수
temp_monthpivot = pd.pivot_table(tr, index='cust_id', columns='month', values='amount', aggfunc=np.size, fill_value=0)
    ## 칼럼명 변경화
oldname = temp_monthpivot.columns.tolist()
newname = [str(i)+'_month_buy' for i in oldname]
temp_monthpivot = temp_monthpivot.rename(columns=dict(zip(oldname, newname)))
f = temp_monthpivot.reset_index()
features.append(f); f



    # 월별 방문
# 총 방문일수
total_visits = tr.groupby('cust_id')['tran_date'].nunique()

    # 월별 방문일수
temp_monthpivot = pd.pivot_table(tr, index='cust_id', columns='month', values='tran_date', aggfunc=(lambda x: x.nunique()), fill_value=0)
    # 칼럼명 변경화
oldname = temp_monthpivot.columns.tolist()
newname = [str(i)+'_month_visits' for i in oldname]
temp_monthpivot = temp_monthpivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 월별 방문비율
for i in range(len(temp_monthpivot.columns)):
    temp_monthpivot.iloc[:,i] = temp_monthpivot.iloc[:,i] / total_visits
f = temp_monthpivot.reset_index()
features.append(f); f



    # 월초, 월중, 월말 방문
    # 월초, 월중, 월말 중 선호방문때
f = tr.groupby('cust_id')['month_when'].agg([
    ('perfer_month_when', lambda x: x.value_counts().index[0])
]).reset_index()
    # 문자를 숫자로 바꿔주기: '0to10':0 ~ '21to31': 2
f.perfer_month_when = f.perfer_month_when.map({'0to10':0, '11to20': 1, '21to31':2})
features.append(f); f

# 고객별 구매금액 총합
total_amount =  tr.groupby('cust_id')['amount'].sum()
    # 고객별 월초, 월중, 월말별 구매금액
temp_monthwhenpivot = pd.pivot_table(tr, index='cust_id', columns='month_when', values='amount', aggfunc=sum, fill_value=0)
    # 칼럼명 변경화
oldname = temp_monthwhenpivot.columns.tolist()
newname = [str(i)+'_amount' for i in oldname]
temp_monthwhenpivot = temp_monthwhenpivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 월초, 월중, 월말별 구매금액비율
for i in range(len(temp_monthwhenpivot.columns)):
    temp_monthwhenpivot.iloc[:,i] = temp_monthwhenpivot.iloc[:,i] / total_amount
f = temp_monthwhenpivot.reset_index()
features.append(f); f

    # 고객별 월초, 월중, 월말 구매건수
temp_monthwhenpivot = pd.pivot_table(tr, index='cust_id', columns='month_when', values='amount', aggfunc=np.size, fill_value=0)
# 칼럼명 변경화
oldname = temp_monthwhenpivot.columns.tolist()
newname = [str(i)+'_buy' for i in oldname]
temp_monthwhenpivot = temp_monthwhenpivot.rename(columns=dict(zip(oldname, newname)))
f = temp_monthwhenpivot.reset_index()
features.append(f); f

# 총 방문일수
total_visits = tr.groupby('cust_id')['tran_date'].nunique()
    # 고객별 월초, 월중, 월말 방문일수
temp_monthwhenpivot = pd.pivot_table(tr, index='cust_id', columns='month_when', values='tran_date', aggfunc=(lambda x: x.nunique()), fill_value=0)
# 칼럼명 변경화
oldname = temp_monthwhenpivot.columns.tolist()
newname = [str(i)+'_visits' for i in oldname]
temp_monthwhenpivot = temp_monthwhenpivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 월초, 월중, 월말 방문비율
for i in range(len(temp_monthwhenpivot.columns)):
        temp_monthwhenpivot.iloc[:,i] = temp_monthwhenpivot.iloc[:,i] / total_visits
f = temp_monthwhenpivot.reset_index()
features.append(f); f



    # 선호방문요일
f = tr.groupby('cust_id')['date'].agg([
    ('prefer_date', lambda x: x.value_counts().index[0])
]).reset_index()
    # 요일을 숫자로 바꿔주기: 월요일: 0 ~ 일요일: 6
f.prefer_date = f.prefer_date.map({'월':0,'화':1,'수':2,'목':3,'금':4,'토':5,'일':6})
features.append(f); f

# 고객별 구매금액 총합
total_amount =  tr.groupby('cust_id')['amount'].sum()
    # 고객별 요일별 구매금액
temp_datepivot = pd.pivot_table(tr, index='cust_id', columns='date', values='amount', aggfunc=sum, fill_value=0)
    # 칼럼명 변경화
oldname = temp_datepivot.columns.tolist()
newname = [str(i)+'_amount' for i in oldname]
temp_datepivot = temp_datepivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 요일별 구매금액비율
for i in range(len(temp_datepivot.columns)):
    temp_datepivot.iloc[:,i] = temp_datepivot.iloc[:,i] / total_amount
f = temp_datepivot.reset_index()
features.append(f); f

    # 고객별 요일별 구매건수
temp_datepivot = pd.pivot_table(tr, index='cust_id', columns='date', values='amount', aggfunc=np.size, fill_value=0)
    # 칼럼명 변경화
oldname = temp_datepivot.columns.tolist()
newname = [str(i)+'_buy' for i in oldname]
temp_datepivot = temp_datepivot.rename(columns=dict(zip(oldname, newname)))
f = temp_datepivot.reset_index()
features.append(f); f

# 총 방문일수
total_visits = tr.groupby('cust_id')['tran_date'].nunique()
    # 고객별 요일별 방문일수
temp_datepivot = pd.pivot_table(tr, index='cust_id', columns='date', values='tran_date', aggfunc=(lambda x: x.nunique()), fill_value=0)
# 칼럼명 변경화
oldname = temp_datepivot.columns.tolist()
newname = [str(i)+'_visits' for i in oldname]
temp_datepivot = temp_datepivot.rename(columns=dict(zip(oldname, newname)))
    # 고객별 요일별 방문비율
for i in range(len(temp_datepivot.columns)):
        temp_datepivot.iloc[:,i] = temp_datepivot.iloc[:,i] / total_visits
f = temp_datepivot.reset_index()
features.append(f); f



    # 주말 구매건수
temp_datepivot = pd.pivot_table(tr, index='cust_id', columns='date', values='amount', aggfunc=np.size, fill_value=0)
f = temp_datepivot[['토','일']].sum(axis=1).reset_index().rename(columns={0:'weekend_buy'})
features.append(f); f

    # 주말방문비율
f = tr.groupby('cust_id')['tran_date'].agg([
    ('주말방문비율', lambda x: np.mean(pd.to_datetime(x).dt.dayofweek>4))
]).reset_index()
features.append(f); f




### 장소, 위치 관련 features────────────────────────────────────────


    # 주구매지점
f = tr.groupby('cust_id')['store_nm'].agg([
    ('주구매지점', lambda x: x.value_counts().index[0])
]).reset_index()
features.append(f); f


    # 고객별 방문 지점의 수
f = tr.groupby('cust_id')['store_nm'].nunique().reset_index().rename(columns={'store_nm':'stores_count'})
features.append(f); f


# top-12에 해당하는 지점들
top12store = tr.store_nm.value_counts()[:12].index.tolist()

    # 고객별 점포별 구매건수
temp_storepivot = pd.pivot_table(tr, index='cust_id', columns='store_nm', values='amount', aggfunc=np.size, fill_value=0)
    # 칼럼명 변경화
oldname = temp_storepivot.columns.tolist()
newname = [str(i)+'_buy' for i in oldname]
temp_storepivot = temp_storepivot.rename(columns=dict(zip(oldname, newname)))
f = temp_storepivot.reset_index()
features.append(f); f

    # 고객별 top-12 점포별 방문비율
# 고객별 전체방문일수
total_visits = tr.groupby('cust_id')['tran_date'].nunique()
    # 고객별 top-12 점포별 방문일수
temp_storepivot = pd.pivot_table(tr.query('store_nm in @top12store'), index='cust_id', columns='store_nm', values='tran_date', aggfunc=(lambda x: x.nunique()), fill_value=0)
    # 칼럼명 변경화
oldname = temp_storepivot.columns.tolist()
newname = [str(i)+'_visits' for i in oldname]
temp_storepivot = temp_storepivot.rename(columns=dict(zip(oldname, newname)))

    # 고객별 top-12 점포별 방문비율
for i in range(len(temp_storepivot.columns)):
    temp_storepivot.iloc[:,i] = temp_storepivot.iloc[:,i] / total_visits
f = temp_storepivot.reset_index()
features.append(f); f




### 제품 관련 features────────────────────────────────────────


    # 주구매상품소분류
 - 총 3471개 물품 중 3500명이 가장 많이 사는 물품들은 850개 정도로 겹치는 물품들이 존재하였다.
f = tr.groupby('cust_id')['goods_id'].agg([
    ('주구매상품', lambda x: x.value_counts().index[0])
]).reset_index()
features.append(f); f

    # 주구매상품대분류
f = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([
    ('주구매상품대분류', lambda x: x.value_counts().index[0])
]).reset_index()
features.append(f); f



    # 상품별 구매순서
    # 가장 먼저 방문하는 상품대분류명군
temp_lst = []
for i in range(3500):
    temp_lst.append(tr.query(f'cust_id == {i}').drop_duplicates('tran_date')['gds_grp_mclas_nm'].value_counts().index[0])
f = pd.DataFrame({'cust_id': np.arange(3500)})
f['first_visit_place'] = temp_lst
features.append(f); f



    # 구매상품종류1(goods_id), 구매상품종류2(gds_grp_nm), 구매상품종류3(gds_grp_mclas_nm)
f = tr.groupby('cust_id').agg({
    'goods_id': [('구매상품종류1', lambda x: x.nunique())],
    'gds_grp_nm': [('구매상품종류2', lambda x: x.nunique())],
    'gds_grp_mclas_nm': [('구매상품종류3', lambda x: x.nunique())]
})
f.columns = f.columns.droplevel()  # 동일한 코드: f.columns = [j for _, j in f.columns]
f=f.reset_index()
features.append(f); f



    # 상품 분류별  top-x 구매금액비율 , 구매건수
    # - 상품분류별 중 90퍼센트 이상을 차지하는 상품들의 구매금액비율과 구매건수를 구한다
        # - goods_id(상품소분류별) top550 구매금액비율, 구매건수
        # - gds_grp_nm(상품중분류별) top75 구매금액비율, 구매건수
        # - gds_grp_mclas_nm(상품대분류별) top25 구매금액비율, 구매건수

    # 많이 구매한 상품 소분류별 best-x 구매금액비율, 구매건수(3471 중 x)
# 전체 구매금액
total = tr.amount.sum()
# 전체 구매금액 중 goods_id top550이 전체의 90퍼센트 정도를 차지한다.
top550 = tr.groupby('goods_id')['amount'].sum().reset_index().sort_values('amount', ascending=False)[:550]
top550sum = top550['amount'].sum()
top550goods = top550['goods_id'].tolist()
print(top550sum / total)

    # goods_id top550 구매금액비율
# 고객별 전체 구매금액
totalsum = tr.groupby('cust_id')['amount'].sum()
# 고객별 goods_id top550 구매
temp_goodsdf = tr.query('goods_id in @top550goods')
temp_goodspivot = pd.pivot_table(temp_goodsdf, index='cust_id', columns='goods_id', values='amount', aggfunc=sum, fill_value=0)
for i in range(len(temp_goodspivot.columns)):
    temp_goodspivot.iloc[:,i] = temp_goodspivot.iloc[:,i] / totalsum
f = temp_goodspivot.reset_index()

merge_tempdf = pd.DataFrame({'cust_id': np.arange(3500)})
f = pd.merge(merge_tempdf, f, how='outer').fillna(0)
features.append(f); f

    # goods_id top550 구매금액건수
temp_goodspivot = pd.pivot_table(temp_goodsdf, index='cust_id', columns='goods_id', values='amount', aggfunc=np.size, fill_value=0)
# 변수명이 겹치는 것을 막기 위해서 변수명을 다르게 바꿔주는 과정
oldname = temp_goodspivot.columns.tolist()
newname = [(str(i)+'_buy') for i in oldname]
temp_goodspivot = temp_goodspivot.rename(columns = dict(zip(oldname, newname)))
f = temp_goodspivot.reset_index() # 3258행
# merge
merge_tempdf = pd.DataFrame({'cust_id': np.arange(3500)})
f = pd.merge(merge_tempdf, f, how='outer').fillna(0).astype(int)
features.append(f); f



    # 남성경향물품, 여성경향물품






### 기타 features────────────────────────────────────────

















────────────────────────────────────────────────────────────
