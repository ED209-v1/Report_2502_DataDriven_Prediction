#!/usr/bin/env python3                  # 유닉스 계열에서 이 파일을 직접 실행 가능하게 하는 셔뱅(shebang)
# -*- coding: utf-8 -*-                 # 소스 파일 인코딩: UTF-8 (한글 주석/문자 포함 가능)

"""
K-NN 회귀 (시계열용 래깅 특징) + 시각화
- 입력: ./data.csv (컬럼: NUM, Date, Sensor, Quality)
- 출력: ./knn_output/ 폴더에 그래프 3장 + 평가표 CSV
"""

# ===== 표준 라이브러리 임포트 =====
import os                                # 경로/폴더 생성 등 파일시스템 작업
import sys                               # 운영체제/파이썬 인터프리터 관련 유틸
import warnings                          # 경고 제어 (필요시 무시)
from datetime import datetime            # 현재 날짜/시간 표기용

# ===== 서드파티 라이브러리 임포트 =====
import numpy as np                       # 수치 연산
import pandas as pd                      # 데이터 프레임/시계열 처리
import matplotlib.pyplot as plt          # 시각화
from matplotlib import font_manager      # 폰트 등록(WSL/리눅스 한글 폰트)

# scikit-learn: 전처리-모형-튜닝 파이프라인
from sklearn.pipeline import Pipeline                     # 전처리/모형 순차 실행
from sklearn.preprocessing import StandardScaler          # 표준화(평균0, 분산1)
from sklearn.neighbors import KNeighborsRegressor         # K-NN 회귀 모델
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # 하이퍼파라미터 탐색/시계열 CV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 평가지표

# =========================================================
# 0) 한글 폰트(가능 시) & 경고 정리
# =========================================================
def setup_korean_font():
    """플랫폼에 맞게 한글 폰트를 설정한다. (Windows: 맑은고딕, Linux/WSL: 나눔/노토)
       폰트가 없으면 조용히 패스하고 기본 폰트 사용(영문 표기)"""
    try:
        candidates = []                                      # 후보 폰트 이름 리스트
        if sys.platform.startswith("win"):                   # 윈도우인 경우
            candidates = ["Malgun Gothic", "맑은 고딕"]        # 맑은고딕 우선
        else:                                                # 리눅스/WSL인 경우
            # 시스템 폰트 경로 중 존재하는 파일을 폰트 매니저에 수동 등록 시도
            for p in [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]:
                if os.path.exists(p):                        # 경로가 실제 존재하면
                    try:
                        font_manager.fontManager.addfont(p)  # 해당 폰트를 런타임 등록
                    except Exception:
                        pass                                  # 실패해도 무시(필수는 아님)
            candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK"]  # 등록 후 이름으로 탐색
        # 현재 환경에서 사용 가능한 폰트 목록
        available = {f.name for f in font_manager.fontManager.ttflist}
        # 후보 중 첫 번째로 사용 가능한 폰트를 matplotlib 전역 폰트로 설정
        for name in candidates:
            if name in available:
                plt.rc("font", family=name)
                break
        # 마이너스 부호가 네모로 깨지는 현상 방지
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass                                                  # 폰트 설정이 실패해도 기능엔 영향 없음

# 특정 FutureWarning(버전 예고성 경고)을 전역에서 숨김
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1) 데이터 로드 & 전처리
# =========================================================
DATA_PATH = "./data.csv"               # 입력 데이터 경로(요구사항: 소스에 고정)
OUT_DIR   = "./knn_output"             # 결과물을 저장할 폴더

# 특징(피처) 생성 하이퍼파라미터(필요시 조정)
LAGS  = 5                              # 래그(지연) 특성 개수: lag_1..lag_5
ROLLS = [3, 5, 7]                      # 이동 윈도우 크기(평균/표준편차 생성)
TEST_RATIO = 0.2                       # 데이터의 마지막 20%를 테스트 세트로 사용

def load_series(csv_path=DATA_PATH, time_col="Date", value_col="Sensor"):
    """CSV에서 시계열을 읽어 단일 Series로 반환.
       - time_col: 시각 문자열(UTC 포함) → datetime 변환 후 인덱스
       - 중복 타임스탬프는 평균 집계
       - value_col: 예측 대상 수치 컬럼"""
    df = pd.read_csv(csv_path)                                     # CSV 파일 로드

    # 필수 컬럼 여부 확인(없으면 즉시 예외 발생)
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV에 '{time_col}', '{value_col}' 컬럼이 필요합니다. 실제 컬럼: {list(df.columns)}")

    # 시간 문자열을 datetime으로 변환(UTC 포함) + 변환 실패는 NaT 처리
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])                               # 시간값 NaT는 제거
    # statsmodels/일부 도구 호환을 위해 timezone 정보 제거(tz-aware → tz-naive)
    df[time_col] = df[time_col].dt.tz_localize(None)
    df = df.sort_values(time_col)                                   # 시간순 정렬

    # 동일 타임스탬프가 여러 행이라면 수치형 컬럼 평균으로 집계
    df = df.groupby(time_col, as_index=False).mean(numeric_only=True).set_index(time_col)

    # 대상 컬럼을 숫자로 강제 변환(문자 등 섞여 있으면 NaN)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])                              # NaN 행 제거

    # 단일 Series로 반환(이름은 value_col)
    series = df[value_col].astype(float).copy()
    series.name = value_col
    return series

# =========================================================
# 2) 시계열 → 지도학습용 특징 만들기
#    - lag_1..lag_LAGS (이전 시점 값)
#    - roll_mean_k, roll_std_k (직전까지의 이동 통계)
# =========================================================
def make_supervised_features(series: pd.Series, lags=LAGS, rolls=ROLLS):
    """시계열 단변량 데이터를 지도학습용 X(특징), y(목표)로 변환한다."""
    df = pd.DataFrame({"y": series})                       # 예측 목표 y를 먼저 프레임으로
    # 래깅(이전 시점) 피처 생성: 예) lag_1은 바로 직전 값
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = series.shift(i)                   # i만큼 뒤로 민 값(현재 관측 시점에서 과거)
    # 이동 평균/표준편차: '현재 관측 전에'만 보도록 한 칸 시프트
    for k in rolls:
        df[f"roll_mean_{k}"] = series.rolling(k, min_periods=1).mean().shift(1)  # k창 평균, 미래 정보 누수 방지 위해 .shift(1)
        df[f"roll_std_{k}"]  = series.rolling(k, min_periods=1).std(ddof=0).shift(1)  # k창 표준편차(모표준편차)

    df = df.dropna()                                      # 래그/롤링으로 생긴 선두 NaN 제거
    X = df.drop(columns=["y"])                            # 특징 행렬
    y = df["y"]                                           # 타깃 벡터
    return X, y

# =========================================================
# 3) 시계열 분할 (시간 순서 유지)
#    - 무작위 분할은 시계열에서 데이터 누수를 유발할 수 있음 → 마지막 일부를 Test로
# =========================================================
def train_test_split_time(X, y, test_ratio=TEST_RATIO):
    """시간 순서를 유지한 채 Train/Test로 분할(마지막 일부를 Test)."""
    n = len(X)                                            # 총 샘플 수
    n_test = max(1, int(n * test_ratio))                  # 테스트 샘플 수(최소 1 보장)
    n_train = n - n_test                                  # 훈련 샘플 수
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]  # 앞부분=Train, 뒷부분=Test
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test

# =========================================================
# 4) KNN 파이프라인 + 그리드서치(TimeSeriesSplit)
#    - 스케일 표준화 + KNN을 하나의 파이프라인으로 묶고
#      TimeSeriesSplit을 사용해 하이퍼파라미터 탐색
# =========================================================
def fit_knn(X_train, y_train):
    """표준화-모형 파이프라인 구성 후, 시계열 교차검증으로 하이퍼파라미터 탐색."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),                     # 입력 특징 스케일 표준화(거리기반 KNN에서 필수적)
        ("knn", KNeighborsRegressor())                    # 기본 파라미터의 KNN 회귀
    ])

    # 탐색할 하이퍼파라미터 그리드 정의
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],            # k(최근접 이웃 수)
        "knn__weights": ["uniform", "distance"],         # 거리 가중: 균등 or 거리 반비례
        "knn__p": [1, 2]                                 # 거리 지수 p=1(맨해튼), p=2(유클리드)
    }

    # 시계열 데이터를 위한 분할자: 과거→미래 순서를 지키며 여러 폴드 생성
    tscv = TimeSeriesSplit(n_splits=5)

    # 그리드서치: MAE를 음수로 바꾼 값이 점수(scoring 규칙) → 절대값이 작을수록 좋음
    gscv = GridSearchCV(
        pipe, param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1                                         # 가용 코어 병렬 사용
    )
    gscv.fit(X_train, y_train)                            # 훈련 데이터로 최적 하이퍼파라미터 탐색/학습
    return gscv                                           # gscv.best_estimator_, gscv.best_params_ 사용 가능

# =========================================================
# 5) 평가 지표
# =========================================================
def metrics(y_true, y_pred):
    """MAE, RMSE, R2를 계산해 딕셔너리로 반환."""
    mae  = mean_absolute_error(y_true, y_pred)            # 평균 절대 오차(낮을수록 좋음)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())       # 제곱오차 평균의 제곱근(이상치 민감)
    r2   = r2_score(y_true, y_pred)                       # 결정계수(설명력, 1에 가까울수록 좋음)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# =========================================================
# 6) 시각화 함수
#    - (A) 시간축: 실제 vs 예측 (Train/Test 구분)
#    - (B) 산점도: y_true vs y_pred (Test)
#    - (C) 잔차 시계열: residual = y - ŷ (Test)
# =========================================================
def plot_time_series(y_train, y_train_pred, y_test, y_test_pred, out_path):
    """시간축 상에서 Train/Test 실제값과 예측값을 함께 그린다."""
    plt.figure(figsize=(12, 5))                           # 그림 크기 지정
    plt.plot(y_train.index, y_train.values, label="Train 실제", alpha=0.8)          # 훈련 실제
    plt.plot(y_train.index, y_train_pred, label="Train 예측", linestyle="--")       # 훈련 예측
    plt.plot(y_test.index,  y_test.values,  label="Test 실제",  alpha=0.9)          # 테스트 실제
    plt.plot(y_test.index,  y_test_pred,  label="Test 예측",  linestyle="--")       # 테스트 예측
    plt.title("K-NN 회귀: 실제 vs 예측 (시간축)")                                      # 제목
    plt.legend()                                           # 범례
    plt.tight_layout()                                     # 레이아웃 자동 조정
    plt.savefig(out_path, dpi=150)                         # 파일로 저장(해상도 150dpi)
    plt.close()                                            # 메모리 해제

def plot_scatter(y_true, y_pred, out_path):
    """테스트 구간에서 실제값과 예측값의 산점도를 그리고 y=x 기준선 표시."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)          # 산점도
    lims = [
        min(y_true.min(), y_pred.min()),                  # 축 하한: 실제/예측 최소 중 작은 값
        max(y_true.max(), y_pred.max())                   # 축 상한: 실제/예측 최대 중 큰 값
    ]
    plt.plot(lims, lims, linestyle="--")                  # 이상적인 예측선(y=x)
    plt.xlim(lims); plt.ylim(lims)                        # x/y축 범위를 동일하게
    plt.xlabel("실제 값 (y)")                               # x축 레이블
    plt.ylabel("예측 값 (ŷ)")                               # y축 레이블
    plt.title("K-NN 회귀: 실제 vs 예측 산점도 (Test)")         # 제목
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, out_path):
    """테스트 구간에서 잔차(residual = y - ŷ)의 시간적 추이를 그린다."""
    res = y_true - y_pred                                 # 잔차 계산(벡터 연산)
    plt.figure(figsize=(12, 4))
    plt.plot(y_true.index, res, marker="o", linestyle="-", linewidth=1)  # 잔차 시계열
    plt.axhline(0, color="gray", linestyle="--")          # 기준선(0) 추가
    plt.title("K-NN 회귀: 잔차(residual) 추이 (Test)")
    plt.ylabel("잔차 (y - ŷ)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =========================================================
# 7) 메인 루틴
# =========================================================
def main():
    # 출력 폴더 생성(이미 있으면 무시)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 한글 폰트 설정(가능 시)
    setup_korean_font()

    # 1) 데이터 로드(./data.csv → Series: index=Datetime, values=Sensor(float))
    series = load_series()

    # 2) 지도학습용 특징/타깃 생성(lag/rolling 기반 피처)
    X, y = make_supervised_features(series, lags=LAGS, rolls=ROLLS)

    # 3) 시간 순서 분할(누수 방지: 마지막 20%를 테스트로)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_ratio=TEST_RATIO)

    # 4) 모델 학습(그리드서치+TimeSeriesSplit)
    model = fit_knn(X_train, y_train)

    # 5) 훈련/테스트 구간 예측값 생성(인덱스를 y와 맞춰 정렬)
    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index, name="train_pred")
    y_test_pred  = pd.Series(model.predict(X_test),  index=y_test.index,  name="test_pred")

    # 6) 평가 지표 계산
    tr_metrics = metrics(y_train, y_train_pred)
    te_metrics = metrics(y_test, y_test_pred)

    # 7) 콘솔 출력(가독성을 위해 소수점 반올림)
    print("✅ 최적 하이퍼파라미터:", model.best_params_)                         # 예: {'knn__n_neighbors': 7, 'knn__p': 2, 'knn__weights': 'distance'}
    print("✅ Train  점수:", {k: round(v, 6) for k, v in tr_metrics.items()})    # 훈련 세트 성능(MAE/RMSE/R2)
    print("✅ Test   점수:", {k: round(v, 6) for k, v in te_metrics.items()})    # 테스트 세트 성능

    # 8) 평가표를 CSV로 저장(재현/보고용)
    eval_df = pd.DataFrame([
        {"구분": "Train", **tr_metrics},
        {"구분": "Test",  **te_metrics},
    ])
    eval_csv = os.path.join(OUT_DIR, "knn_metrics.csv")
    eval_df.to_csv(eval_csv, index=False)

    # 9) 시각화 저장(세 종류)
    plot_time_series(
        y_train, y_train_pred, y_test, y_test_pred,
        out_path=os.path.join(OUT_DIR, "A_timeseries_true_vs_pred.png")
    )
    plot_scatter(
        y_test, y_test_pred,
        out_path=os.path.join(OUT_DIR, "B_scatter_true_vs_pred_test.png")
    )
    plot_residuals(
        y_test, y_test_pred,
        out_path=os.path.join(OUT_DIR, "C_residuals_test.png")
    )

    # 10) 예측 결과(테스트 구간)의 상세 표 저장(후속 분석에 유용)
    result_df = pd.DataFrame({
        "y_true": y_test,                                  # 실제 값
        "y_pred": y_test_pred,                             # 예측 값
        "residual": y_test - y_test_pred                   # 잔차
    })
    result_df.to_csv(os.path.join(OUT_DIR, "knn_predictions_test.csv"))

    # 11) 요약 경로 출력
    print(f"\n📁 결과 저장 폴더: {OUT_DIR}")
    print(f"- 평가표: {eval_csv}")
    print("- 그래프: A_timeseries_*, B_scatter_*, C_residuals_*")
    print("- 예측표: knn_predictions_test.csv")

# 파이썬 스크립트로 직접 실행될 때만 main() 호출(모듈 임포트 시 실행 방지)
if __name__ == "__main__":
    main()
