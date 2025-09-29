import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from tqdm import tqdm
from unidecode import unidecode

# 1. 데이터 로딩
df = pd.read_csv("final.csv", dtype=str)
df["fail_count"] = df["예약실패건수"].astype(int)
df["datetime"] = pd.to_datetime(df["출발일자"] + df["출발시각"], format="%Y%m%d%H%M%S")

# 2. 1000건 이상 OD쌍 추출
od_fail = df.groupby(["출발역", "도착역"])["fail_count"].sum().reset_index()
od_pairs = od_fail[od_fail["fail_count"] >= 1000][["출발역", "도착역"]].values.tolist()
print(f"1000건 이상 OD쌍 개수: {len(od_pairs)}")   # 163 예상

# 3. Prophet 분석 함수
def prophet_od_analysis(df, ori, dst, save_dir="plots/prophet"):
    od_df = df[(df["출발역"] == ori) & (df["도착역"] == dst)][["datetime", "fail_count"]].copy()
    od_df.rename(columns={"datetime": "ds", "fail_count": "y"}, inplace=True)
    od_df = od_df.sort_values("ds")

    if len(od_df) < 100:  # 시계열 너무 짧으면 스킵
        print(f"Skip: {ori}→{dst} (too short)")
        return

    # Prophet 모델 생성 및 학습
    model = Prophet(seasonality_mode='multiplicative')
    model.fit(od_df)

    # 미래 1일치 예측 (15분 간격, 96개)
    future = model.make_future_dataframe(periods=96, freq='15min')
    forecast = model.predict(future)

    # 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일명용 영문 변환
    dep_eng, arr_eng = unidecode(ori), unidecode(dst)

    # 전체 예측 시계열
    fig1 = model.plot(forecast)
    plt.title(f"Prophet Forecast: {dep_eng}→{arr_eng}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dep_eng}_{arr_eng}_forecast.png")
    plt.close(fig1)

    # 트렌드, 시즌성 등 컴포넌트 분해 그래프
    fig2 = model.plot_components(forecast)
    plt.title(f"Prophet Components: {dep_eng}→{arr_eng}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dep_eng}_{arr_eng}_components.png")
    plt.close(fig2)

    print(f"Saved Prophet result: {ori}→{dst}")

# 4. 전체 OD쌍 반복 분석 (진행바)
for ori, dst in tqdm(od_pairs):
    prophet_od_analysis(df, ori, dst)

print("분석 및 그래프 저장 완료! plots/prophet 폴더를 확인하세요.")
