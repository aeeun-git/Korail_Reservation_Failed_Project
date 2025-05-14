import pandas as pd
import numpy as np
import holidays

# 1) 원본 데이터 로드
df = pd.read_csv(
    'filtered_reservation_failed_rev2.csv',
    dtype={'출발시각': str}
)

# 2) OD 칼럼 생성 & 필요한 OD만 필터
df['OD'] = df['출발역'] + '_' + df['도착역']
selected = {
    '서울_동대구','서울_대전','서울_오송',
    '동대구_서울','대전_서울','오송_서울'
}
df = df[df['OD'].isin(selected)].copy()

# 3) 출발시각을 6자리 문자열로
df['출발시각6'] = df['출발시각'].str.zfill(6)

# 4) datetime 파싱
df['출발일자_dt'] = pd.to_datetime(df['출발일자'], format='%Y%m%d')
df['시']   = df['출발시각6'].str[:2].astype(int)
df['분'] = df['출발시각6'].str[2:4].astype(int)
df['초'] = df['출발시각6'].str[4:6].astype(int)
df['출발일시'] = (
    df['출발일자_dt']
    + pd.to_timedelta(df['시'],   unit='h')
    + pd.to_timedelta(df['분'], unit='m')
    + pd.to_timedelta(df['초'], unit='s')
)

# ── 여기서 2024-11-27까지 데이터만 남기기 ──────────────────────
cutoff = pd.to_datetime('2024-11-27')
df = df[df['출발일자_dt'] <= cutoff].copy()

# 5) 요일 사이클릭 인코딩
df['요일'] = df['출발일시'].dt.weekday  # 0=월 … 6=일
df['요일_사인'] = np.sin(2*np.pi * df['요일']/7)
df['요일_코사인'] = np.cos(2*np.pi * df['요일']/7)

# 6) 주말 여부
df['주말여부'] = df['요일'].isin([5,6]).astype(int)

# 7) 공휴일 / 전날 공휴일 / 다음날 공휴일
kr  = holidays.KR()
df['공휴일여부']   = df['출발일시'].dt.date.map(lambda d: int(d in kr))
df['공휴일전여부'] = df['출발일시'].dt.date.map(lambda d: int((d + pd.Timedelta(days=1)) in kr))
df['공휴일후여부'] = df['출발일시'].dt.date.map(lambda d: int((d - pd.Timedelta(days=1)) in kr))

# 8) 날씨 데이터 로드 & 전처리
weather = pd.read_csv(
    'OBS_ASOS_TIM_20250507162026.csv',
    parse_dates=['일시'],
    encoding='euc_kr'
).rename(columns={
    '지점명': '기상지점',
    '일시':   '출발일시',
    '기온(°C)':   '기온',
    '강수량(mm)': '강수량',
    '적설(cm)':   '적설'
})
weather_agg = (
    weather
    .groupby(['기상지점','출발일시'])[['기온','강수량','적설']]
    .mean()
    .reset_index()
)

# 9) 출발역 → 가까운 기상지점 매핑
station_to_weather = {
    '서울':'서울', '동대구':'대구', '대전':'대전', '오송':'청주'
}
df['기상지점'] = df['출발역'].map(station_to_weather)

# 10) 날씨 merge
df = df.merge(
    weather_agg,
    on=['기상지점','출발일시'],
    how='left'
)

# 11) 최종 칼럼 선택 및 한글 컬럼명 설정
out = df[[
    '출발역','도착역','출발일자','출발시각6','예약실패건수',
    '요일_사인','요일_코사인','주말여부',
    '공휴일여부','공휴일전여부','공휴일후여부',
    '기온','강수량','적설'
]].copy()
out.columns = [
    '출발역','도착역','출발일자','출발시각',
    '예약실패건수','요일_사인','요일_코사인','주말여부',
    '공휴일여부','공휴일전여부','공휴일후여부',
    '기온','강수량','적설'
]

out.to_csv('preprocessed_6OD.csv', index=False, encoding='utf-8-sig')
print("preprocessed_6OD.csv 생성 완료")
