import os, shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Attention, Dense, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 하이퍼파라미터 탐색 디렉터리 초기화
BASE_DIR = r"C:\temp\korail_kt"
if os.path.isdir(BASE_DIR):
    shutil.rmtree(BASE_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)

# 데이터 로드, 전처리 
df = pd.read_csv('preprocessed_6OD.csv', dtype={'출발시각': str})
cols_to_fill = [
    '요일_사인','요일_코사인',
    '주말여부','공휴일여부','공휴일전여부','공휴일후여부',
    '기온','강수량','적설'
]
df[cols_to_fill] = df[cols_to_fill].fillna(0)
df['OD'] = df['출발역'] + '_' + df['도착역']
df['출발시각'] = df['출발시각'].str.zfill(6)
df['depart_dt'] = pd.to_datetime(
    df['출발일자'].astype(str) + df['출발시각'],
    format='%Y%m%d%H%M%S'
)

# OD 지정
single_od = '서울_대전'
df = df[df['OD'] == single_od].sort_values('depart_dt')

ts_fail = df.set_index('depart_dt')['예약실패건수']
feat_ts = df.drop_duplicates('depart_dt').set_index('depart_dt')[cols_to_fill]

# LAG 인코딩
hour_steps = 6
day_steps  = 24 * hour_steps
week_steps = 7 * day_steps
lag_df = pd.DataFrame(index=ts_fail.index)
for d in range(1,8):
    lag_df[f'lag{d}d'] = ts_fail.shift(d * day_steps)
for w in range(1,4):
    lag_df[f'lag{w}w'] = ts_fail.shift(w * week_steps)
lag_df['avg3d'] = lag_df[[f'lag{d}d' for d in range(1,4)]].mean(axis=1)
lag_df['avg7d'] = lag_df[[f'lag{d}d' for d in range(1,8)]].mean(axis=1)
lag_df.fillna(0, inplace=True)

# 피처 정규화
scaler_feat = MinMaxScaler()
feat_scaled = pd.DataFrame(
    scaler_feat.fit_transform(feat_ts),
    index=feat_ts.index,
    columns=feat_ts.columns
).fillna(0)

# 시퀀스 생성
SEQ_LEN = 72
def make_sequences(series, feat_df, lag_df):
    X, y = [], []
    for i in range(len(series) - SEQ_LEN):
        seq = series[i:i+SEQ_LEN].reshape(-1,1)
        fts = feat_df.iloc[i:i+SEQ_LEN].values
        lgs = lag_df.iloc[i:i+SEQ_LEN].values
        X.append(np.hstack([seq, fts, lgs]))
        y.append(series[i+SEQ_LEN])
    return np.array(X), np.array(y)

# 모델 빌더
def build_model(hp):
    units      = hp.Int('units',      min_value=16, max_value=128, step=16)
    lr         = hp.Float('lr',       1e-4, 1e-2, sampling='log')
    batch_size = hp.Choice('batch_size', [16, 32, 64, 128, 256])
    inp        = Input((SEQ_LEN, feat_dim))
    lstm_out, st_h, _ = LSTM(units, return_sequences=True, return_state=True)(inp)
    q     = Reshape((1, units))(st_h)
    attn  = Attention()([q, lstm_out])
    out   = Dense(1)(Flatten()(attn))

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error', metrics=['mae']
    )
    model._batch_size = batch_size
    return model

# 학습, 예측
raw      = ts_fail.values
raw_log  = np.log1p(raw).reshape(-1,1)
scaler_raw = MinMaxScaler().fit(raw_log)
scaled = scaler_raw.transform(raw_log).flatten()

# 시퀀스 만들기
X, y = make_sequences(scaled, feat_scaled, lag_df)
times   = ts_fail.index[SEQ_LEN:]
split_dt = pd.to_datetime('2024-12-14')
mask    = times >= split_dt
X_train, X_test = X[~mask], X[mask]
y_train, y_test = y[~mask], y[mask]

# NaN 제거
valid      = (~np.isnan(X_train).any(axis=(1,2))) & (~np.isnan(y_train))
X_train, y_train = X_train[valid], y_train[valid]
X_train = X_train.astype('float32'); X_test  = X_test.astype('float32')
y_train = y_train.astype('float32'); y_test  = y_test.astype('float32')
feat_dim = X_train.shape[-1]

# 하이퍼파라미터 탐색
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory=BASE_DIR,
    project_name='tune_single',
    overwrite=True
)
tuner.search(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=0
)

# 최적 하이퍼파라미터 추출
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model   = build_model(best_hp)
bs      = best_hp.get('batch_size') 

# 최적 모델 재학습
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=bs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1
)

# 평가, 예측
loss, mae      = model.evaluate(X_test, y_test, verbose=0)
pred_scaled    = model.predict(X_test, verbose=0).flatten()
pred_scaled    = np.clip(pred_scaled, 0.0, 1.0)

# 역변환
y_test_log     = scaler_raw.inverse_transform(y_test.reshape(-1,1)).flatten()
y_test_orig    = np.expm1(y_test_log)
pred_log       = scaler_raw.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
pred_orig      = np.expm1(pred_log).clip(0.0)

print(f"MAE = {mae:.4f}, batch_size = {bs}")

# CSV 저장
out = pd.DataFrame({
    'depart_dt': times[mask],
    'true':      y_test_orig,
    'pred':      pred_orig
})
out['OD'] = single_od
out.to_csv('predictions_대전2.csv', index=False)
print("완료: predictions_대전2.csv")
