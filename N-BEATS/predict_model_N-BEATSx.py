import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from unidecode import unidecode
import duckdb
import math


# 하이퍼파라미터 & lag 설정
DATA_PATH      = "final.csv"
SEQ_LEN        = 96      # 과거 12시간(15분 간격 96스텝)
PRED_LEN       = 4       # 1시간(4스텝) 예측
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 0.0005
WEIGHT_DECAY   = 0.0001
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_START = pd.to_datetime("20241120", format="%Y%m%d")
OUTPUT_DIR = "models_nbeatsx_interpretable"

# 데이터 로딩 및 전처리
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found."
df = pd.read_csv(DATA_PATH, dtype=str).rename(columns={
    "출발역":"ori","도착역":"dst",
    "출발일자":"date","출발시각":"time",
    "예약실패건수":"fail_count",
    "요일_사인":"dow_sin","요일_코사인":"dow_cos",
    "주말여부":"is_weekend","공휴일여부":"is_holiday",
    "공휴일전여부":"before_holiday","공휴일후여부":"after_holiday",
    "기온":"temperature","강수량":"precipitation","적설":"snowfall",
})
df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
df = df.sort_values("datetime")
df["fail_count"] = df["fail_count"].astype(int)
for c in ["dow_sin","dow_cos","is_weekend","is_holiday","before_holiday","after_holiday"]:
    df[c] = df[c].astype(float)
for c in ["temperature","precipitation","snowfall"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").ffill().fillna(0.0)
df.drop(columns=["date","time"], inplace=True)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, X_exog, y):
        self.X = X
        self.X_exog = X_exog
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.X_exog[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32)
        )

# <<< 1단계: 모델 클래스를 N-BEATSx (해석 가능 버전)으로 수정 >>>
class NBeatsInterpretableXBlock(nn.Module):
    """
    외부 변수를 받는 N-BEATS의 기본 블록.
    """
    def __init__(self, in_sz, exog_sz, hid_sz, n_layers, theta_sz):
        super().__init__()
        # 입력으로 x와 x_exog를 함께 받으므로 입력 크기가 in_sz + exog_sz
        layers = [nn.Linear(in_sz + exog_sz, hid_sz), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hid_sz, hid_sz), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.theta_b_fc = nn.Linear(hid_sz, theta_sz, bias=False)
        self.theta_f_fc = nn.Linear(hid_sz, theta_sz, bias=False)

    def forward(self, x, x_exog):
        combined_input = torch.cat([x, x_exog], dim=1)
        hidden_state = self.layers(combined_input)
        return self.theta_b_fc(hidden_state), self.theta_f_fc(hidden_state)

class TrendBlockX(nn.Module):
    """
    외부 변수를 사용하는 트렌드 블록.
    """
    def __init__(self, in_sz, out_sz, exog_sz, hid_sz, n_layers, p_degree):
        super().__init__()
        self.block = NBeatsInterpretableXBlock(in_sz, exog_sz, hid_sz, n_layers, p_degree + 1)
        t = torch.arange(in_sz, dtype=torch.float32) / in_sz
        self.backcast_basis = torch.stack([t ** i for i in range(p_degree + 1)], dim=1)
        t = torch.arange(out_sz, dtype=torch.float32) / out_sz
        self.forecast_basis = torch.stack([t ** i for i in range(p_degree + 1)], dim=1)

    def forward(self, x, x_exog):
        theta_b, theta_f = self.block(x, x_exog)
        backcast = torch.matmul(self.backcast_basis.to(x.device), theta_b.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis.to(x.device), theta_f.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast

class SeasonalityBlockX(nn.Module):
    """
    외부 변수를 사용하는 계절성 블록.
    """
    def __init__(self, in_sz, out_sz, exog_sz, hid_sz, n_layers, num_harmonics):
        super().__init__()
        self.block = NBeatsInterpretableXBlock(in_sz, exog_sz, hid_sz, n_layers, 2 * num_harmonics)
        t = torch.arange(in_sz, dtype=torch.float32) / in_sz
        backcast_harmonics = [torch.cos(2 * math.pi * i * t) for i in range(1, num_harmonics + 1)] + \
                             [torch.sin(2 * math.pi * i * t) for i in range(1, num_harmonics + 1)]
        self.backcast_basis = torch.stack(backcast_harmonics, dim=1)
        t = torch.arange(out_sz, dtype=torch.float32) / out_sz
        forecast_harmonics = [torch.cos(2 * math.pi * i * t) for i in range(1, num_harmonics + 1)] + \
                             [torch.sin(2 * math.pi * i * t) for i in range(1, num_harmonics + 1)]
        self.forecast_basis = torch.stack(forecast_harmonics, dim=1)

    def forward(self, x, x_exog):
        theta_b, theta_f = self.block(x, x_exog)
        backcast = torch.matmul(self.backcast_basis.to(x.device), theta_b.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis.to(x.device), theta_f.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast

class NBeatsInterpretableX(nn.Module):
    def __init__(self, in_sz, out_sz, exog_sz):
        super().__init__()
        self.out_sz = out_sz
        self.trend_stack = nn.ModuleList([TrendBlockX(in_sz, out_sz, exog_sz, 256, 4, p_degree=2) for _ in range(3)])
        self.seasonality_stack = nn.ModuleList([SeasonalityBlockX(in_sz, out_sz, exog_sz, 256, 4, num_harmonics=5) for _ in range(3)])

    def forward(self, x, x_exog):
        total_forecast = torch.zeros(x.size(0), self.out_sz).to(x.device)
        backcast = x
        for block in self.trend_stack:
            b, f = block(backcast, x_exog)
            backcast = backcast - b
            total_forecast = total_forecast + f
        for block in self.seasonality_stack:
            b, f = block(backcast, x_exog)
            backcast = backcast - b
            total_forecast = total_forecast + f
        return total_forecast

def train_one_od(ori: str, dst: str):
    sub = df[(df.ori==ori)&(df.dst==dst)].sort_values("datetime").copy()
    if len(sub) < SEQ_LEN + 96:
        return None

    # <<< 2단계: 외부 피처(Exogenous Features) 생성 로직 활성화 >>>
    sub = sub.drop_duplicates(subset=['datetime']).set_index('datetime')
    ts = sub['fail_count']
    
    hour = sub.index.hour + sub.index.minute/60.0
    sub["hour_sin"] = np.sin(2*np.pi*hour/24)
    sub["hour_cos"] = np.cos(2*np.pi*hour/24)

    lag = pd.DataFrame(index=sub.index)
    for d in range(1, 8): lag[f'lag_{d}d'] = ts.shift(d * (24*4))
    for w in range(1, 4): lag[f'lag_{w}w'] = ts.shift(w * (7*24*4))
    
    static_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
        "before_holiday", "after_holiday", "temperature", "precipitation", "snowfall"
    ]
    exog_feat = pd.concat([sub[static_cols], lag], axis=1).fillna(0.0)

    scaler_exog = StandardScaler().fit(exog_feat.loc[exog_feat.index < TEST_START])
    exog_s = pd.DataFrame(scaler_exog.transform(exog_feat), index=exog_feat.index, columns=exog_feat.columns)
    
    # Target 스케일링
    raw_log = np.log1p(ts.values).reshape(-1, 1)
    train_raw_log = raw_log[ts.index < TEST_START]
    scaler_raw = MinMaxScaler().fit(train_raw_log)
    scaled = scaler_raw.transform(raw_log).flatten()

    # <<< 3단계: 시퀀스 데이터 생성에 외부 변수 포함 >>>
    X, X_exog, y = [], [], []
    for i in range(len(scaled) - SEQ_LEN - PRED_LEN + 1):
        X.append(scaled[i : i + SEQ_LEN])
        X_exog.append(exog_s.iloc[i : i + SEQ_LEN].values)
        y.append(scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
    
    X = np.array(X, dtype=np.float32)
    X_exog = np.array(X_exog, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    times = ts.index[SEQ_LEN + PRED_LEN - 1:]
    mask = times >= TEST_START
    X_train, X_test = X[~mask], X[mask]
    X_exog_train, X_exog_test = X_exog[~mask], X_exog[mask]
    y_train, y_test = y[~mask], y[mask]

    split = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:split], y_train[:split]
    X_exog_tr = X_exog_train[:split]
    X_val, y_val = X_train[split:], y_train[split:]
    X_exog_val = X_exog_train[split:]

    tr_loader = DataLoader(TimeSeriesDataset(X_tr, X_exog_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, X_exog_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TimeSeriesDataset(X_test, X_exog_test, y_test), batch_size=BATCH_SIZE)

    # <<< 4단계: 모델 생성 방식 변경 (N-BEATSx Interpretable) >>>
    exog_dim = X_exog.shape[2] * X_exog.shape[1]

    model = NBeatsInterpretableX(
        in_sz=SEQ_LEN,
        out_sz=PRED_LEN,
        exog_sz=exog_dim
    ).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss() 

    train_ls, val_ls, best_val, no_imp_cnt, PATIENCE, MIN_DELTA = [], [], float('inf'), 0, 5, 1e-4

    for ep in range(1, EPOCHS+1):
        # <<< 5단계: 학습/검증/테스트 루프 수정 (외부 변수 입력) >>>
        model.train()
        tot, ct = 0, 0
        for xb, x_exog_b, yb in tr_loader:
            xb, x_exog_b, yb = xb.to(DEVICE), x_exog_b.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, x_exog_b.view(xb.size(0), -1))
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
            tot += loss.item() * xb.size(0)
            ct  += xb.size(0)
        train_ls.append(tot/ct)

        model.eval()
        tot, ct = 0, 0
        with torch.no_grad():
            for xb, x_exog_b, yb in val_loader:
                xb, x_exog_b, yb = xb.to(DEVICE), x_exog_b.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, x_exog_b.view(xb.size(0), -1))
                loss = crit(pred, yb)
                tot += loss.item() * xb.size(0)
                ct += xb.size(0)
        val_loss = tot / ct
        val_ls.append(val_loss)

        print(f"[{ori}->{dst}] Ep{ep:02d}  train={train_ls[-1]:.4f}  val={val_loss:.4f}")

        if best_val - val_loss > MIN_DELTA:
            best_val, no_imp_cnt = val_loss, 0
        else:
            no_imp_cnt += 1
            if no_imp_cnt >= PATIENCE:
                print(f"Early stopping at epoch {ep}")
                break
    
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, x_exog_b, yb in test_loader:
            xb, x_exog_b, yb = xb.to(DEVICE), x_exog_b.to(DEVICE), yb.to(DEVICE)
            out = model(xb, x_exog_b.view(xb.size(0), -1)).cpu().numpy()
            preds.append(out)
            acts.append(yb.cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    acts_arr = np.concatenate(acts, axis=0)
    
    preds_flat = scaler_raw.inverse_transform(preds_arr[:, 0].reshape(-1,1)).flatten()
    acts_flat  = scaler_raw.inverse_transform(acts_arr[:, 0].reshape(-1,1)).flatten()
    
    preds_flat = np.expm1(preds_flat)
    acts_flat  = np.expm1(acts_flat)
    actual_int = np.round(acts_flat).astype(int)

    dep, arr = unidecode(ori), unidecode(dst)
    os.makedirs(f"{OUTPUT_DIR}/loss", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(train_ls, label="Train Loss")
    plt.plot(val_ls,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"{dep} → {arr} Loss"); plt.legend(); plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/loss/{ori}_{dst}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    test_dates = times[mask]
    
    df_pred = pd.DataFrame({
        "datetime":  test_dates,
        "ori":       ori,
        "dst":       dst,
        "actual":    actual_int,
        "predicted": np.maximum(0, preds_flat)
    })
    dt = pd.to_datetime(df_pred["datetime"])
    mask = ((dt.dt.hour >= 5)&(dt.dt.hour <= 23)) | ((dt.dt.hour==0)&(dt.dt.minute<=45))
    df_pred = df_pred.loc[mask].reset_index(drop=True)

    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,6),gridspec_kw={'hspace':0.4})
    ax1.plot(df_pred["datetime"], df_pred["actual"],    color='gray',    linewidth=1, label='GT')
    ax1.set_title(f"{dep} → {arr} GT");   ax1.set_ylabel("Fail Count"); ax1.legend()
    ax2.plot(df_pred["datetime"], df_pred["predicted"], color='tab:blue',linewidth=1, label='Pred')
    ax2.set_title(f"{dep} → {arr} Pred"); ax2.set_ylabel("Fail Count"); ax2.legend()
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/{ori}_{dst}_gt_pred.png", dpi=300)
    plt.close()

    os.makedirs(f"{OUTPUT_DIR}/prediction", exist_ok=True)
    df_pred.to_csv(f"{OUTPUT_DIR}/prediction/{ori}_{dst}.csv", index=False, encoding="utf-8-sig")

    y_true = df_pred["actual"].values
    y_pred = df_pred["predicted"].values
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    nz   = y_true > 0
    mape = (np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]))*100) if nz.any() else np.nan
    print(f"[{ori}->{dst}] MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")

    return {
        "scaler_target": scaler_raw, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape
    }

if __name__ == "__main__":
    target_od_pairs = [
        ('동대구', '서울'),    ('부산', '서울'),    ('대전', '서울'),    ('서울', '대전'),
        ('서울', '동대구'),    ('서울', '부산'),    ('동대구', '광명'),    ('광주송정', '용산'),
        ('울산', '서울'),    ('서울', '오송'),    ('부산', '광명'),    ('용산', '광주송정'),
        ('오송', '서울'),    ('서울', '천안아산'),    ('서울', '울산'),    ('대전', '광명'),
        ('천안아산', '서울'),    ('익산', '용산'),    ('광명', '대전'),    ('광명', '동대구'),
        ('동대구', '천안아산'),    ('김천구미', '서울'),    ('용산', '익산'),    ('광명', '부산'),
        ('전주', '용산'),    ('용산', '오송'),    ('경주', '서울'),    ('오송', '용산'),
        ('부산', '천안아산'),    ('용산', '전주'),    ('부산', '대전'),    ('광명', '오송'),
        ('용산', '천안아산'),    ('대전', '영등포'),    ('동대구', '대전'),    ('울산', '광명'),
        ('동대구', '오송'),    ('서울', '김천구미'),    ('부산', '오송'),    ('창원중앙', '서울'),
        ('광주송정', '서울'),    ('광주송정', '광명'),    ('순천', '용산'),    ('영등포', '대전'),
        ('오송', '광명'),    ('천안아산', '용산'),    ('서울', '경주'),    ('광명', '광주송정'),
        ('용산', '순천'),    ('익산', '광명')
    ]
    df_target = pd.DataFrame(target_od_pairs, columns=['ori', 'dst'])

    od_fail = df.groupby(["ori", "dst"])["fail_count"].sum().reset_index()
    merged_df = pd.merge(df_target, od_fail, on=['ori', 'dst'], how='inner')
    final_pairs_df = merged_df[merged_df['fail_count'] >= 1000]
    final_od_pairs = list(final_pairs_df[['ori', 'dst']].itertuples(index=False, name=None))
    
    print(f"지정된 리스트 중 실패 건수 1000건 이상인 OD쌍 개수: {len(final_od_pairs)}")

    metrics = []
    for ori, dst in final_od_pairs:
        print(f"\n=== {ori} → {dst} ===")
        res = train_one_od(ori, dst)
        if res:
            metrics.append({
                "ori": ori, "dst": dst,
                "mse": res["mse"], "rmse": res["rmse"],
                "mae": res["mae"], "mape": res["mape"]
            })

    # DuckDB 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = duckdb.connect(f"{OUTPUT_DIR}/results.duckdb")
    dfm = pd.DataFrame(metrics)
    con.execute("CREATE OR REPLACE TABLE metrics_summary AS SELECT * FROM dfm")
    con.execute("""
        CREATE OR REPLACE TABLE predictions (
            ori VARCHAR, dst VARCHAR, datetime TIMESTAMP,
            actual INTEGER, predicted DOUBLE
        )
    """)
    for ori,dst in final_od_pairs:
        path = f"{OUTPUT_DIR}/prediction/{ori}_{dst}.csv"
        if os.path.exists(path):
            dd = pd.read_csv(path, parse_dates=["datetime"])
            con.register("tmp", dd)
            con.execute("""
                INSERT INTO predictions
                SELECT ori, dst, datetime, actual, predicted
                FROM tmp
            """)
            con.unregister("tmp")
    dfm.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False, encoding="utf-8-sig")
    con.close()

    print(f"\n모든 작업 완료. 결과는 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
