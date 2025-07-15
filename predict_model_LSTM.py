import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from unidecode import unidecode
import duckdb

# ─────────────────────────────────────────────────────────────
# 1) 하이퍼파라미터 & lag 설정
# ─────────────────────────────────────────────────────────────
DATA_PATH      = "final.csv"
SEQ_LEN        = 96       # 과거 12시간(15분 간격 96스텝)
PRED_LEN       = 1        # 다음 15분 예측
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 0.0035
WEIGHT_DECAY   = 0.0001
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — lag 세팅
WEEK_LAGS      = [1, 2]
TOD_LAGS_DAYS  = [3]
STEPS_PER_DAY  = 24 * 4
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# — 테스트셋 시작일
TEST_START = pd.to_datetime("20241120", format="%Y%m%d")

# ──────────────────────────────────────────────────────────────
# 2) 데이터 로딩 및 전처리
# ──────────────────────────────────────────────────────────────
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

STATIC_FEATURES     = [
    "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
    "is_holiday","before_holiday","after_holiday",
    "temperature","precipitation","snowfall"
]
WEEKLY_LAG_FEATURES = [f"lag_{w}w"     for w in WEEK_LAGS]
DAILY_LAG_FEATURES  = [f"lag_{d}d"     for d in range(1, max(TOD_LAGS_DAYS)+1)]
TOD_AVG_FEATURES    = [f"tod_avg_{d}d" for d in TOD_LAGS_DAYS]
SHORT_LAGS          = [f"lag_{k}step"  for k in range(1,5)]
DIFF_FEATURE        = ["diff_1step"]

FEATURE_COLS = (
    STATIC_FEATURES +
    WEEKLY_LAG_FEATURES +
    DAILY_LAG_FEATURES +
    TOD_AVG_FEATURES +
    SHORT_LAGS +
    DIFF_FEATURE
)

def make_feature_df(sub: pd.DataFrame) -> pd.DataFrame:
    df_feat = sub.set_index("datetime").copy()
    hour = df_feat.index.hour + df_feat.index.minute/60.0
    df_feat["hour_sin"] = np.sin(2*np.pi*hour/24)
    df_feat["hour_cos"] = np.cos(2*np.pi*hour/24)
    for w in WEEK_LAGS:
        df_feat[f"lag_{w}w"] = df_feat["fail_count"].shift(w*STEPS_PER_WEEK)
    for d in range(1, max(TOD_LAGS_DAYS)+1):
        df_feat[f"lag_{d}d"] = df_feat["fail_count"].shift(d*STEPS_PER_DAY)
    for d in TOD_LAGS_DAYS:
        cols = [f"lag_{k}d" for k in range(1,d+1)]
        df_feat[f"tod_avg_{d}d"] = df_feat[cols].mean(axis=1)
    for k in range(1,5):
        df_feat[f"lag_{k}step"] = df_feat["fail_count"].shift(k)
    df_feat["diff_1step"] = df_feat["fail_count"] - df_feat["fail_count"].shift(1)
    return df_feat.dropna()

def make_sequences(arr: np.ndarray, seq_len: int, pred_len: int):
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

# ──────────────────────────────────────────────────────────────
# AttentionLSTM 클래스 (LSTM 기반 어텐션)
# ──────────────────────────────────────────────────────────────
class AttentionLSTM(nn.Module):
    def __init__(self, in_size, hid_size, attn_size, pred_len, num_layers=1):
        super().__init__()
        self.lstm    = nn.LSTM(in_size, hid_size, num_layers=num_layers, batch_first=True)
        self.attn_W1 = nn.Linear(hid_size, attn_size)
        self.attn_W2 = nn.Linear(hid_size, attn_size)
        self.attn_V  = nn.Linear(attn_size, 1)
        self.fc1     = nn.Linear(hid_size*2, hid_size)
        self.fc2     = nn.Linear(hid_size, pred_len)

    def attention(self, enc_out, hidden):
        hid_exp = hidden.unsqueeze(1)
        score   = self.attn_V(torch.tanh(self.attn_W1(enc_out) + self.attn_W2(hid_exp)))
        weights = torch.softmax(score, dim=1)
        return (weights * enc_out).sum(dim=1)

    def forward(self, x):
        enc_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        ctx    = self.attention(enc_out, h_last)
        cat    = torch.cat([ctx, h_last], dim=1)
        out    = torch.relu(self.fc1(cat))
        return self.fc2(out)

def train_one_od(ori: str, dst: str):
    sub = df[(df["ori"]==ori)&(df["dst"]==dst)].copy()
    if len(sub) < SEQ_LEN + PRED_LEN: return None

    # feature + target 준비
    df_feat     = make_feature_df(sub)
    dates       = df_feat.index
    mask_tv     = dates < TEST_START
    scaler_feat = StandardScaler().fit(df_feat.loc[mask_tv, FEATURE_COLS])
    scaler_tgt  = StandardScaler().fit(df_feat.loc[mask_tv, ["fail_count"]])
    arr_f       = scaler_feat.transform(df_feat[FEATURE_COLS])
    arr_t       = scaler_tgt.transform(df_feat[["fail_count"]])

    X, _        = make_sequences(arr_f, SEQ_LEN, PRED_LEN)
    _, y        = make_sequences(arr_t, SEQ_LEN, PRED_LEN)
    y           = y.squeeze(-1)  # (N_seq,)

    # 시퀀스 기준시각, 가중치
    seq_dates   = dates[SEQ_LEN:]
    w_all       = df_feat["is_weekend"].replace({0:1.0,1:1.2}).values[SEQ_LEN:]

    # train/val/test split
    mask_seq_tv   = seq_dates < TEST_START
    mask_seq_test = seq_dates >= TEST_START
    test_dates_seq= seq_dates[mask_seq_test]

    X_tv, y_tv, w_tv = X[mask_seq_tv], y[mask_seq_tv], w_all[mask_seq_tv]
    split            = int(len(X_tv)*0.8)
    X_tr, y_tr, w_tr = X_tv[:split],    y_tv[:split],    w_tv[:split]
    X_val,y_val,w_val= X_tv[split:],    y_tv[split:],    w_tv[split:]
    X_test,y_test,w_test = X[mask_seq_test], y[mask_seq_test], w_all[mask_seq_test]

    tr_loader   = DataLoader(TimeSeriesDataset(X_tr, y_tr, w_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader  = DataLoader(TimeSeriesDataset(X_val, y_val, w_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, w_test), batch_size=BATCH_SIZE)

    # 모델 정의 (LSTM attention)
    model     = AttentionLSTM(X.shape[2], 128, 32, PRED_LEN, num_layers=2).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.MSELoss(reduction="none")

    train_losses, val_losses = [], []
    best_val, no_imp = float('inf'), 0
    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        tot, cnt = 0, 0
        for xb, yb, wb in tr_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = (criterion(pred, yb.squeeze(-1)) * wb).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            cnt += xb.size(0)
        train_losses.append(tot/cnt)
        scheduler.step()

        # val
        model.eval()
        tot, cnt = 0, 0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE).unsqueeze(1)
                pred = model(xb).squeeze(-1)
                l    = (criterion(pred, yb.squeeze(-1)) * wb).mean()
                tot += l.item() * xb.size(0)
                cnt += xb.size(0)
        val_losses.append(tot/cnt)

        # early stopping
        if val_losses[-1] < best_val:
            best_val, no_imp = val_losses[-1], 0
        else:
            no_imp += 1
            if no_imp >= 15:
                break

    dep_en, arr_en = unidecode(ori), unidecode(dst)
    # ── loss curve 저장 ─────────────────────────────────────────
    os.makedirs("10OD/loss", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"{dep_en} → {arr_en} Loss"); plt.legend(); plt.grid()
    plt.savefig(f"10OD/loss/{ori}_{dst}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── test 예측 & 역스케일 ─────────────────────────────────────
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            xb = xb.to(DEVICE)
            out= model(xb).squeeze(-1).cpu().numpy()
            preds.append(out)
            acts.append(yb.cpu().numpy().squeeze(-1))
    preds_arr  = np.concatenate(preds, axis=0)
    acts_arr   = np.concatenate(acts,  axis=0)
    preds_flat = scaler_tgt.inverse_transform(preds_arr.reshape(-1,1)).flatten()
    acts_flat  = scaler_tgt.inverse_transform(acts_arr.reshape(-1,1)).flatten()
    actual_int = np.round(acts_flat).astype(int)

    # ── 예측 시각 (15분 후) ───────────────────────────────────────
    test_dates = test_dates_seq + np.timedelta64(15, 'm')

    # ── 서비스 시간 필터링 ───────────────────────────────────────
    df_pred = pd.DataFrame({
        "datetime":  test_dates,
        "ori":       ori,
        "dst":       dst,
        "actual":    actual_int,
        "predicted": preds_flat
    })
    dt   = pd.to_datetime(df_pred["datetime"])
    mask = ((dt.dt.hour >= 5)&(dt.dt.hour <= 23)) | ((dt.dt.hour == 0)&(dt.dt.minute <= 45))
    df_pred = df_pred.loc[mask].reset_index(drop=True)

    # ── GT/Pred 플롯 ─────────────────────────────────────────────
    os.makedirs("10OD/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,6),gridspec_kw={'hspace':0.4})
    ax1.plot(df_pred["datetime"], df_pred["actual"],    color='gray',    linewidth=1, label='GT')
    ax1.set_title(f"{dep_en} → {arr_en} GT"); ax1.set_ylabel("Fail Count"); ax1.legend()
    ax2.plot(df_pred["datetime"], df_pred["predicted"], color='tab:blue',linewidth=1, label='Pred')
    ax2.set_title(f"{dep_en} → {arr_en} Pred"); ax2.set_ylabel("Fail Count"); ax2.legend()
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(f"10OD/plots/{ori}_{dst}_gt_pred.png", dpi=300)
    plt.close()

    # ── CSV 저장 ─────────────────────────────────────────────────
    os.makedirs("10OD/prediction", exist_ok=True)
    df_pred.to_csv(f"10OD/prediction/{ori}_{dst}.csv", index=False, encoding="utf-8-sig")

    # ── 지표 계산 & 반환 ─────────────────────────────────────────
    y_true = df_pred["actual"].values
    y_pred = df_pred["predicted"].values
    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mse)
    mae    = mean_absolute_error(y_true, y_pred)
    nz     = y_true > 0
    mape   = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) if nz.any() else np.nan
    print(f"[{ori}->{dst}] MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")

    return {
        "state_dict":    model.state_dict(),
        "scaler_feat":   scaler_feat,
        "scaler_target": scaler_tgt,
        "mse":           mse,
        "rmse":          rmse,
        "mae":           mae,
        "mape":          mape
    }

if __name__ == "__main__":
    od_pairs = [
        ("서울","대전"),("대전","서울"),
        ("서울","동대구"),("동대구","서울"),
        ("서울","부산"),("부산","서울"),
        ("광명","동대구"),("광명","대전"),
        ("서울","천안아산"),("수원","영등포"),
    ]
    metrics = []
    for ori, dst in od_pairs:
        print(f"=== {ori} → {dst} ===")
        res = train_one_od(ori, dst)
        if res:
            metrics.append({
                "ori": ori, "dst": dst,
                "mse": res["mse"], "rmse": res["rmse"],
                "mae": res["mae"], "mape": res["mape"]
            })

    # DuckDB에 저장
    os.makedirs("10OD", exist_ok=True)
    con = duckdb.connect("10OD/results.duckdb")
    dfm = pd.DataFrame(metrics)
    con.execute("CREATE TABLE IF NOT EXISTS metrics_summary AS SELECT * FROM dfm")
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ori VARCHAR, dst VARCHAR, datetime TIMESTAMP,
            actual INTEGER, predicted DOUBLE
        )
    """)
    for ori, dst in od_pairs:
        path = f"10OD/prediction/{ori}_{dst}.csv"
        if os.path.exists(path):
            tmp = pd.read_csv(path, parse_dates=["datetime"])
            con.register("tmp", tmp)
            con.execute("""
                INSERT INTO predictions
                SELECT ori, dst, datetime, actual, predicted
                FROM tmp
            """)
            con.unregister("tmp")
    dfm.to_csv("10OD/metrics_summary.csv", index=False, encoding="utf-8-sig")
    con.close()
