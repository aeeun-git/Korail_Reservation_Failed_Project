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
HORIZON_STEPS  = 4        # 1시간 뒤 (15분*4 스텝)
PRED_LEN       = 1        # 예측 1지점
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 0.0035
WEIGHT_DECAY   = 0.0001
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEEK_LAGS      = [1, 2]
TOD_LAGS_DAYS  = [3]
STEPS_PER_DAY  = 24 * 4
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

TEST_START = pd.to_datetime("20241120", format="%Y%m%d")

# ─────────────────────────────────────────────────────────────
# 2) 데이터 로딩 및 전처리
# ─────────────────────────────────────────────────────────────
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

def make_sequences(arr: np.ndarray, seq_len: int, horizon: int, pred_len: int):
    X, y = [], []
    for i in range(len(arr) - seq_len - horizon - pred_len + 2):
        X.append(arr[i:i+seq_len])
        start = i + seq_len + horizon - 1
        y.append(arr[start:start+pred_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X,y,w):
        self.X, self.y, self.w = X,y,w
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.w[i], dtype=torch.float32)
        )

class AttentionGRU(nn.Module):
    def __init__(self, in_sz, hid_sz, attn_sz, pred_len, num_layers=1):
        super().__init__()
        self.gru     = nn.GRU(in_sz, hid_sz, num_layers=num_layers, batch_first=True)
        self.attn_W1 = nn.Linear(hid_sz, attn_sz)
        self.attn_W2 = nn.Linear(hid_sz, attn_sz)
        self.attn_V  = nn.Linear(attn_sz, 1)
        self.fc1     = nn.Linear(hid_sz*2, hid_sz)
        self.fc2     = nn.Linear(hid_sz, pred_len)
    def attention(self, enc, hid):
        hid_e = hid.unsqueeze(1)
        score = self.attn_V(torch.tanh(self.attn_W1(enc) + self.attn_W2(hid_e)))
        wgt   = torch.softmax(score, dim=1)
        return (wgt*enc).sum(dim=1)
    def forward(self, x):
        enc, h_n = self.gru(x)
        h_last   = h_n[-1]
        ctx      = self.attention(enc, h_last)
        cat      = torch.cat([ctx, h_last], dim=1)
        return self.fc2(torch.relu(self.fc1(cat)))

def train_one_od(ori: str, dst: str):
    sub = df[(df.ori==ori)&(df.dst==dst)].copy()
    if len(sub) < SEQ_LEN + HORIZON_STEPS:
        return None

    df_feat = make_feature_df(sub)
    dates   = df_feat.index
    mask_tv, mask_test = dates<TEST_START, dates>=TEST_START

    scaler_f = StandardScaler().fit(df_feat.loc[mask_tv, FEATURE_COLS])
    scaler_t = StandardScaler().fit(df_feat.loc[mask_tv, ["fail_count"]])
    arr_f    = scaler_f.transform(df_feat[FEATURE_COLS])
    arr_t    = scaler_t.transform(df_feat[["fail_count"]])

    X_all, _ = make_sequences(arr_f, SEQ_LEN, HORIZON_STEPS, PRED_LEN)
    _, y_all = make_sequences(arr_t, SEQ_LEN, HORIZON_STEPS, PRED_LEN)
    y_all    = y_all.squeeze(-1)   # (N_seq,)

    offset      = SEQ_LEN + HORIZON_STEPS - 1
    seq_dates   = dates[offset:]
    w_all       = df_feat["is_weekend"].replace({0:1,1:1.2}).values[offset:]

    mask_seq_tv   = seq_dates<TEST_START
    mask_seq_test = seq_dates>=TEST_START

    X_tv, y_tv, w_tv = X_all[mask_seq_tv], y_all[mask_seq_tv], w_all[mask_seq_tv]
    split = int(len(X_tv)*0.8)
    X_tr, y_tr, w_tr = X_tv[:split], y_tv[:split], w_tv[:split]
    X_val,y_val,w_val= X_tv[split:],y_tv[split:],w_tv[split:]
    X_test,y_test,w_test = X_all[mask_seq_test], y_all[mask_seq_test], w_all[mask_seq_test]
    test_dates_seq      = seq_dates[mask_seq_test]

    tr_loader  = DataLoader(TimeSeriesDataset(X_tr, y_tr, w_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val, w_val), batch_size=BATCH_SIZE)
    test_loader= DataLoader(TimeSeriesDataset(X_test, y_test, w_test), batch_size=BATCH_SIZE)

    model = AttentionGRU(X_all.shape[2], 128, 32, PRED_LEN, num_layers=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = StepLR(opt, step_size=5, gamma=0.5)
    crit  = nn.MSELoss(reduction="none")

    train_ls, val_ls, best, cnt_no_imp = [], [], 1e9, 0
    for ep in range(1, EPOCHS+1):
        # --- train ---
        model.train()
        tot,ct=0,0
        for xb,yb,wb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).squeeze(-1)      # ← yb: [batch,1] → [batch]
            wb = wb.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)         # pred: [batch]
            loss = (crit(pred, yb)*wb).mean()
            loss.backward(); opt.step()
            tot+=loss.item()*xb.size(0); ct+=xb.size(0)
        train_ls.append(tot/ct)

        # --- val ---
        model.eval()
        tot,ct=0,0
        with torch.no_grad():
            for xb,yb,wb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).squeeze(-1)  # ← 여기도 동일
                wb = wb.to(DEVICE).unsqueeze(1)
                p  = model(xb).squeeze(-1)
                l  = (crit(p, yb)*wb).mean()
                tot+=l.item()*xb.size(0); ct+=xb.size(0)
        val_ls.append(tot/ct)

        print(f"[{ori}->{dst}] Ep{ep} tr={train_ls[-1]:.4f} val={val_ls[-1]:.4f}")
        if val_ls[-1] < best:
            best, cnt_no_imp = val_ls[-1], 0
        else:
            cnt_no_imp += 1
            if cnt_no_imp >= 15:
                print("Early stopping"); break
        sched.step()

    # --- Loss Curve ---
    dep, arr = unidecode(ori), unidecode(dst)
    os.makedirs("10OD/loss", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(train_ls, label="Train Loss")
    plt.plot(val_ls,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"{dep} → {arr} Loss"); plt.legend(); plt.grid()
    plt.savefig(f"10OD/loss/{ori}_{dst}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Test 예측 & 복원 ---
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb,yb,_ in test_loader:
            xb = xb.to(DEVICE)
            out= model(xb).squeeze(-1).cpu().numpy()
            preds.append(out); acts.append(yb.numpy())
    preds_arr  = np.concatenate(preds, axis=0)
    acts_arr   = np.concatenate(acts,  axis=0)
    preds_flat = scaler_t.inverse_transform(preds_arr.reshape(-1,1)).flatten()
    acts_flat  = scaler_t.inverse_transform(acts_arr.reshape(-1,1)).flatten()
    actual_int = np.round(acts_flat).astype(int)

    # --- 예측 시각 = 기준 시각 + 1시간 ---
    test_dates = test_dates_seq + np.timedelta64(HORIZON_STEPS*15, 'm')

    # --- 서비스 시간 필터링 ---
    df_pred = pd.DataFrame({
        "datetime":  test_dates,
        "ori":       ori,
        "dst":       dst,
        "actual":    actual_int,
        "predicted": preds_flat
    })
    dt = pd.to_datetime(df_pred["datetime"])
    mask = ((dt.dt.hour >= 5)&(dt.dt.hour <= 23)) | ((dt.dt.hour==0)&(dt.dt.minute<=45))
    df_pred = df_pred.loc[mask].reset_index(drop=True)

    # --- GT/Pred 플롯 ---
    os.makedirs("10OD/plots", exist_ok=True)
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,6),gridspec_kw={'hspace':0.4})
    ax1.plot(df_pred["datetime"], df_pred["actual"],    color='gray',    linewidth=1, label='GT')
    ax1.set_title(f"{dep} → {arr} GT");   ax1.set_ylabel("Fail Count"); ax1.legend()
    ax2.plot(df_pred["datetime"], df_pred["predicted"], color='tab:blue',linewidth=1, label='Pred')
    ax2.set_title(f"{dep} → {arr} Pred"); ax2.set_ylabel("Fail Count"); ax2.legend()
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(f"10OD/plots/{ori}_{dst}_gt_pred.png", dpi=300)
    plt.close()

    # --- CSV 저장 ---
    os.makedirs("10OD/prediction", exist_ok=True)
    df_pred.to_csv(f"10OD/prediction/{ori}_{dst}.csv", index=False, encoding="utf-8-sig")

    # --- 지표 계산 & 반환 ---
    y_true = df_pred["actual"].values
    y_pred = df_pred["predicted"].values
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    nz   = y_true > 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) if nz.any() else np.nan
    print(f"[{ori}->{dst}] MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")

    return {
        "state_dict":    model.state_dict(),
        "scaler_feat":   scaler_f,
        "scaler_target": scaler_t,
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
    for ori,dst in od_pairs:
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
    for ori,dst in od_pairs:
        path = f"10OD/prediction/{ori}_{dst}.csv"
        if os.path.exists(path):
            dd = pd.read_csv(path, parse_dates=["datetime"])
            con.register("tmp", dd)
            con.execute("""
                INSERT INTO predictions
                SELECT ori, dst, datetime, actual, predicted
                FROM tmp
            """)
            con.unregister("tmp")
    dfm.to_csv("10OD/metrics_summary.csv", index=False, encoding="utf-8-sig")
    con.close()
