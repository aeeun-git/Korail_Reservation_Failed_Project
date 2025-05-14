import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

# ──────────────────────────────────────────────────────────────
# 1) 하이퍼파라미터 & lag 설정
# ──────────────────────────────────────────────────────────────
DATA_PATH      = "preprocessed_6OD_rev.csv"
SEQ_LEN        = 36
PRED_LEN       = 1       # 10분
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 0.001
WEIGHT_DECAY   = 0.0001
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — lag 세팅
WEEK_LAGS      = [1]
TOD_LAGS_DAYS  = [3,4]
STEPS_PER_DAY  = 24 * 6
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# — 테스트셋 시작일
TEST_START = pd.to_datetime("20241120", format="%Y%m%d")

# ──────────────────────────────────────────────────────────────
# 2) 데이터 로딩 및 전처리 (STL 제거)
# ──────────────────────────────────────────────────────────────
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found."
df = pd.read_csv(DATA_PATH, dtype=str)

df = df.rename(columns={
    "출발역":       "ori",
    "도착역":       "dst",
    "출발일자":     "date",
    "출발시각":     "time",
    "예약실패건수": "fail_count",
    "요일_사인":    "dow_sin",
    "요일_코사인":  "dow_cos",
    "주말여부":     "is_weekend",
    "공휴일여부":   "is_holiday",
    "공휴일전여부": "before_holiday",
    "공휴일후여부": "after_holiday",
    "기온":         "temperature",
    "강수량":       "precipitation",
    "적설":         "snowfall",
})

df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
df = df.sort_values("datetime")
df["fail_count"] = df["fail_count"].astype(int)
for col in ["dow_sin","dow_cos","is_weekend","is_holiday","before_holiday","after_holiday"]:
    df[col] = df[col].astype(float)
for col in ["temperature","precipitation","snowfall"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0)
df = df.drop(columns=["date","time"])

STATIC_FEATURES     = [
    "hour_sin","hour_cos",
    "dow_sin","dow_cos","is_weekend",
    "is_holiday","before_holiday","after_holiday",
    "temperature","precipitation","snowfall"
]
WEEKLY_LAG_FEATURES = [f"lag_{w}w"     for w in WEEK_LAGS]
MAX_DAY_LAG         = max(TOD_LAGS_DAYS)
DAILY_LAG_FEATURES  = [f"lag_{d}d"     for d in range(1, MAX_DAY_LAG+1)]
TOD_AVG_FEATURES    = [f"tod_avg_{d}d" for d in TOD_LAGS_DAYS]

FEATURE_COLS = (
    STATIC_FEATURES +
    WEEKLY_LAG_FEATURES +
    DAILY_LAG_FEATURES +
    TOD_AVG_FEATURES
)

def make_feature_df(sub: pd.DataFrame) -> pd.DataFrame:
    df_feat = sub.set_index("datetime").copy()
    hour = df_feat.index.hour + df_feat.index.minute/60
    df_feat["hour_sin"] = np.sin(2*np.pi*hour/24)
    df_feat["hour_cos"] = np.cos(2*np.pi*hour/24)
    for w in WEEK_LAGS:
        df_feat[f"lag_{w}w"] = df_feat["fail_count"].shift(w * STEPS_PER_WEEK)
    for d in range(1, MAX_DAY_LAG+1):
        df_feat[f"lag_{d}d"] = df_feat["fail_count"].shift(d * STEPS_PER_DAY)
    for d in TOD_LAGS_DAYS:
        cols = [f"lag_{k}d" for k in range(1,d+1)]
        df_feat[f"tod_avg_{d}d"] = df_feat[cols].mean(axis=1)
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

class AttentionGRU(nn.Module):
    def __init__(self, in_size, hid_size, attn_size, pred_len,
                 num_layers=2):   # ← num_layers=2 로 업그레이드
        super().__init__()
        self.gru = nn.GRU(in_size, hid_size,
                          num_layers=num_layers,
                          dropout=0.1,            # optional: 레이어 간 dropout
                          batch_first=True)
        self.attn_W1 = nn.Linear(hid_size, attn_size)
        self.attn_W2 = nn.Linear(hid_size, attn_size)
        self.attn_V  = nn.Linear(attn_size, 1)
        self.fc1     = nn.Linear(hid_size*2, hid_size)
        self.fc2     = nn.Linear(hid_size, pred_len)

    def attention(self, enc_out, hidden):
        hid_exp = hidden.unsqueeze(1)
        score   = self.attn_V(torch.tanh(
                    self.attn_W1(enc_out) + self.attn_W2(hid_exp)))
        weights = torch.softmax(score, dim=1)
        return (weights * enc_out).sum(dim=1)

    def forward(self, x):
        enc_out, h_n = self.gru(x)
        h_last = h_n[-1]            # 마지막 레이어의 마지막 히든
        ctx    = self.attention(enc_out, h_last)
        cat    = torch.cat([ctx, h_last], dim=1)
        out    = torch.relu(self.fc1(cat))
        return self.fc2(out)

def train_one_od(ori: str, dst: str):
    sub = df[(df["ori"]==ori)&(df["dst"]==dst)].copy()
    if len(sub) < SEQ_LEN + PRED_LEN:
        return None

    df_feat = make_feature_df(sub)
    dates   = df_feat.index

    mask_tv   = dates < TEST_START
    mask_test = dates >= TEST_START

    scaler_feat = StandardScaler().fit(df_feat.loc[mask_tv, FEATURE_COLS])
    scaler_tgt  = StandardScaler().fit(df_feat.loc[mask_tv, ["fail_count"]])

    arr_feat = scaler_feat.transform(df_feat[FEATURE_COLS])
    arr_tgt  = scaler_tgt.transform(df_feat[["fail_count"]])

    X, _     = make_sequences(arr_feat, SEQ_LEN, PRED_LEN)
    _, y     = make_sequences(arr_tgt,  SEQ_LEN, PRED_LEN)
    y        = y.squeeze(-1)
    raw_counts = df_feat["fail_count"].values[SEQ_LEN:]
    # 실패 건수 5건 초과면 2배, 아니면 1배
    w_all = np.where(raw_counts > 5, 2.0, 1.0)
    seq_dates= dates[SEQ_LEN:]

    mask_tv_seq   = seq_dates < TEST_START
    mask_test_seq = seq_dates >= TEST_START

    X_tv, y_tv, w_tv = X[mask_tv_seq], y[mask_tv_seq], w_all[mask_tv_seq]
    split = int(len(X_tv) * 0.8)
    X_tr, y_tr, w_tr = X_tv[:split], y_tv[:split], w_tv[:split]
    X_val, y_val, w_val = X_tv[split:], y_tv[split:], w_tv[split:]
    X_test, y_test, w_test = X[mask_test_seq], y[mask_test_seq], w_all[mask_test_seq]

    tr_loader  = DataLoader(TimeSeriesDataset(X_tr, y_tr, w_tr),
                            batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val, w_val),
                            batch_size=BATCH_SIZE)
    test_loader= DataLoader(TimeSeriesDataset(X_test, y_test, w_test),
                            batch_size=BATCH_SIZE)

    model     = AttentionGRU(X.shape[2], 64, 32, PRED_LEN,
                             num_layers=2).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.L1Loss(reduction="none")

    best_val, no_imp = float('inf'), 0
    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        total, cnt = 0, 0
        for xb, yb, wb in tr_loader:
            xb,yb,wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = (criterion(pred, yb.squeeze(-1)) * wb).mean()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            cnt   += xb.size(0)
        scheduler.step()
        train_loss = total / cnt

        # Validation
        model.eval()
        total, cnt = 0, 0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb,yb,wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                pred = model(xb).squeeze(-1)
                l = (criterion(pred, yb.squeeze(-1)) * wb).mean()
                total += l.item() * xb.size(0)
                cnt   += xb.size(0)
        val_loss = total / cnt

        print(f"[{ori}->{dst}] Epoch {epoch}/{EPOCHS}  "
              f"train_MAE={train_loss:.4f}  val_MAE={val_loss:.4f}")

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
        else:
            no_imp += 1
            if no_imp >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

    # ────────────────────────────────────────────
    # 테스트 예측 및 저장
    # ────────────────────────────────────────────
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb).squeeze(-1).cpu().numpy()
            preds.append(out)
            acts.append(yb.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    acts  = np.concatenate(acts).flatten()
    test_dates = seq_dates[mask_test_seq]

    actual_unscaled = scaler_tgt.inverse_transform(acts.reshape(-1,1)).flatten()
    pred_unscaled   = scaler_tgt.inverse_transform(preds.reshape(-1,1)).flatten()
    actual_int = np.round(actual_unscaled).astype(int)
    pred_clip  = np.clip(pred_unscaled, 0, None)

    df_pred = pd.DataFrame({
        "datetime":  test_dates,
        "ori":       ori,
        "dst":       dst,
        "actual":    actual_int,
        "predicted": pred_clip
    })
    os.makedirs("models", exist_ok=True)
    df_pred.to_csv(f"models/predictions_{ori}_{dst}.csv",
                   index=False, encoding="utf-8-sig")

    return {
        "state_dict":    model.state_dict(),
        "scaler_feat":   scaler_feat,
        "scaler_target": scaler_tgt
    }

if __name__ == "__main__":
    od_pairs  = df[["ori","dst"]].drop_duplicates().values.tolist()
    all_models = {}
    for ori, dst in od_pairs:
        print(f"\n=== Training {ori} → {dst} ===")
        res = train_one_od(ori, dst)
        if res:
            all_models[f"{ori}_{dst}"] = res
    torch.save(all_models, "models/all_od_models_v5.pt")
