import json
import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

# ============ 参数 ============
MIN_CAP = 50e8      # 流通市值下限(元)
MAX_CAP = 300e8     # 流通市值上限(元)
LOOKBACK = 10       # 最近10个交易日
MIN_GAIN = 0.05     # 涨幅>=5%
WORKERS = 4         # 并发数
# ==============================


def retry(func, times=3, wait=3):
    last = None
    for i in range(times):
        try:
            return func()
        except Exception as e:
            last = e
            time.sleep(wait * (i + 1))
    raise last


def fetch_kline(code):
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=30)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    try:
        df = retry(lambda: ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                "最高": "high", "最低": "low"})
    except Exception:
        # 备用源：腾讯
        if code.startswith("6"):
            prefix = "sh"
        elif code.startswith(("4", "8", "9")):
            raise
        else:
            prefix = "sz"
        df = retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=prefix + code, start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
    return df[["date", "open", "close", "high", "low"]]


def analyze(df):
    """返回 (最后交易日, 有效信号或None)"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "close", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < LOOKBACK + 1:
        return None, None
    last = len(df) - 1
    last_date = df.loc[last, "date"]
    latest_close = float(df.loc[last, "close"])

    # 从最近一天往前找，第一个有效信号即"最近有效信号日"
    for i in range(last, last - LOOKBACK, -1):
        prev_close = float(df.loc[i - 1, "close"])
        close_i = float(df.loc[i, "close"])
        open_i = float(df.loc[i, "open"])
        gain = close_i / prev_close - 1
        if gain >= MIN_GAIN and close_i > open_i:
            sig_low = float(df.loc[i, "low"])
            after = df.loc[i + 1:, "low"]
            after_min = float(after.min()) if len(after) else None
            if after_min is None or after_min >= sig_low - 1e-9:
                return last_date, {
                    "signal_date": df.loc[i, "date"],
                    "signal_gain": round(gain * 100, 2),
                    "signal_low": round(sig_low, 2),
                    "after_low": None if after_min is None else round(after_min, 2),
                    "latest_close": round(latest_close, 2),
                    "safe_space": round((latest_close - sig_low) / latest_close * 100, 2),
                }
    return last_date, None


def main():
    t0 = time.time()
    print("1) 获取全市场行情与流通市值 ...")
    spot = retry(lambda: ak.stock_zh_a_spot_em(), times=3, wait=10)
    print("行情列名:", list(spot.columns))
    if "流通市值" not in spot.columns:
        raise SystemExit("找不到[流通市值]列，请把上面的列名发给我")
    spot["流通市值"] = pd.to_numeric(spot["流通市值"], errors="coerce")
    print("流通市值抽样(单位应为元):")
    print(spot[["代码", "名称", "流通市值"]].head(5).to_string())

    pool = spot[(spot["流通市值"] >= MIN_CAP) & (spot["流通市值"] <= MAX_CAP)]
    print(f"全市场 {len(spot)} 只, 市值50~300亿共 {len(pool)} 只")
    stocks = [(str(r["代码"]).zfill(6), r["名称"], float(r["流通市值"]))
              for _, r in pool.iterrows()]

    print("2) 逐只拉取日K并筛选 ...")
    rows, failed, last_dates = [], [], []

    def work(item):
        code, name, cap = item
        df = fetch_kline(code)
        last_date, sig = analyze(df)
        return code, name, cap, last_date, sig

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, s): s for s in stocks}
        for f in as_completed(futs):
            s = futs[f]
            done += 1
            try:
                code, name, cap, last_date, sig = f.result()
                if last_date:
                    last_dates.append(last_date)
                if sig:
                    sig.update({"code": code, "name": name,
                                "float_cap": round(cap / 1e8, 2),
                                "_last": last_date})
                    rows.append(sig)
            except Exception as e:
                failed.append(s[0])
                print("失败:", s[0], repr(e)[:100])
            if done % 100 == 0:
                print(f"进度 {done}/{len(stocks)}, 已用时 {int(time.time()-t0)} 秒")

    cutoff = max(last_dates) if last_dates else None
    # 剔除停牌股(最后交易日早于截止日)
    result = [r for r in rows if r.pop("_last") == cutoff]
    result.sort(key=lambda r: (r["signal_date"], r["safe_space"]), reverse=True)

    out = {
        "updated_at": (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
        "cutoff_date": cutoff,
        "pool_count": len(pool),
        "failed_count": len(failed),
        "failed_codes": failed[:50],
        "count": len(result),
        "note": "流通市值为最新值，非信号日历史市值；价格为前复权。",
        "stocks": result,
    }
    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)

    print("\n" + "#" * 60)
    print(f"截止日 {cutoff} | 市值池 {len(pool)} | 失败 {len(failed)} | 入选 {len(result)} | 总耗时 {int(time.time()-t0)} 秒")
    if result:
        print(pd.DataFrame(result)[["code", "name", "signal_date", "signal_gain",
              "signal_low",
