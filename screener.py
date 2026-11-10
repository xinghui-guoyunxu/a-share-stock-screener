import json
import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

# ============ 参数 ============
MIN_CAP = 50e8       # 流通市值下限(元)
MAX_CAP = 300e8      # 流通市值上限(元)
MIN_GAIN = 0.05      # 信号日涨幅>=5%
DAYS_MIN = 5         # 信号日在往前第5个交易日
DAYS_MAX = 7         # 至往前第7个交易日
MAX_ABOVE = 0.03     # 现价不超过阳线高点的3%
TREND_MODE = "bull"  # "bull"=5>10>20日线; "ma10"=收盘价>10日线
WORKERS = 4
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
    start = (today - datetime.timedelta(days=100)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    try:
        df = retry(lambda: ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                "最高": "high", "最低": "low"})
    except Exception:
        prefix = "sh" if code.startswith("6") else "sz"
        df = retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=prefix + code, start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
    return df[["date", "open", "close", "high", "low"]]


def trend_ok(df):
    c = df["close"]
    ma5 = c.rolling(5).mean().iloc[-1]
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    if pd.isna(ma20):
        return False
    if TREND_MODE == "ma10":
        return c.iloc[-1] > ma10
    return ma5 > ma10 > ma20


def analyze(df):
    """返回 (最后交易日, 有效信号或None)"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "close", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < 25:
        return None, None
    last = len(df) - 1
    last_date = df.loc[last, "date"]
    latest_close = float(df.loc[last, "close"])
    if not trend_ok(df):
        return last_date, None

    # 按交易日: 往前第5~7个交易日, 取距今最近的有效信号
    for d in range(DAYS_MIN, DAYS_MAX + 1):
        i = last - d
        if i < 1:
            break
        prev_close = float(df.loc[i - 1, "close"])
        close_i = float(df.loc[i, "close"])
        open_i = float(df.loc[i, "open"])
        gain = close_i / prev_close - 1
        if gain < MIN_GAIN or close_i <= open_i:
            continue
        sig_low = float(df.loc[i, "low"])
        sig_high = float(df.loc[i, "high"])
        after_min = float(df.loc[i + 1:, "low"].min())
        if after_min < sig_low - 1e-9:
            continue
        if latest_close > sig_high * (1 + MAX_ABOVE):
            continue
        return last_date, {
            "signal_date": df.loc[i, "date"],
            "days_ago": d,
            "signal_gain": round(gain * 100, 2),
            "signal_low": round(sig_low, 2),
            "signal_high": round(sig_high, 2),
            "after_low": round(after_min, 2),
            "latest_close": round(latest_close, 2),
            "above_high": round((latest_close / sig_high - 1) * 100, 2),
            "safe_space": round((latest_close - sig_low) / latest_close * 100, 2),
        }
    return last_date, None


def main():
    t0 = time.time()
    print("1) 获取全市场行情与流通市值 ...")
    spot = retry(lambda: ak.stock_zh_a_spot_em(), times=3, wait=10)
    spot["流通市值"] = pd.to_numeric(spot["流通市值"], errors="coerce")
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)

    # 剔除科创板(688)、北交所(4/8/9开头)
    mask_ex = spot["代码"].str.startswith(("688", "4", "8", "9"))
    spot = spot[~mask_ex]
    pool = spot[(spot["流通市值"] >= MIN_CAP) & (spot["流通市值"] <= MAX_CAP)]
    print(f"剔除科创板/北交所后 {len(spot)} 只, 市值50~300亿共 {len(pool)} 只")
    stocks = [(r["代码"], r["名称"], float(r["流通市值"]))
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
    result = [r for r in rows if r.pop("_last") == cutoff]
    result.sort(key=lambda r: r["safe_space"], reverse=True)

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
    cols = ["code", "name", "signal_date", "days_ago", "signal_gain",
            "signal_low", "signal_high", "after_low", "latest_close",
            "above_high", "safe_space", "float_cap"]
    if result:
        print(pd.DataFrame(result)[cols].head(30).to_string())


if __name__ == "__main__":
    main()
