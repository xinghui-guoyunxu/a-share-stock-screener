import json
import os
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

# ============ 参数 ============
MIN_CAP = 50e8        # 流通市值下限(元)
MAX_CAP = 300e8       # 流通市值上限(元)
MIN_GAIN = 0.05       # 信号日涨幅>=5%
DAYS_MIN = 5          # 信号日在往前第5个交易日
DAYS_MAX = 7          # 至往前第7个交易日
MAX_ABOVE = 0.03      # 现价不超过阳线高点3%
GAIN20_MAX = 0.18     # 20日涨幅上限(打分项)
DIST_MA20_MAX = 0.08  # 距MA20上限(打分项)
PULL_DAYS = 3         # 回踩窗口(最近N个交易日,不含今天)
TOUCH_TOL = 0.01      # 回踩容差
VOL_RATIO = 0.8       # 回踩日缩量比例
TOP_SCORE = 5         # 精选得分门槛
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
    start = (today - datetime.timedelta(days=150)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    try:
        df = retry(lambda: ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                "最高": "high", "最低": "low", "成交量": "volume"})
    except Exception:
        prefix = "sh" if code.startswith("6") else "sz"
        df = retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=prefix + code, start_date=start,
            end_date=end, adjust="qfq"), times=2, wait=2)
        if "volume" not in df.columns:
            df["volume"] = df["amount"] if "amount" in df.columns else 0
    return df[["date", "open", "close", "high", "low", "volume"]]


def analyze(df):
    """返回 (最后交易日, 有效信号或None)"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        return None, None
    last = len(df) - 1
    last_date = df.loc[last, "date"]
    latest_close = float(df.loc[last, "close"])

    c = df["close"]
    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    vol_ma5 = df["volume"].rolling(5).shift(1) if False else df["volume"].rolling(5).mean().shift(1)

    # 硬性: 最新收盘>=MA20, MA20今天高于5天前
    if latest_close < ma20.iloc[last]:
        return last_date, None
    if not ma20.iloc[last] > ma20.iloc[last - 5]:
        return last_date, None

    # 硬性: 信号日(往前第5~7个交易日, 取距今最近的有效信号)
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

        # ---- 打分项 ----
        hits = []
        if ma5.iloc[last] > ma10.iloc[last] > ma20.iloc[last]:
            hits.append("多头排列")
        gain20 = latest_close / float(c.iloc[last - 20]) - 1
        if gain20 <= GAIN20_MAX:
            hits.append("20日涨幅达标")
        dist = latest_close / ma20.iloc[last] - 1
        if 0 <= dist <= DIST_MA20_MAX:
            hits.append("近MA20")
        if df.loc[last, "close"] > df.loc[last, "open"] and \
                df.loc[last, "close"] > df.loc[last - 1, "close"]:
            hits.append("重新收阳")
        pull_date, pull_to, shrink = "", "", False
        for j in range(last - 1, max(i, last - PULL_DAYS - 1), -1):
            low_j = df.loc[j, "low"]
            touch = None
            if low_j <= ma10.iloc[j] * (1 + TOUCH_TOL):
                touch = "MA10"
            elif low_j <= ma20.iloc[j] * (1 + TOUCH_TOL):
                touch = "MA20"
            if touch is None or df.loc[j, "close"] < ma20.iloc[j] * 0.99:
                continue
            pull_date, pull_to = df.loc[j, "date"], touch
            vm = vol_ma5.iloc[j]
            shrink = (not pd.isna(vm)) and vm > 0 and df.loc[j, "volume"] <= vm * VOL_RATIO
            break
        if pull_date:
            hits.append("回踩" + pull_to)
        if shrink:
            hits.append("回踩缩量")

        score = sum(1 for h in hits if h in
                    ("多头排列", "20日涨幅达标", "近MA20", "重新收阳")) \
            + (1 if pull_date else 0) + (1 if shrink else 0)

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
            "gain20": round(gain20 * 100, 2),
            "dist_ma20": round(dist * 100, 2),
            "pullback_date": pull_date,
            "score": score,
            "hit": "、".join(hits),
        }
    return last_date, None


def main():
    t0 = time.time()
    print("1) 获取全市场行情与流通市值 ...")
    spot = retry(lambda: ak.stock_zh_a_spot_em(), times=3, wait=10)
    spot["流通市值"] = pd.to_numeric(spot["流通市值"], errors="coerce")
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    spot["名称"] = spot["名称"].astype(str)

    # 剔除科创板(688)、北交所(4/8/9开头)、ST/*ST/退市
    mask_ex = spot["代码"].str.startswith(("688", "4", "8", "9"))
    mask_st = spot["名称"].str.contains("ST|退", case=False, regex=True)
    spot = spot[~mask_ex & ~mask_st]
    pool = spot[(spot["流通市值"] >= MIN_CAP) & (spot["流通市值"] <= MAX_CAP)]
    print(f"剔除科创板/北交所/ST后 {len(spot)} 只, 市值50~300亿共 {len(pool)} 只")
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
    result.sort(key=lambda r: (r["score"], r["safe_space"]), reverse=True)

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

    # 导出文件: csv + txt + 通达信blk(全部/精选)
    if result:
        df_out = pd.DataFrame(result).rename(columns={
            "code": "代码", "name": "名称", "score": "得分", "hit": "命中条件",
            "signal_date": "信号日", "days_ago": "距今交易日",
            "signal_gain": "信号涨幅%", "signal_low": "信号日最低价",
            "signal_high": "信号日最高价", "after_low": "后续最低价",
            "latest_close": "最新收盘价", "above_high": "现价高于阳线高点%",
            "safe_space": "安全空间%", "gain20": "20日涨幅%",
            "dist_ma20": "距MA20%", "pullback_date": "回踩日",
            "float_cap": "流通市值(亿)"})
        first = ["代码", "名称", "得分", "命中条件"]
        df_out = df_out[first + [c for c in df_out.columns if c not in first]]
        df_out.to_csv("docs/stocks.csv", index=False, encoding="utf-8-sig")
        with open("docs/stocks.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(r["code"] for r in result) + "\n")

        def blk(rs):
            return "".join(("1" if r["code"].startswith("6") else "0")
                           + r["code"] + "\r\n" for r in rs)
        with open("docs/tdx_block.blk", "w", encoding="gbk", newline="") as fp:
            fp.write(blk(result))
        top = [r for r in result if r["score"] >= TOP_SCORE]
        with open("docs/tdx_top.blk", "w", encoding="gbk", newline="") as fp:
            fp.write(blk(top))

    print("\n" + "#" * 60)
    print(f"截止日 {cutoff} | 市值池 {len(pool)} | 失败 {len(failed)} | 入选 {len(result)} | 总耗时 {int(time.time()-t0)} 秒")
    if result:
        dist = pd.Series([r["score"] for r in result]).value_counts().sort_index(ascending=False)
        print("得分分布:", dict(dist))
        cols = ["code", "name", "score", "hit", "signal_date", "days_ago",
                "signal_gain", "latest_close", "safe_space", "float_cap"]
        print(pd.DataFrame(result)[cols].head(30).to_string())


if __name__ == "__main__":
    main()
