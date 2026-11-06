import time
import datetime
import traceback
import akshare as ak
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

today = datetime.date.today()
start = (today - datetime.timedelta(days=45)).strftime("%Y%m%d")
end = today.strftime("%Y%m%d")
results = {}


def run(name, func):
    print("\n" + "=" * 60)
    print("测试:", name)
    t = time.time()
    try:
        df = func()
        cost = round(time.time() - t, 1)
        print(f"成功, 耗时 {cost} 秒, 行数 {len(df)}")
        print("列名:", list(df.columns))
        print(df.tail(3).to_string())
        results[name] = f"成功 ({cost}s, {len(df)}行)"
    except Exception as e:
        cost = round(time.time() - t, 1)
        print(f"失败, 耗时 {cost} 秒")
        print("错误:", repr(e)[:300])
        results[name] = f"失败: {repr(e)[:80]}"


# ---------- A. 全市场列表 + 流通市值 ----------
run("A1 东方财富-全市场行情(含流通市值)", lambda: ak.stock_zh_a_spot_em())
run("A2 新浪-全市场行情", lambda: ak.stock_zh_a_spot())
run("A3 东方财富-单只个股信息(含流通市值)",
    lambda: ak.stock_individual_info_em(symbol="000001"))

# ---------- B. 日K线 ----------
run("B1 东方财富-日K 不复权",
    lambda: ak.stock_zh_a_hist(symbol="000001", period="daily",
                               start_date=start, end_date=end, adjust=""))
run("B2 东方财富-日K 前复权",
    lambda: ak.stock_zh_a_hist(symbol="000001", period="daily",
                               start_date=start, end_date=end, adjust="qfq"))
run("B3 新浪-日K 前复权",
    lambda: ak.stock_zh_a_daily(symbol="sz000001", start_date=start,
                                end_date=end, adjust="qfq"))
run("B4 腾讯-日K 前复权",
    lambda: ak.stock_zh_a_hist_tx(symbol="sz000001", start_date=start,
                                  end_date=end, adjust="qfq"))

# ---------- C. 连续拉取速度测试(检测是否被限速) ----------
codes = ["000001", "000002", "000063", "000100", "000333",
         "000651", "000725", "000858", "002415", "600036"]


def batch():
    ok = 0
    for c in codes:
        ak.stock_zh_a_hist(symbol=c, period="daily",
                           start_date=start, end_date=end, adjust="qfq")
        ok += 1
    return pd.DataFrame({"ok": range(ok)})


run("C1 东方财富-连续拉10只股票日K", batch)

# ---------- 汇总 ----------
print("\n" + "#" * 60)
print("汇总结果")
for k, v in results.items():
    print(f"{k}: {v}")
