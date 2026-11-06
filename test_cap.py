import time
import traceback
import akshare as ak
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

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
        print(df.head(3).to_string())
        results[name] = f"成功 ({cost}s, {len(df)}行) 列={list(df.columns)[:12]}"
    except Exception as e:
        cost = round(time.time() - t, 1)
        print(f"失败, 耗时 {cost} 秒")
        print("错误:", repr(e)[:300])
        results[name] = f"失败: {repr(e)[:100]}"
    time.sleep(5)


# 1. 同花顺-股票列表/行情
run("D1 同花顺-A股行情 stock_zh_a_spot_ths(若存在)",
    lambda: ak.stock_zh_a_spot_ths())

# 2. 雪球-全市场行情
run("D2 雪球-全部A股 stock_zh_a_spot_xq(若存在)",
    lambda: ak.stock_zh_a_spot_xq())

# 3. 沪深京A股列表(只有代码名称)
run("D3 交易所-代码名称 stock_info_a_code_name",
    lambda: ak.stock_info_a_code_name())

# 4. 深交所A股列表(含流通股本)
run("D4 深交所-A股列表 stock_info_sz_name_code",
    lambda: ak.stock_info_sz_name_code(symbol="A股列表"))

# 5. 上交所股票列表(含流通股本)
run("D5 上交所-主板A股 stock_info_sh_name_code",
    lambda: ak.stock_info_sh_name_code(symbol="主板A股"))

# 6. 百度股市通-单只估值(含总市值)
run("D6 百度-单只总市值 stock_zh_valuation_baidu",
    lambda: ak.stock_zh_valuation_baidu(symbol="000001", indicator="总市值", period="近一年"))

# 7. 腾讯日K是否带成交额/换手率(辅助算市值)
run("D7 腾讯-日K列 stock_zh_a_hist_tx",
    lambda: ak.stock_zh_a_hist_tx(symbol="sz000001", start_date="20261001",
                                  end_date="20261106", adjust="qfq"))

print("\n" + "#" * 60)
print("汇总结果")
for k, v in results.items():
    print(f"{k}: {v}")
