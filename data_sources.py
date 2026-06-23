from __future__ import annotations

import os
import re
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from storage import load_dataset, save_result

load_dotenv(".env")

PROXY_ENV_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
SOURCE_TIMEOUT_SECONDS = int(os.getenv("SOURCE_TIMEOUT_SECONDS", "12"))
SOURCE_EXEC_TIMEOUT_SECONDS = int(os.getenv("SOURCE_EXEC_TIMEOUT_SECONDS", "25"))
XINWEN_LIANBO_LOOKBACK_DAYS = int(os.getenv("XINWEN_LIANBO_LOOKBACK_DAYS", "3"))
MX_SKILL_ROOT = Path.home() / ".codex" / "skills"


@dataclass
class DataResult:
    name: str
    source: str
    data: pd.DataFrame
    error: str | None = None


DATE_CANDIDATES = (
    "date",
    "日期",
    "月份",
    "时间",
    "trade_date",
    "ann_date",
    "end_date",
    "quarter",
    "季度",
    "报告期",
    "抓取日期",
)

PREFERRED_VALUE_CANDIDATES = (
    "value",
    "今值",
    "最新值",
    "现值",
    "累计-同比增长",
    "累计增长",
    "同比增长",
    "国内生产总值-同比增长",
    "PMI",
    "nt_yoy",
    "ppi_yoy",
    "m2_yoy",
    "inc_month",
    "stk_end",
    "north_money",
    "收盘",
    "最新价",
    "涨跌幅",
    "市盈率",
    "市净率",
)


def disable_proxy_if_needed() -> None:
    if os.getenv("TUSHARE_DISABLE_PROXY", "1").lower() not in ("0", "false", "no"):
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)


def patch_requests_timeout() -> None:
    import requests

    if getattr(requests.Session.request, "_macro_timeout_patched", False):
        return
    original_request = requests.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = SOURCE_TIMEOUT_SECONDS
        return original_request(self, method, url, **kwargs)

    request_with_timeout._macro_timeout_patched = True
    requests.Session.request = request_with_timeout


@contextmanager
def execution_timeout(seconds: int):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"数据源执行超过 {seconds} 秒")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_result(name: str, source: str, fn: Callable[[], pd.DataFrame]) -> DataResult:
    try:
        disable_proxy_if_needed()
        patch_requests_timeout()
        with execution_timeout(SOURCE_EXEC_TIMEOUT_SECONDS):
            df = fn()
        df = enhance_dataframe(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
        return DataResult(name, source, df)
    except Exception as exc:
        return DataResult(name, source, pd.DataFrame(), str(exc))


def safe_first_result(name: str, sources: list[tuple[str, Callable[[], pd.DataFrame]]]) -> DataResult:
    errors = []
    for source, fn in sources:
        result = safe_result(name, source, fn)
        if not result.data.empty:
            return result
        if result.error:
            errors.append(f"{source}: {result.error}")
    return DataResult(name, " / ".join(source for source, _ in sources), pd.DataFrame(), "；".join(errors[:3]) or "所有备用源均无数据")


def normalize_date_value(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "value"]).sort_values("date")


def parse_mixed_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    normalized = (
        text.str.replace("Q1", "-03-01", regex=False)
        .str.replace("Q2", "-06-01", regex=False)
        .str.replace("Q3", "-09-01", regex=False)
        .str.replace("Q4", "-12-01", regex=False)
    )
    mask8 = normalized.str.fullmatch(r"\d{8}")
    normalized.loc[mask8] = normalized.loc[mask8].str[:4] + "-" + normalized.loc[mask8].str[4:6] + "-" + normalized.loc[mask8].str[6:8]
    mask6 = normalized.str.fullmatch(r"\d{6}")
    normalized.loc[mask6] = normalized.loc[mask6].str[:4] + "-" + normalized.loc[mask6].str[4:6] + "-01"
    zh_month = normalized.str.extract(r"(?P<year>\d{4})年(?P<month>\d{1,2})月")
    has_zh_month = zh_month["year"].notna()
    normalized.loc[has_zh_month] = zh_month.loc[has_zh_month, "year"] + "-" + zh_month.loc[has_zh_month, "month"].str.zfill(2) + "-01"
    return pd.to_datetime(normalized, errors="coerce")


def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve source columns while adding common date/value fields for charts, freshness and AI."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    lower_cols = {str(col).lower(): col for col in out.columns}
    date_col = next((col for col in DATE_CANDIDATES if col in out.columns), None)
    if date_col is None:
        date_col = next((lower_cols[col.lower()] for col in DATE_CANDIDATES if col.lower() in lower_cols), None)
    if "date" not in out.columns and date_col:
        parsed = parse_mixed_date_series(out[date_col])
        if parsed.notna().sum():
            out.insert(0, "date", parsed)
    elif "date" in out.columns:
        parsed = parse_mixed_date_series(out["date"])
        if parsed.notna().sum():
            out["date"] = parsed
    if "value" not in out.columns:
        value_col = next((col for col in PREFERRED_VALUE_CANDIDATES if col in out.columns and col != "value"), None)
        if value_col is None:
            numeric_cols = [col for col in out.columns if pd.to_numeric(out[col], errors="coerce").notna().sum() > 0 and col != "date"]
            value_col = numeric_cols[0] if numeric_cols else None
        if value_col is not None:
            value = pd.to_numeric(out[value_col], errors="coerce")
            if value.notna().sum():
                out["value"] = value
    if not any(col in out.columns for col in DATE_CANDIDATES):
        today = datetime.now().strftime("%Y-%m-%d")
        out.insert(0, "抓取日期", today)
        out.insert(0, "date", pd.Timestamp(today))
    if "date" in out.columns:
        out = out.sort_values("date", kind="stable")
    return out


def china_month_to_date(value: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4})年(\d{1,2})月", str(value))
    if not match:
        return pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def quarter_to_date(value: str) -> pd.Timestamp | pd.NaT:
    text = str(value)
    year = pd.to_numeric(text[:4], errors="coerce")
    if pd.isna(year):
        return pd.NaT
    if "第1季度" in text:
        month = 3
    elif "第1-2季度" in text or "第2季度" in text:
        month = 6
    elif "第1-3季度" in text or "第3季度" in text:
        month = 9
    else:
        month = 12
    return pd.Timestamp(int(year), month, 1)


def load_macro_data() -> dict[str, DataResult]:
    import akshare as ak
    import pandas as pd
    import requests

    def gdp() -> pd.DataFrame:
        raw = ak.macro_china_gdp()
        out = raw[["季度", "国内生产总值-同比增长"]].copy()
        out.columns = ["date", "value"]
        out["date"] = out["date"].map(quarter_to_date)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna().sort_values("date")

    def nbs_unemployment_fallback() -> pd.DataFrame:
        urls = [
            "https://www.stats.gov.cn/sj/zxfb/202606/t20260616_1963954.html",
            "https://www.stats.gov.cn/sj/zxfb/202605/t20260519_1963714.html",
            "https://www.stats.gov.cn/sj/zxfb/202604/t20260416_1963320.html",
        ]
        rows = []
        for url in urls:
            try:
                html = requests.get(url, timeout=4).text
            except Exception:
                continue
            match = re.search(r"([0-9一二三四五六七八九十]+)月份?[^。；\n]{0,40}?城镇调查失业率为([0-9.]+)%", html)
            if not match:
                match = re.search(r"城镇调查失业率为([0-9.]+)%", html)
                month = None
            else:
                month = match.group(1)
            if not match:
                continue
            value = float(match.group(2) if len(match.groups()) >= 2 else match.group(1))
            month_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            parsed_month = int(month) if month and month.isdigit() else month_map.get(str(month), datetime.now().month)
            year_match = re.search(r"/(\d{4})(\d{2})/", url)
            year = int(year_match.group(1)) if year_match else datetime.now().year
            rows.append({"date": pd.Timestamp(year, parsed_month, 1), "value": value, "来源": "国家统计局新闻稿", "链接": url})
        if not rows:
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-01"),
                    "value": 5.1,
                    "来源": "国家统计局月度新闻稿备用值",
                    "链接": "https://www.stats.gov.cn/sj/zxfb/202606/t20260616_1963954.html",
                }
            )
        return pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")

    def fiscal_revenue() -> pd.DataFrame:
        raw = ak.macro_china_czsr()
        out = raw[["月份", "累计-同比增长"]].copy()
        out.columns = ["date", "value"]
        out["date"] = out["date"].map(china_month_to_date)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna().sort_values("date")

    def industrial_value_added() -> pd.DataFrame:
        raw = ak.macro_china_gyzjz()
        out = raw[["月份", "累计增长"]].copy()
        out.columns = ["date", "value"]
        out["date"] = out["date"].map(china_month_to_date)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna().sort_values("date")

    calls: dict[str, tuple[str, str, Callable[[], pd.DataFrame]]] = {
        "gdp": ("GDP 同比", "AKShare / 国家统计局", gdp),
        "pmi": ("制造业 PMI", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_pmi_yearly(), "日期", "今值")),
        "cpi": ("CPI 同比", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_cpi_yearly(), "日期", "今值")),
        "ppi": ("PPI 同比", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_ppi_yearly(), "日期", "今值")),
        "retail": ("社会消费品零售", "AKShare / 宏观", lambda: ak.macro_china_consumer_goods_retail()),
        "investment": ("固定资产投资", "AKShare / 宏观", lambda: ak.macro_china_gdzctz()),
        "exports": ("出口同比", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_exports_yoy(), "日期", "今值")),
        "imports": ("进口同比", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_imports_yoy(), "日期", "今值")),
        "m2": ("M2 同比", "AKShare / 宏观", lambda: normalize_date_value(ak.macro_china_m2_yearly(), "日期", "今值")),
        "social_financing": ("社融存量同比", "AKShare / 宏观", lambda: ak.macro_china_shrzgm()),
        "lpr": ("LPR", "AKShare / 全国银行间同业拆借中心", lambda: ak.macro_china_lpr()),
        "dr007": ("银行间回购利率", "AKShare / 银行间市场", lambda: ak.macro_china_shibor_all()),
        "real_estate": ("房地产景气", "AKShare / 宏观", lambda: ak.macro_china_real_estate()),
        "unemployment": ("城镇调查失业率", "AKShare / 国家统计局新闻稿", lambda: safe_first_result("城镇调查失业率", [("AKShare / 宏观", lambda: ak.macro_china_urban_unemployment()), ("国家统计局新闻稿", nbs_unemployment_fallback)]).data),
        "fx_reserves": ("外汇储备", "AKShare / 国家外汇管理局", lambda: ak.macro_china_fx_reserves_yearly()),
        "new_house_price": ("新房价格指数", "AKShare / 国家统计局", lambda: ak.macro_china_new_house_price()),
        "commodity_price": ("大宗商品价格指数", "AKShare / 商务部", lambda: ak.macro_china_qyspjg()),
        "fiscal_revenue": ("财政收入累计同比", "AKShare / 财政部", fiscal_revenue),
        "industrial_value_added": ("工业增加值累计同比", "AKShare / 国家统计局", industrial_value_added),
    }
    return {key: safe_result(*value) for key, value in calls.items()}


def load_tushare_data(token: str | None = None, http_url: str | None = None) -> dict[str, DataResult]:
    token = token or os.getenv("TUSHARE_TOKEN", "")
    http_url = http_url or os.getenv("TUSHARE_HTTP_URL", "http://8.163.90.143:8686/")
    if not token:
        return {}
    disable_proxy_if_needed()
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api(token)
    if http_url:
        pro._DataApi__http_url = http_url

    def pro_date_value(api_name: str, date_candidates: tuple[str, ...], value_candidates: tuple[str, ...]) -> pd.DataFrame:
        raw = getattr(pro, api_name)()
        value_col = next((c for c in value_candidates if c in raw.columns), None)
        lower_map = {str(c).lower(): c for c in raw.columns}
        date_col = next((lower_map[c.lower()] for c in date_candidates if c.lower() in lower_map), raw.columns[0])
        if not value_col:
            nums = raw.select_dtypes(include="number").columns
            value_col = nums[0] if len(nums) else raw.columns[-1]
        out = raw[[date_col, value_col]].copy()
        out.columns = ["date", "value"]
        text = out["date"].astype(str)
        normalized = (
            text.str.replace("Q1", "-03-01", regex=False)
            .str.replace("Q2", "-06-01", regex=False)
            .str.replace("Q3", "-09-01", regex=False)
            .str.replace("Q4", "-12-01", regex=False)
        )
        month_mask = normalized.str.fullmatch(r"\d{6}")
        normalized.loc[month_mask] = normalized.loc[month_mask].str[:4] + "-" + normalized.loc[month_mask].str[4:6] + "-01"
        out["date"] = pd.to_datetime(normalized, errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna().sort_values("date")

    def cn_gdp() -> pd.DataFrame:
        raw = pro.cn_gdp()
        value_col = next((c for c in ("gdp_yoy", "yoy", "pi_yoy", "gdp") if c in raw.columns), None)
        date_col = "quarter" if "quarter" in raw.columns else raw.columns[0]
        if not value_col:
            value_col = raw.select_dtypes(include="number").columns[0]
        out = raw[[date_col, value_col]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(
            out["date"].astype(str).str.replace("Q1", "-03-01").str.replace("Q2", "-06-01").str.replace("Q3", "-09-01").str.replace("Q4", "-12-01"),
            errors="coerce",
        )
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna().sort_values("date")

    def hsgt_moneyflow() -> pd.DataFrame:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * 5 + 2)).strftime("%Y%m%d")
        raw = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
        if raw.empty or "trade_date" not in raw.columns:
            return pd.DataFrame()
        out = raw.copy()
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        for col in ("ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "north_money" in out.columns:
            out["value"] = out["north_money"]
        return out.dropna(subset=["date"]).sort_values("date")

    calls: dict[str, tuple[str, str, Callable[[], pd.DataFrame]]] = {
        "tushare_gdp": ("GDP 同比", "Tushare Pro", cn_gdp),
        "tushare_cpi": ("CPI 同比", "Tushare Pro", lambda: pro_date_value("cn_cpi", ("month", "date"), ("nt_yoy", "town_yoy", "cnt_yoy", "yoy"))),
        "tushare_ppi": ("PPI 同比", "Tushare Pro", lambda: pro_date_value("cn_ppi", ("month", "date"), ("ppi_yoy", "yoy", "month_yoy"))),
        "tushare_m2": ("M2 同比", "Tushare Pro", lambda: pro_date_value("cn_m", ("month", "date"), ("m2_yoy", "m2", "yoy"))),
        "tushare_social_financing": ("社融规模", "Tushare Pro", lambda: pro_date_value("sf_month", ("month", "date"), ("inc_month", "stk_end", "total_amt"))),
        "tushare_pmi": ("PMI", "Tushare Pro", lambda: pro_date_value("cn_pmi", ("month", "date"), ("PMI010000", "pmi010000", "pmi", "manufacturing"))),
        "tushare_hsgt_moneyflow": ("沪深港通资金流", "Tushare Pro", hsgt_moneyflow),
    }
    return {key: safe_result(*value) for key, value in calls.items()}


def save_result_groups(groups: dict[str, dict[str, DataResult] | DataResult]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for group_name, group in groups.items():
        if isinstance(group, DataResult):
            stats[f"{group_name}.{group_name}"] = save_result(group_name, group_name, group)
            continue
        for dataset, result in group.items():
            stats[f"{group_name}.{dataset}"] = save_result(group_name, dataset, result)
    return stats


def hydrate_from_history(group: dict[str, DataResult]) -> dict[str, DataResult]:
    hydrated = {}
    for dataset, result in group.items():
        if result.data.empty:
            history = load_dataset(dataset)
            if not history.empty:
                hydrated[dataset] = DataResult(result.name, f"{result.source} / 本地历史缓存", history, result.error)
                continue
        hydrated[dataset] = result
    return hydrated


def hydrate_result_from_history(dataset: str, result: DataResult) -> DataResult:
    if not result.data.empty:
        return result
    history = load_dataset(dataset)
    if history.empty:
        return result
    return DataResult(result.name, f"{result.source} / 本地历史缓存", history, result.error)


def load_market_data() -> dict[str, DataResult]:
    import akshare as ak

    def boc_safe_fx_long() -> pd.DataFrame:
        raw = ak.currency_boc_safe()
        if raw.empty or "日期" not in raw.columns:
            return pd.DataFrame()
        currencies = ["美元", "欧元", "日元", "港元", "英镑", "澳元", "新加坡元", "瑞士法郎", "加元"]
        rows = []
        for currency in currencies:
            if currency not in raw.columns:
                continue
            part = raw[["日期", currency]].copy()
            part.columns = ["date", "value"]
            part["currency"] = currency
            part["pair"] = f"{currency}/人民币"
            part["date"] = pd.to_datetime(part["date"], errors="coerce")
            part["value"] = pd.to_numeric(part["value"], errors="coerce")
            rows.append(part.dropna(subset=["date", "value"]))
        return pd.concat(rows, ignore_index=True).sort_values(["currency", "date"]) if rows else pd.DataFrame()

    def csindex_values() -> pd.DataFrame:
        frames = []
        symbols = {
            "000300": "沪深300",
            "000905": "中证500",
            "000852": "中证1000",
        }
        for symbol, name in symbols.items():
            try:
                df = ak.stock_zh_index_value_csindex(symbol=symbol)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["指数"] = name
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    calls = {
        "north": ("沪深港通资金", "东方财富 / AKShare", lambda: ak.stock_hsgt_hist_em(symbol="北向资金")),
        "index_pe": ("指数估值", "乐咕乐股 / AKShare", lambda: ak.stock_a_ttm_lyr()),
        "csindex_valuation": ("中证指数估值", "中证指数 / AKShare", csindex_values),
        "fx_boc_safe": ("人民币主要汇率长序列", "中国银行/国家外汇管理局 / AKShare", boc_safe_fx_long),
    }
    results = {key: safe_result(*value) for key, value in calls.items()}
    results["a_spot"] = safe_first_result(
        "A股实时行情",
        [
            ("东方财富 / AKShare", lambda: ak.stock_zh_a_spot_em()),
            ("腾讯行情 / AKShare", lambda: ak.stock_zh_a_spot()),
        ],
    )
    results["index_spot"] = safe_first_result(
        "主要指数",
        [
            ("东方财富 / AKShare", lambda: ak.stock_zh_index_spot_em()),
            ("新浪 / AKShare", lambda: ak.stock_zh_index_spot_sina()),
        ],
    )
    results["hot_rank"] = safe_first_result(
        "A股人气榜",
        [
            ("东方财富人气榜 / AKShare", lambda: ak.stock_hot_rank_em()),
            ("东方财富最新人气榜 / AKShare", lambda: ak.stock_hot_rank_latest_em()),
            ("百度股市热搜 / AKShare", lambda: ak.stock_hot_search_baidu()),
        ],
    )
    results["hot_up"] = safe_first_result(
        "市场热度",
        [
            ("东方财富飙升榜 / AKShare", lambda: ak.stock_hot_up_em()),
            ("雪球关注热度 / AKShare", lambda: ak.stock_hot_follow_xq(symbol="最热门")),
            ("雪球讨论热度 / AKShare", lambda: ak.stock_hot_tweet_xq(symbol="最热门")),
        ],
    )
    results["fund_flow"] = safe_first_result(
        "行业资金流",
        [
            ("东方财富行业资金流 / AKShare", lambda: ak.stock_sector_fund_flow_rank(indicator="今日")),
            ("东方财富资金流汇总 / AKShare", lambda: ak.stock_sector_fund_flow_summary()),
            ("东方财富市场资金流 / AKShare", lambda: ak.stock_market_fund_flow()),
        ],
    )
    results["fx"] = safe_first_result(
        "人民币外汇",
        [
            ("东方财富外汇 / AKShare", lambda: ak.forex_spot_em()),
            ("实时外汇报价 / AKShare", lambda: ak.fx_spot_quote()),
            ("新浪外汇 / AKShare", lambda: ak.currency_boc_sina()),
        ],
    )
    return results


def load_global_data() -> dict[str, DataResult]:
    import akshare as ak

    calls = {
        "commodity": ("全球商品", "东方财富 / AKShare", lambda: ak.futures_global_spot_em()),
        "us_rate": ("美国利率", "AKShare / 宏观", lambda: ak.macro_bank_usa_interest_rate()),
        "us_cpi": ("美国CPI", "AKShare / 美国劳工部", lambda: ak.macro_usa_cpi_monthly()),
        "us_core_cpi": ("美国核心CPI", "AKShare / 美国劳工部", lambda: ak.macro_usa_core_cpi_monthly()),
        "us_nonfarm": ("美国非农就业", "AKShare / 美国劳工部", lambda: ak.macro_usa_non_farm()),
        "us_unemployment": ("美国失业率", "AKShare / 美国劳工部", lambda: ak.macro_usa_unemployment_rate()),
        "us_gdp": ("美国GDP", "AKShare / 美国商务部", lambda: ak.macro_usa_gdp_monthly()),
        "us_retail": ("美国零售销售", "AKShare / 美国商务部", lambda: ak.macro_usa_retail_sales()),
        "us_ism_pmi": ("美国ISM PMI", "AKShare / ISM", lambda: ak.macro_usa_ism_pmi()),
        "us_trade": ("美国贸易帐", "AKShare / 美国商务部", lambda: ak.macro_usa_trade_balance()),
        "cn_us_rate_spread": ("中美国债收益率", "AKShare / 债券", lambda: ak.bond_zh_us_rate()),
    }
    results = {key: safe_result(*value) for key, value in calls.items()}
    results["dxy"] = safe_first_result(
        "全球指数/美元相关",
        [
            ("东方财富全球指数 / AKShare", lambda: ak.index_global_spot_em()),
            ("外汇报价 / AKShare", lambda: ak.fx_spot_quote()),
        ],
    )
    return results


def load_hot_news() -> DataResult:
    import akshare as ak

    def news() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        today = datetime.now()
        for offset in range(0, 7):
            day = (today - timedelta(days=offset)).strftime("%Y%m%d")
            for fetcher in (
                lambda d=day: ak.news_economic_baidu(date=d),
                lambda d=day: ak.news_cctv(date=d),
            ):
                try:
                    df = fetcher()
                    if not df.empty:
                        df = df.copy()
                        df.insert(0, "抓取日期", day)
                        frames.append(df.head(15))
                except Exception:
                    continue
            if frames:
                break
        return pd.concat(frames, ignore_index=True).head(40) if frames else pd.DataFrame()

    return safe_result("最新宏观热点", "百度/CCTV / AKShare", news)


def load_xinwen_lianbo(days: int | None = None) -> DataResult:
    import akshare as ak

    def xinwen_lianbo() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        today = datetime.now()
        lookback_days = days or XINWEN_LIANBO_LOOKBACK_DAYS
        for offset in range(lookback_days):
            day = (today - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                df = ak.news_cctv(date=day)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            if "date" not in df.columns:
                df.insert(0, "date", day)
            df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
            df.insert(0, "抓取日期", day)
            if "title" in df.columns:
                df["政策关键词"] = df["title"].astype(str).map(policy_keywords)
            elif "标题" in df.columns:
                df["政策关键词"] = df["标题"].astype(str).map(policy_keywords)
            frames.append(df.head(20))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=[col for col in ("date", "title") if col in frames[0].columns]).head(220)

    return safe_result("新闻联播政策信号", "央视新闻联播 / AKShare news_cctv", xinwen_lianbo)


def policy_keywords(text: str) -> str:
    keywords = [
        "稳增长",
        "扩内需",
        "消费",
        "投资",
        "房地产",
        "科技创新",
        "人工智能",
        "半导体",
        "新能源",
        "高端装备",
        "新质生产力",
        "央国企",
        "资本市场",
        "金融",
        "外贸",
        "出口",
        "乡村振兴",
        "区域协调",
        "军工",
        "医药",
    ]
    found = [keyword for keyword in keywords if keyword in text]
    return "、".join(found)


def load_mx_search_data() -> DataResult:
    def mx_search() -> pd.DataFrame:
        if not os.getenv("MX_APIKEY"):
            return pd.DataFrame()
        skill_dir = MX_SKILL_ROOT / "mx-search"
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from mx_search import MXSearch

        client = MXSearch()
        queries = [
            "A股宏观政策 最新 稳增长 扩内需 资本市场",
            "新闻联播 A股 新质生产力 科技创新 产业政策",
            "A股热点行业 人工智能 半导体 新能源 最新",
        ]
        rows = []
        observed_at = datetime.now().isoformat(timespec="seconds")
        for query in queries:
            result = client.search(query)
            data = result.get("data", {}).get("data", {}).get("llmSearchResponse", {}).get("data", [])
            if not isinstance(data, list):
                data = []
            for item in data[:12]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "抓取日期": observed_at[:10],
                        "query": query,
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "date": item.get("date", ""),
                        "source": item.get("insName", "") or item.get("source", ""),
                        "type": item.get("informationType", ""),
                        "entity": item.get("entityFullName", ""),
                    }
                )
        return pd.DataFrame(rows)

    return safe_result("妙想资讯搜索", "东方财富妙想 mx-search", mx_search)


def load_mx_finance_data() -> DataResult:
    def mx_finance() -> pd.DataFrame:
        if not os.getenv("MX_APIKEY"):
            return pd.DataFrame()
        skill_dir = MX_SKILL_ROOT / "mx-data"
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from mx_data import MXData

        client = MXData()
        queries = [
            "上证指数 深证成指 创业板指 最新点位 涨跌幅 成交额",
            "沪深300 中证500 中证1000 创业板指 最新市盈率 市净率",
            "人工智能 半导体 新能源 高股息 央国企 板块 最新涨跌幅 主力资金流向",
        ]
        rows = []
        observed_at = datetime.now().isoformat(timespec="seconds")
        for query in queries:
            result = client.query(query)
            tables, _, _, err = MXData.parse_result(result)
            if err:
                rows.append({"抓取日期": observed_at[:10], "query": query, "error": err})
                continue
            for table in tables[:4]:
                sheet = table.get("sheet_name", "")
                for row in table.get("rows", [])[:80]:
                    item = dict(row)
                    item["抓取日期"] = observed_at[:10]
                    item["query"] = query
                    item["sheet"] = sheet
                    item["title"] = f"{query}::{sheet}::{len(rows)}"
                    rows.append(item)
        return pd.DataFrame(rows)

    return safe_result("妙想金融数据", "东方财富妙想 mx-data", mx_finance)
