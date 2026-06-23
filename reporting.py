from __future__ import annotations

import hashlib
import os
from datetime import timedelta
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from data_sources import DataResult
from llm import build_prompt, build_quality_review_prompt, build_role_prompt, build_synthesis_prompt, call_llm_with_meta
from scoring import classify_environment, latest_number, trend
from storage import latest_ai_reports, latest_refresh, load_dataset, load_dataset_recent, save_ai_chunk, save_ai_report

load_dotenv(".env")

HISTORY_DAYS = 365 * 5 + 2

FREQUENCY_RULES = {
    "daily": 10,
    "weekly": 21,
    "monthly": 75,
    "quarterly": 150,
    "event": 365,
    "snapshot": 30,
}

DATASET_FREQUENCY = {
    "gdp": "quarterly", "tushare_gdp": "quarterly", "us_gdp": "quarterly",
    "pmi": "monthly", "cpi": "monthly", "ppi": "monthly", "retail": "monthly",
    "investment": "monthly", "exports": "monthly", "imports": "monthly", "m2": "monthly",
    "social_financing": "monthly", "tushare_cpi": "monthly", "tushare_ppi": "monthly",
    "tushare_m2": "monthly", "tushare_social_financing": "monthly", "tushare_pmi": "monthly",
    "real_estate": "monthly", "unemployment": "monthly", "fx_reserves": "monthly",
    "new_house_price": "monthly", "fiscal_revenue": "monthly", "industrial_value_added": "monthly",
    "us_cpi": "monthly", "us_core_cpi": "monthly", "us_nonfarm": "monthly",
    "us_unemployment": "monthly", "us_retail": "monthly", "us_ism_pmi": "monthly",
    "us_trade": "monthly", "lpr": "daily", "dr007": "daily", "tushare_hsgt_moneyflow": "daily",
    "north": "daily", "index_spot": "snapshot", "a_spot": "snapshot", "hot_rank": "snapshot",
    "hot_up": "snapshot", "fund_flow": "daily", "index_pe": "daily", "csindex_valuation": "daily",
    "fx": "snapshot", "fx_boc_safe": "daily", "dxy": "snapshot", "commodity": "snapshot",
    "us_rate": "event", "cn_us_rate_spread": "daily", "commodity_price": "weekly",
    "au_report": "event", "news": "daily", "xinwen_lianbo": "daily", "mx_search": "daily",
    "mx_finance": "daily",
}


DATASETS = {
    "macro": {
        "gdp": ("GDP 同比", "AKShare / 国家统计局"),
        "pmi": ("制造业 PMI", "AKShare / 宏观"),
        "cpi": ("CPI 同比", "AKShare / 宏观"),
        "ppi": ("PPI 同比", "AKShare / 宏观"),
        "retail": ("社会消费品零售", "AKShare / 宏观"),
        "investment": ("固定资产投资", "AKShare / 宏观"),
        "exports": ("出口同比", "AKShare / 宏观"),
        "imports": ("进口同比", "AKShare / 宏观"),
        "m2": ("M2 同比", "AKShare / 宏观"),
        "social_financing": ("社融存量同比", "AKShare / 宏观"),
        "lpr": ("LPR", "AKShare / 全国银行间同业拆借中心"),
        "dr007": ("银行间回购利率", "AKShare / 银行间市场"),
        "real_estate": ("房地产景气", "AKShare / 宏观"),
        "unemployment": ("城镇调查失业率", "AKShare / 国家统计局"),
        "fx_reserves": ("外汇储备", "AKShare / 国家外汇管理局"),
        "new_house_price": ("新房价格指数", "AKShare / 国家统计局"),
        "commodity_price": ("大宗商品价格指数", "AKShare / 商务部"),
        "au_report": ("黄金储备/黄金报告", "AKShare / 宏观"),
        "fiscal_revenue": ("财政收入累计同比", "AKShare / 财政部"),
        "industrial_value_added": ("工业增加值累计同比", "AKShare / 国家统计局"),
    },
    "tushare": {
        "tushare_gdp": ("GDP 同比", "Tushare Pro"),
        "tushare_cpi": ("CPI 同比", "Tushare Pro"),
        "tushare_ppi": ("PPI 同比", "Tushare Pro"),
        "tushare_m2": ("M2 同比", "Tushare Pro"),
        "tushare_social_financing": ("社融规模", "Tushare Pro"),
        "tushare_pmi": ("PMI", "Tushare Pro"),
        "tushare_hsgt_moneyflow": ("沪深港通资金流", "Tushare Pro"),
    },
    "market": {
        "a_spot": ("A股实时行情", "东方财富 / AKShare"),
        "index_spot": ("主要指数", "东方财富 / AKShare"),
        "hot_rank": ("A股人气榜", "东方财富 / AKShare"),
        "hot_up": ("东方财富飙升榜", "东方财富 / AKShare"),
        "fund_flow": ("行业资金流", "东方财富 / AKShare"),
        "north": ("沪深港通资金", "东方财富 / AKShare"),
        "index_pe": ("指数估值", "乐咕乐股 / AKShare"),
        "csindex_valuation": ("中证指数估值", "中证指数 / AKShare"),
        "fx": ("人民币外汇", "东方财富 / AKShare"),
        "fx_boc_safe": ("人民币主要汇率长序列", "中国银行/国家外汇管理局 / AKShare"),
    },
    "global": {
        "dxy": ("美元指数", "东方财富 / AKShare"),
        "commodity": ("全球商品", "东方财富 / AKShare"),
        "us_rate": ("美国利率", "AKShare / 宏观"),
        "us_cpi": ("美国CPI", "AKShare / 美国劳工部"),
        "us_core_cpi": ("美国核心CPI", "AKShare / 美国劳工部"),
        "us_nonfarm": ("美国非农就业", "AKShare / 美国劳工部"),
        "us_unemployment": ("美国失业率", "AKShare / 美国劳工部"),
        "us_gdp": ("美国GDP", "AKShare / 美国商务部"),
        "us_retail": ("美国零售销售", "AKShare / 美国商务部"),
        "us_ism_pmi": ("美国ISM PMI", "AKShare / ISM"),
        "us_trade": ("美国贸易帐", "AKShare / 美国商务部"),
        "cn_us_rate_spread": ("中美国债收益率", "AKShare / 债券"),
    },
}


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _latest_ts(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    for col in ("date", "日期", "trade_date", "TRADE_DATE", "ann_date", "end_date", "quarter", "报告期", "抓取日期", "observed_at", "updated_at"):
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col].astype(str), errors="coerce")
        if parsed.notna().sum():
            return parsed.max().strftime("%Y-%m-%d")
    return ""


def data_freshness_for_ai(groups: dict[str, dict[str, DataResult]], refresh_log: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    today = pd.Timestamp.now().normalize()
    latest_errors = {}
    if not refresh_log.empty and {"dataset", "status", "error", "created_at"}.issubset(refresh_log.columns):
        for _, row in refresh_log.iterrows():
            latest_errors.setdefault(row["dataset"], row.to_dict())
    for group_name, group in groups.items():
        for dataset, result in group.items():
            latest = _latest_ts(result.data)
            lag_days = None
            if latest:
                lag_days = int((today - pd.to_datetime(latest)).days)
            freq = DATASET_FREQUENCY.get(dataset, "monthly")
            max_lag = FREQUENCY_RULES.get(freq, 75)
            row = latest_errors.get(dataset, {})
            status = "正常"
            if result.data.empty:
                status = "无数据"
            elif lag_days is None and freq != "snapshot":
                status = "无日期字段"
            elif lag_days is not None and lag_days > max_lag * 2:
                status = "明显滞后"
            elif lag_days is not None and lag_days > max_lag:
                status = "偏旧"
            if str(row.get("status", "")) == "error" and not result.data.empty:
                status = "使用历史缓存/源异常"
            rows.append(
                {
                    "group": group_name,
                    "dataset": dataset,
                    "name": result.name,
                    "latest_date": latest or "无日期字段",
                    "lag_days": lag_days,
                    "freshness_status": status,
                    "latest_refresh_status": row.get("status", ""),
                    "latest_error": str(row.get("error") or "")[:160],
                }
            )
    return rows


def load_local_bundle():
    bundle = {}
    for group_name, datasets in DATASETS.items():
        group = {}
        for dataset, (name, source) in datasets.items():
            df = load_dataset_recent(dataset, days=HISTORY_DAYS)
            if df.empty:
                df = load_dataset(dataset, limit=800)
            group[dataset] = DataResult(name, source + " / 本地库", df)
        bundle[group_name] = group
    news_df = load_dataset_recent("news", days=HISTORY_DAYS)
    if news_df.empty:
        news_df = load_dataset("news", limit=200)
    bundle["news"] = DataResult("最新宏观热点", "百度/CCTV / 本地历史库", news_df)
    xwlb_df = load_dataset_recent("xinwen_lianbo", days=HISTORY_DAYS, limit=2500)
    if xwlb_df.empty:
        xwlb_df = load_dataset("xinwen_lianbo", limit=500)
    bundle["xinwen_lianbo"] = DataResult("新闻联播政策信号", "央视新闻联播 / AKShare / 本地历史库", xwlb_df)
    mx_search_df = load_dataset_recent("mx_search", days=HISTORY_DAYS, limit=1200)
    if mx_search_df.empty:
        mx_search_df = load_dataset("mx_search", limit=500)
    mx_finance_df = load_dataset_recent("mx_finance", days=HISTORY_DAYS, limit=1200)
    if mx_finance_df.empty:
        mx_finance_df = load_dataset("mx_finance", limit=500)
    bundle["mx"] = {
        "mx_search": DataResult("妙想资讯搜索", "东方财富妙想 mx-search / 本地历史库", mx_search_df),
        "mx_finance": DataResult("妙想金融数据", "东方财富妙想 mx-data / 本地历史库", mx_finance_df),
    }
    bundle["refresh_log"] = latest_refresh()
    return bundle


def first_latest(groups: list[dict], keys: list[str]) -> float | None:
    for group in groups:
        for key in keys:
            result = group.get(key)
            if result is not None and not result.data.empty:
                value = latest_number(result.data)
                if value is not None:
                    return value
    return None


def build_scores(bundle):
    macro = bundle["macro"]
    tushare_data = bundle["tushare"]
    market = bundle["market"]
    social_financing = None
    for result in (tushare_data.get("tushare_social_financing"), macro.get("social_financing")):
        if result and not result.data.empty:
            social_financing = latest_number(result.data)
            if social_financing is not None:
                break
    market_pe = None
    if market.get("index_pe") and not market.get("index_pe").data.empty:
        nums = market.get("index_pe").data.select_dtypes(include="number")
        if not nums.empty:
            market_pe = float(nums.iloc[-1].dropna().iloc[0])
    metrics = {
        "gdp": first_latest([tushare_data, macro], ["tushare_gdp", "gdp"]),
        "pmi": first_latest([tushare_data, macro], ["tushare_pmi", "pmi"]),
        "cpi": first_latest([tushare_data, macro], ["tushare_cpi", "cpi"]),
        "ppi": first_latest([tushare_data, macro], ["tushare_ppi", "ppi"]),
        "m2": first_latest([tushare_data, macro], ["tushare_m2", "m2"]),
        "social_financing": social_financing,
        "market_pe": market_pe,
        "retail_trend": trend(macro.get("retail").data) if macro.get("retail") else 0,
        "industrial_profit_trend": trend(macro.get("ppi").data) if macro.get("ppi") else 0,
        "rate_trend": trend(macro.get("dr007").data) if macro.get("dr007") else 0,
        "lpr_1y_down": trend(macro.get("lpr").data) < 0 if macro.get("lpr") else False,
        "rrr_down": False,
    }
    return classify_environment(metrics), metrics


def dataset_summary(result, full_series: bool = False, recent_rows: int = 8, series_tail: int = 18, max_text_chars: int = 260) -> dict:
    if not result or result.data.empty:
        return {"name": getattr(result, "name", ""), "source": getattr(result, "source", ""), "rows": 0, "status": "missing"}
    df = result.data.copy()
    summary = {"name": result.name, "source": result.source, "rows": len(df), "columns": list(df.columns)[:12]}
    if {"currency", "date", "value"}.issubset(df.columns):
        rows = []
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        for currency, part in data.dropna(subset=["date", "value"]).sort_values("date").groupby("currency"):
            tail = part.tail(6)
            rows.append(
                {
                    "currency": currency,
                    "pair": str(part["pair"].iloc[-1]) if "pair" in part.columns and not part.empty else "",
                    "latest_date": part["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "latest_value": float(part["value"].iloc[-1]),
                    "trend_3_period": float(part["value"].tail(3).iloc[-1] - part["value"].tail(3).iloc[0]) if len(part.tail(3)) >= 2 else 0,
                    "recent_points": [
                        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                        for _, row in tail.iterrows()
                    ],
                }
            )
        summary["currency_series"] = rows
        return summary
    if {"date", "value"}.issubset(df.columns):
        series = df[["date", "value"]].copy()
        series["date"] = pd.to_datetime(series["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        series = series.dropna()
        summary.update(
            {
                "latest_date": series["date"].iloc[-1] if not series.empty else "",
                "latest_value": latest_number(df),
                "trend_3_period": trend(df),
                "series_points": compact_series_points(series) if full_series else series.tail(series_tail).to_dict(orient="records"),
            }
        )
    else:
        summary["recent_rows"] = compact_records(df.tail(recent_rows).iloc[:, :8], max_text_chars=max_text_chars)
    return summary


def compact_series_points(series: pd.DataFrame, max_points: int = 28) -> list[dict[str, Any]]:
    if series.empty:
        return []
    data = series.copy()
    data["_year"] = pd.to_datetime(data["date"], errors="coerce").dt.year
    yearly = data.groupby("_year", dropna=True).tail(2).drop(columns=["_year"], errors="ignore")
    latest = data.tail(10).drop(columns=["_year"], errors="ignore")
    out = pd.concat([yearly, latest], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    return out.tail(max_points).to_dict(orient="records")


def compact_records(df: pd.DataFrame, max_text_chars: int = 260) -> list[dict[str, Any]]:
    rows = []
    for record in df.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            text = str(value)
            clean[key] = text[:max_text_chars] + "..." if len(text) > max_text_chars else value
        rows.append(clean)
    return rows


def policy_signal_summary(result, limit: int = 80) -> dict[str, Any]:
    summary = dataset_summary(result, recent_rows=min(limit, 30), max_text_chars=220)
    if not result or result.data.empty:
        return summary
    df = result.data.copy()
    text_cols = [col for col in ("title", "标题", "content", "内容", "政策关键词") if col in df.columns]
    keyword_counts: dict[str, int] = {}
    for _, row in df.tail(240).iterrows():
        text = " ".join(str(row.get(col, "")) for col in text_cols)
        for keyword in [
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
        ]:
            if keyword in text:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    summary["policy_keyword_counts"] = dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:15])
    return summary


ROLE_DATA_MAP = [
    {"dataset": "gdp/pmi/cpi/ppi/retail/investment/exports/imports/real_estate/unemployment/industrial_value_added", "used_by": "宏观经济学家", "purpose": "判断经济周期、过去五年阶段切换和弱复苏/下行风险"},
    {"dataset": "fiscal_revenue/lpr/xinwen_lianbo/hot_news", "used_by": "政策研究员", "purpose": "判断财政、货币、产业政策力度和政策主线"},
    {"dataset": "m2/social_financing/lpr/dr007/tushare_hsgt_moneyflow/north/fx/fx_boc_safe", "used_by": "流动性与利率交易员", "purpose": "判断银行间流动性、股市可用资金、外资风险偏好和汇率压力"},
    {"dataset": "ppi/index_pe/csindex_valuation/index_spot/a_spot", "used_by": "权益估值分析师", "purpose": "判断盈利修复、宽基估值、安全边际和估值陷阱"},
    {"dataset": "dxy/us_rate/us_cpi/us_core_cpi/us_nonfarm/us_unemployment/us_ism_pmi/cn_us_rate_spread/fx_boc_safe/commodity", "used_by": "全球宏观策略师", "purpose": "判断美债、美元、人民币、中美利差和全球商品对A股估值的约束"},
    {"dataset": "fund_flow/hot_rank/hot_up/mx_search/mx_finance/xinwen_lianbo/hot_news", "used_by": "行业配置策略师", "purpose": "核对政策、资金、热点和产业证据，筛出4-5个行业方向"},
    {"dataset": "real_estate/new_house_price/tushare_hsgt_moneyflow/north/fx/fx_boc_safe/fx_reserves/us_rate/cn_us_rate_spread/csindex_valuation", "used_by": "投资风险管理官", "purpose": "形成主要风险数据库：地产、外资、人民币、美债、估值、盈利和政策落地风险"},
]


def risk_evidence_database(snapshot_groups: dict[str, dict[str, DataResult]]) -> dict[str, Any]:
    macro = snapshot_groups["macro"]
    tushare_data = snapshot_groups["tushare"]
    market = snapshot_groups["market"]
    global_data = snapshot_groups["global"]
    items = {
        "地产拖累": [
            ("real_estate", macro.get("real_estate")),
            ("new_house_price", macro.get("new_house_price")),
        ],
        "外资与北向资金": [
            ("tushare_hsgt_moneyflow", tushare_data.get("tushare_hsgt_moneyflow")),
            ("north", market.get("north")),
        ],
        "人民币与外汇": [
            ("fx_boc_safe", market.get("fx_boc_safe")),
            ("fx", market.get("fx")),
            ("fx_reserves", macro.get("fx_reserves")),
        ],
        "海外利率与中美利差": [
            ("us_rate", global_data.get("us_rate")),
            ("cn_us_rate_spread", global_data.get("cn_us_rate_spread")),
            ("dxy", global_data.get("dxy")),
        ],
        "估值与盈利下修": [
            ("csindex_valuation", market.get("csindex_valuation")),
            ("index_pe", market.get("index_pe")),
            ("ppi", macro.get("ppi")),
            ("industrial_value_added", macro.get("industrial_value_added")),
        ],
    }
    return {
        risk_name: [
            {"dataset": dataset, **dataset_summary(result, recent_rows=10, series_tail=18, max_text_chars=180)}
            for dataset, result in datasets
            if result is not None
        ]
        for risk_name, datasets in items.items()
    }


def build_ai_snapshot(bundle, scores):
    macro, tushare_data, market, global_data, news, refresh_log = (
        bundle["macro"],
        bundle["tushare"],
        bundle["market"],
        bundle["global"],
        bundle["news"],
        bundle["refresh_log"],
    )
    xinwen_lianbo = bundle.get("xinwen_lianbo", DataResult("新闻联播政策信号", "央视新闻联播 / 本地历史库", pd.DataFrame()))
    mx = bundle.get("mx", {})
    coverage = []
    for group_name, group in {"macro": macro, "tushare": tushare_data, "market": market, "global": global_data}.items():
        for dataset, result in group.items():
            coverage.append({"group": group_name, "dataset": dataset, "name": result.name, "source": result.source, "rows": len(result.data)})
    coverage.append({"group": "news", "dataset": "news", "name": news.name, "source": news.source, "rows": len(news.data)})
    coverage.append({"group": "policy_news", "dataset": "xinwen_lianbo", "name": xinwen_lianbo.name, "source": xinwen_lianbo.source, "rows": len(xinwen_lianbo.data)})
    for dataset, result in mx.items():
        coverage.append({"group": "eastmoney_mx", "dataset": dataset, "name": result.name, "source": result.source, "rows": len(result.data)})
    important_macro = [
        "gdp", "pmi", "cpi", "ppi", "retail", "investment", "exports", "imports",
        "m2", "social_financing", "lpr", "dr007", "real_estate", "fx_reserves",
        "new_house_price", "fiscal_revenue", "industrial_value_added",
    ]
    important_tushare = ["tushare_gdp", "tushare_cpi", "tushare_ppi", "tushare_m2", "tushare_social_financing", "tushare_pmi", "tushare_hsgt_moneyflow"]
    important_market = ["index_spot", "index_pe", "csindex_valuation", "north", "fund_flow", "hot_rank", "hot_up", "fx", "fx_boc_safe"]
    important_global = ["dxy", "commodity", "us_rate", "us_cpi", "us_core_cpi", "us_nonfarm", "us_unemployment", "us_ism_pmi", "cn_us_rate_spread"]
    freshness_groups = {
        "macro": macro,
        "tushare": tushare_data,
        "market": market,
        "global": global_data,
        "eastmoney_mx": mx,
        "news": {"news": news, "xinwen_lianbo": xinwen_lianbo},
    }
    return {
        "overview_scores": {key: value.__dict__ for key, value in scores.items()},
        "coverage": coverage,
        "analysis_logic": "政策->新闻联播政策信号->经济周期->流动性->盈利->估值->外部环境->行业映射；宏观五年序列用于判断过去/现在/未来，新闻联播用于识别政策重心和产业主线。",
        "macro": {key: dataset_summary(macro[key], full_series=True) for key in important_macro if key in macro},
        "tushare": {key: dataset_summary(tushare_data[key], full_series=True) for key in important_tushare if key in tushare_data},
        "market": {key: dataset_summary(market[key], recent_rows=8, series_tail=16, max_text_chars=180) for key in important_market if key in market},
        "global": {key: dataset_summary(global_data[key], recent_rows=8, series_tail=16, max_text_chars=180) for key in important_global if key in global_data},
        "xinwen_lianbo_policy_signal": policy_signal_summary(xinwen_lianbo, limit=80),
        "eastmoney_mx": {key: dataset_summary(result, recent_rows=24, series_tail=16, max_text_chars=220) for key, result in mx.items()},
        "hot_news": dataset_summary(news, recent_rows=18, max_text_chars=220),
        "risk_evidence_database": risk_evidence_database({"macro": macro, "tushare": tushare_data, "market": market, "global": global_data}),
        "role_data_map": ROLE_DATA_MAP,
        "data_freshness_audit": data_freshness_for_ai(freshness_groups, refresh_log),
        "refresh_log_latest": refresh_log.head(12).to_dict(orient="records") if not refresh_log.empty else [],
    }


ROLE_SPECS = [
    (
        "政策研究员",
        """负责判断政策环境、新闻联播政策信号、政策重心变化和产业政策主线。
分析方法：
1. 把新闻联播高频主题与宏观政策目标对应起来，区分“口号式表述”和“真正可交易的政策方向”。
2. 判断政策力度是偏大规模、中性还是偏紧，必须说明财政、货币、产业政策各自证据。
3. 批判性检查：如果政策表述积极但数据端没有改善，要指出“政策预期强、现实验证弱”；如果政策热点很多但缺少资金/产业验证，要降级判断。
4. 输出必须包含：政策结论、最强政策证据、最弱证据/反证、产业主线线索、未来需观察的政策信号。
只读 overview_scores、analysis_logic、macro 中的 fiscal_revenue、xinwen_lianbo_policy_signal、hot_news。""",
        ["overview_scores", "analysis_logic", "macro", "xinwen_lianbo_policy_signal", "hot_news", "refresh_log_latest"],
    ),
    (
        "宏观经济学家",
        """负责判断经济周期、过去五年阶段切换、当前处于复苏/弱复苏/下行/过热哪一类。
分析方法：
1. 用GDP、PMI、社零、投资、进出口、CPI/PPI、地产、失业率建立经济周期判断，不允许只看单一指标。
2. 对比AKShare和Tushare同类指标，发现冲突要说明采用哪个口径。
3. 批判性检查：名义数据改善但价格/PPI偏弱时，要警惕“量修复、价承压”；PMI改善但地产/消费弱时，要警惕“结构性弱复苏”。
4. 输出必须包含：当前周期结论、过去五年阶段复盘、主要支撑证据、主要拖累项、可能错判的条件。
只读宏观与Tushare中的GDP、PMI、CPI、PPI、社零、投资、进出口、地产、失业率、工业增加值等。""",
        ["overview_scores", "macro", "tushare"],
    ),
    (
        "流动性与利率交易员",
        """负责判断流动性、资金价格、信用扩张和市场资金是否支持A股估值扩张。
分析方法：
1. 同时看M2、社融、LPR、DR007/资金利率、北向资金、人民币汇率，区分“银行间宽松”和“股市可用资金宽松”。
2. 判断信用扩张是总量改善还是结构不佳；利率下行是否真正传导到风险资产。
3. 批判性检查：M2高但社融弱，说明资金空转风险；利率低但北向/基金/成交不配合，说明风险偏好没有恢复。
4. 输出必须包含：流动性结论、利率/信用证据、资金风险偏好证据、反证信号、对估值扩张的影响。
必须明确使用 Tushare 沪深港通资金流、东方财富北向资金、人民币主要汇率长序列；如果东方财富北向净买额为空，以 Tushare moneyflow_hsgt 为主并说明原因。
只读M2、社融、LPR、DR007、北向资金、汇率、市场资金相关数据。""",
        ["overview_scores", "macro", "tushare", "market"],
    ),
    (
        "权益估值分析师",
        """负责判断企业盈利、PPI、宽基指数估值、安全边际和盈利修复斜率。
分析方法：
1. 用PPI、工业品价格、需求指标、宽基指数表现和估值数据判断盈利处于改善、见底还是承压。
2. 估值判断必须区分沪深300、中证500、中证1000、创业板/科创成长方向，不可笼统说“A股低估”。
3. 批判性检查：低估不等于马上上涨，必须检查盈利是否下修、风险溢价是否被外部利率压制；盈利改善如果只来自价格反弹，也要说明持续性风险。
4. 输出必须包含：盈利结论、估值结论、最有安全边际的方向、估值陷阱、后续验证指标。
只读企业盈利、PPI、指数估值、中证指数估值、宽基行情相关数据。""",
        ["overview_scores", "macro", "tushare", "market"],
    ),
    (
        "全球宏观策略师",
        """负责判断外部环境对A股的约束和边际变化。
分析方法：
1. 看美债收益率、美元指数、人民币汇率、中美利差、美国通胀/就业/PMI、全球商品，判断外资和风险偏好的外部压力。
2. 区分“外部风险缓和”和“外部风险彻底解除”，不要把单一指标改善解读为趋势逆转。
3. 批判性检查：若美债/美元仍强，A股成长股估值扩张要打折；若人民币承压，外资和高估值板块要谨慎。
4. 输出必须包含：外部环境结论、对A股估值/外资/行业的影响、最大外部风险、触发改善的信号。
只读全球宏观、汇率、商品、外部利率相关数据。""",
        ["overview_scores", "global", "market", "risk_evidence_database"],
    ),
    (
        "行业配置策略师",
        """负责判断上一轮市场主线、当前主线、未来主线和重点行业配置。
分析方法：
1. 必须把宏观组合映射到行业：弱复苏/宽流动性/政策支撑/外部约束分别对应哪些方向。
2. 用东方财富妙想资讯、行业资金流、人气榜、新闻联播政策信号交叉验证热点是否只是短炒，还是有政策和资金共同确认。
3. 批判性检查：热点行业如果只有新闻热度、没有资金/业绩/政策连续性，要降级；拥挤交易要提示回撤风险。
4. 输出必须包含：上一轮主线、当前主线、当前主线可维持多久、未来可能主线、切换验证信号。
5. 必须给出“行业配置推荐表”的紧凑结构，只保留 4-5 个最值得关注的行业方向，字段包括：行业方向、关注级别、当前主线/未来主线、为什么关注、对应证据、触发条件、风险。关注级别只能用高/中高/中/观察。
6. 每个行业必须写清楚“为什么是它”：至少包含一条宏观/政策证据和一条资金/热点/产业证据；证据不足的方向只能列为“观察”。
只读市场、东方财富妙想、热点新闻、新闻联播政策信号。""",
        ["overview_scores", "market", "eastmoney_mx", "hot_news", "xinwen_lianbo_policy_signal"],
    ),
    (
        "投资风险管理官",
        """负责从反面审视整份宏观判断，列出风险、反证和仓位/风险偏好调整条件。
分析方法：
1. 不负责唱多，专门找判断漏洞：经济修复不及预期、地产拖累、政策落地弱、外资流出、人民币贬值、美债上行、盈利下修。
2. 对每个风险给出触发指标，而不是只写风险名称。
3. 批判性检查：如果多项核心证据相互矛盾，要明确建议降低结论置信度；如果数据缺口影响判断，要列为风险。
4. 输出必须包含：前三大风险、反证信号、风险升级条件、需要降低风险偏好的情形、仍可维持判断的条件。
必须逐项读取主要风险数据库中的外资与北向资金、人民币与外汇、海外利率与中美利差、地产拖累、估值与盈利下修；不得只写泛泛风险。
只读宏观、外部环境、市场、新闻联播政策信号和主要风险数据库。""",
        ["overview_scores", "macro", "tushare", "global", "market", "xinwen_lianbo_policy_signal", "risk_evidence_database"],
    ),
]


def slice_snapshot_for_role(snapshot: dict[str, Any], keys: list[str], role_name: str) -> dict[str, Any]:
    data = {key: snapshot.get(key) for key in keys if key in snapshot}
    data["role_name"] = role_name
    data["coverage_summary"] = [
        item for item in snapshot.get("coverage", []) if item.get("rows", 0) > 0
    ][:30]
    return compact_role_data(data)


def compact_role_data(value: Any, max_records: int = 10, max_text: int = 180) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "coverage_summary" and isinstance(item, list):
                out[key] = item[:30]
            elif key in ("recent_rows", "series_points") and isinstance(item, list):
                out[key] = compact_role_data(item[:max_records], max_records=max_records, max_text=max_text)
            else:
                out[key] = compact_role_data(item, max_records=max_records, max_text=max_text)
        return out
    if isinstance(value, list):
        return [compact_role_data(item, max_records=max_records, max_text=max_text) for item in value[:max_records]]
    if isinstance(value, str) and len(value) > max_text:
        return value[:max_text] + "..."
    return value


def _summary_line(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict):
        return "暂无"
    bits = []
    if summary.get("latest_date"):
        bits.append(f"最新日期{summary.get('latest_date')}")
    if summary.get("latest_value") is not None:
        bits.append(f"最新值{summary.get('latest_value')}")
    if summary.get("trend_3_period") is not None:
        bits.append(f"近3期变化{summary.get('trend_3_period')}")
    if not bits and summary.get("rows") is not None:
        bits.append(f"行数{summary.get('rows')}")
    return "，".join(bits) if bits else "暂无"


def build_local_role_fallback(role_name: str, role_data: dict[str, Any], snapshot: dict[str, Any]) -> str:
    macro = snapshot.get("macro", {})
    tushare = snapshot.get("tushare", {})
    market = snapshot.get("market", {})
    global_data = snapshot.get("global", {})
    news = snapshot.get("hot_news", {})
    policy = snapshot.get("xinwen_lianbo_policy_signal", {})
    risk = snapshot.get("risk_evidence_database", {})
    if role_name == "政策研究员":
        return "\n".join([
            "结论：政策环境偏大规模托底，但仍处在稳增长与扩内需的持续确认期。",
            f"证据：新闻联播关键词{policy.get('policy_keyword_counts', {})}；财政收入{_summary_line(macro.get('fiscal_revenue', {}))}；热点新闻{news.get('latest_date', '暂无')}",
            "反证：若政策信号偏强但社融、地产和外需没有同步改善，说明预期强、落地弱。",
            "后续观察：新闻联播、财政、货币、产业政策是否继续同向。",
        ])
    if role_name == "宏观经济学家":
        return "\n".join([
            "结论：经济周期仍是弱复苏/修复观察期。",
            f"证据：GDP{_summary_line(macro.get('gdp', {}))}；PMI{_summary_line(tushare.get('tushare_pmi') or macro.get('pmi', {}))}；社零{_summary_line(macro.get('retail', {}))}；地产{_summary_line(macro.get('real_estate', {}))}。",
            "反证：PMI、CPI/PPI和地产仍偏弱，说明量价修复不均衡。",
            "后续观察：PMI、社零、投资、出口、地产和失业率是否连续改善。",
        ])
    if role_name == "流动性与利率交易员":
        return "\n".join([
            "结论：流动性偏宽松，但股市可用资金是否转强还需验证。",
            f"证据：M2{_summary_line(tushare.get('tushare_m2') or macro.get('m2', {}))}；社融{_summary_line(tushare.get('tushare_social_financing') or macro.get('social_financing', {}))}；北向资金{_summary_line(tushare.get('tushare_hsgt_moneyflow') or market.get('north', {}))}；汇率{_summary_line(market.get('fx_boc_safe', {}))}。",
            "反证：M2高但社融弱，或利率低但北向/成交不配合，说明资金传导不顺。",
            "后续观察：LPR、DR007、北向、汇率和基金发行。",
        ])
    if role_name == "权益估值分析师":
        return "\n".join([
            "结论：盈利仍在修复中，低估不等于立刻上涨。",
            f"证据：PPI{_summary_line(tushare.get('tushare_ppi') or macro.get('ppi', {}))}；工业增加值{_summary_line(macro.get('industrial_value_added', {}))}；指数估值{_summary_line(market.get('index_pe', {}))}。",
            "反证：盈利若继续下修，即使估值低也可能是价值陷阱。",
            "后续观察：PPI、工业利润、宽基估值和风险溢价。",
        ])
    if role_name == "全球宏观策略师":
        return "\n".join([
            "结论：外部环境仍有约束，美债、美元和中美利差会压制部分估值扩张。",
            f"证据：美元指数{_summary_line(global_data.get('dxy', {}))}；中美国债收益率{_summary_line(global_data.get('cn_us_rate_spread', {}))}；美国利率{_summary_line(global_data.get('us_rate', {}))}。",
            "反证：若美元和美债继续偏强，成长风格和高估值板块弹性会受限。",
            "后续观察：美联储、美元、美债、人民币和全球商品。",
        ])
    if role_name == "行业配置策略师":
        return "\n".join([
            "结论：当前主线仍偏科技成长与政策主题，高股息/央国企是防御锚。",
            "建议：优先关注 AI/算力/半导体、高股息/央国企、出口链、消费/创新药、券商/金融科技。",
            "证据：新闻联播、妙想资讯、资金流、人气榜和宏观数据仍支持结构性配置。",
            f"覆盖提示：{role_data.get('coverage_summary', [])[:2]}",
        ])
    if role_name == "投资风险管理官":
        return "\n".join([
            "结论：主要风险集中在地产、外资/北向、人民币与外汇、美债/中美利差和盈利下修。",
            f"证据：地产{risk.get('地产拖累', [])[:1]}；北向{risk.get('外资与北向资金', [])[:1]}；外汇{risk.get('人民币与外汇', [])[:1]}；海外利率{risk.get('海外利率与中美利差', [])[:1]}。",
            "反证：若风险数据库中的数据没有被角色引用，说明报告必须补证据而不是空写风险。",
            "后续观察：外储、汇率、北向资金、地产销售、估值和盈利修复。",
        ])
    return "数据不足/本角色未成功返回。"


def generate_role_outputs(snapshot: dict[str, Any], model: str, report_run_id: str) -> list[dict[str, Any]]:
    outputs = []
    for role_name, instruction, keys in ROLE_SPECS:
        role_data = slice_snapshot_for_role(snapshot, keys, role_name)
        prompt = build_role_prompt(role_name, instruction, role_data)
        result = call_llm_with_meta(prompt, model=model)
        status = "ok" if result.get("ok") else "error"
        content = result.get("content", "")
        chunk_id = save_ai_chunk(
            report_run_id=report_run_id,
            chunk_name=role_name,
            model=model,
            prompt_hash=prompt_hash(prompt),
            content=content,
            status=status,
            error=None if result.get("ok") else result.get("content", content),
            coverage=role_data.get("coverage_summary"),
            usage=result.get("usage"),
        )
        outputs.append(
            {
                "chunk_id": chunk_id,
                "role": role_name,
                "status": status,
                "content": content,
                "usage": result.get("usage", {}),
            }
        )
    return outputs


def summarize_role_outputs_for_synthesis(role_outputs: list[dict[str, Any]], max_chars: int = 900) -> list[dict[str, Any]]:
    compact = []
    for item in role_outputs:
        content = item.get("content", "") or ""
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        compact.append(
            {
                "role": item.get("role"),
                "status": item.get("status"),
                "content": content,
            }
        )
    return compact


def content_for_role(role_outputs: list[dict[str, Any]], role: str) -> str:
    for item in role_outputs:
        if item.get("role") == role and item.get("status") == "ok":
            return str(item.get("content") or "").strip()
    return "数据不足/本角色未成功返回。"


def llm_failure_report(role_outputs: list[dict[str, Any]], stage: str) -> str:
    failed = [
        {
            "role": item.get("role"),
            "status": item.get("status"),
            "error": str(item.get("content") or "")[:260],
        }
        for item in role_outputs
        if item.get("status") != "ok"
    ]
    lines = [
        "ChatGPT 未成功完成分析，本次不生成正式宏观结论。",
        "",
        f"失败阶段：{stage}",
        "",
        "原因：至少一个 ChatGPT 分析角色或最终汇总角色没有返回成功响应。根据你的要求，系统不会使用本地规则兜底冒充 ChatGPT 分析结果。",
        "",
        "需要处理：",
        "1. 检查 OPENAI_BASE_URL 是否为可用的 /v1 API 地址。",
        "2. 如果服务器返回 Cloudflare 403，说明当前云服务器 IP 被接口站点拦截，需要更换接口出口、让接口服务商放行服务器 IP，或改用不会拦截服务器请求的 OpenAI 兼容接口。",
        "3. 处理后重新点击“手动触发一次AI分析”。报告状态必须为 ok，才代表 ChatGPT 真实分析成功。",
        "",
        "失败明细：",
    ]
    for item in failed[:12]:
        lines.append(f"- {item['role']}：{item['status']}；{item['error']}")
    return "\n".join(lines)


def build_local_fallback_report(role_outputs: list[dict[str, Any]], synthesis_content: str = "") -> str:
    policy = content_for_role(role_outputs, "政策研究员")
    macro = content_for_role(role_outputs, "宏观经济学家")
    liquidity = content_for_role(role_outputs, "流动性与利率交易员")
    valuation = content_for_role(role_outputs, "权益估值分析师")
    global_macro = content_for_role(role_outputs, "全球宏观策略师")
    industry = content_for_role(role_outputs, "行业配置策略师")
    risk = content_for_role(role_outputs, "投资风险管理官")
    synthesis_note = synthesis_content if synthesis_content and not synthesis_content.startswith("LLM接口请求失败") else "首席策略官长请求未获得成功响应，以下为基于七个职业角色成功输出的本地汇总版。"
    return f"""
0. 经济周期曲线定位
- 当前定位：萧条末期 / 复苏初期。
- 为什么：政策和流动性具备托底特征，但经济、地产、盈利和资金风险偏好仍需要后续数据验证；这更像底部向右侧修复的观察期，而不是繁荣期。
- 资产含义：优先采用结构性配置，关注政策科技、高股息/央国企和基本面验证方向，同时控制估值和外部利率风险。

1. 当前A股宏观环境总评
- 综合判断：当前报告采用“多角色分块成功输出 + 本地质控汇总”的口径。{synthesis_note}
- 政策环境：以政策研究员结论为主，重点观察稳增长、扩内需、防风险、科技创新和新闻联播政策信号。
- 经济周期：以宏观经济学家结论为主，重点观察GDP、PMI、社零、投资、出口、CPI/PPI、地产和失业率。
- 流动性环境：以流动性与利率交易员结论为主，区分银行间宽松和股市风险偏好。
- 企业盈利与估值：以权益估值分析师结论为主，避免把低估简单等同于马上上涨。

2. 政策环境
{policy}

3. 经济基础
{macro}

4. 流动性环境
{liquidity}

5. 企业盈利
{valuation}

6. 估值水平
{valuation}

7. 外部环境
{global_macro}

8. A股市场主线推演
{industry}

9. 行业配置方向
{industry}

| 行业方向 | 关注级别 | 当前主线/未来主线 | 为什么关注 | 对应证据 | 触发条件 | 风险 |
|---|---|---|---|---|---|---|
| AI/算力/半导体 | 高 | 当前主线/未来主线 | 政策科技创新和风险偏好改善时弹性较高 | 新闻联播科技信号、东方财富热点、成长板块资金 | 成交放大、资金流入、业绩预期上修 | 拥挤交易、外部限制、估值回撤 |
| 高股息/央国企 | 中高 | 弱复苏防御主线 | 弱复苏和盈利不确定阶段更重视现金流与分红 | 政策防风险、估值安全边际、资金避险 | 利率下行、风险偏好不强 | 风险偏好快速上升时相对跑输 |
| 出口链：家电/汽车/机械/电子 | 中 | 未来验证方向 | 若外需改善和人民币压力缓和，出口链可能修复 | 出口、汇率、海外PMI和行业景气 | 出口数据改善、订单回升 | 贸易摩擦、海外需求回落 |
| 消费/创新药 | 观察 | 未来主线候选 | 取决于社零、居民收入预期和流动性改善 | 社零、政策扩内需、医药产业政策 | 消费数据连续改善、政策催化 | 需求恢复慢、业绩兑现不足 |
| 券商/金融科技 | 观察 | 风险偏好修复弹性方向 | 如果成交和风险偏好回升，金融β可能放大指数弹性 | 成交额、市场风险偏好、政策改革预期 | 指数放量突破、两融和成交持续回暖 | 市场缩量、政策预期落空 |

10. 主要风险
{risk}

11. 最终结论
- 当前A股宏观环境总评：更接近“政策支撑下的弱复苏观察期”，需要用后续经济、盈利和资金数据验证。
- 当前主线：科技成长与政策主题仍是重要观察方向，高股息/央国企承担防御和稳定器角色。
- 下一阶段关注方向：AI/算力/半导体、高股息/央国企、出口链、消费与创新药的验证信号。
- 需要继续观察的数据：新闻联播政策连续性、Tushare宏观指标、东方财富妙想热点与资金、PMI、社零、PPI、社融、汇率和美债。
""".strip()


def generate_ai_report(title: str = "A股宏观环境自动报告") -> dict:
    bundle = load_local_bundle()
    scores, _ = build_scores(bundle)
    snapshot = build_ai_snapshot(bundle, scores)
    report_run_id = datetime_now_id()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    role_outputs = generate_role_outputs(snapshot, model=model, report_run_id=report_run_id)
    failed_roles = [item for item in role_outputs if item.get("status") != "ok"]
    if failed_roles:
        content = llm_failure_report(role_outputs, "职业角色分块分析")
        rid = save_ai_report(
            title=title,
            model=model,
            prompt_hash=prompt_hash(content),
            content=content,
            status="error",
            error=content,
            coverage=snapshot.get("coverage"),
            usage={},
        )
        return {
            "ok": False,
            "content": content,
            "report_id": rid,
            "role_outputs": role_outputs,
            "report_run_id": report_run_id,
            "model": model,
            "model_base_url": os.getenv("OPENAI_BASE_URL", ""),
        }
    synthesis_inputs = summarize_role_outputs_for_synthesis(role_outputs, max_chars=550)
    synthesis_prompt = build_synthesis_prompt(synthesis_inputs, snapshot.get("coverage", [])[:25])
    synthesis_result = call_llm_with_meta(synthesis_prompt, model=model)
    synthesis_status = "ok" if synthesis_result.get("ok") else "error"
    synthesis_content = synthesis_result.get("content", "")
    synthesis_chunk_id = save_ai_chunk(
        report_run_id=report_run_id,
        chunk_name="首席策略官",
        model=model,
        prompt_hash=prompt_hash(synthesis_prompt),
        content=synthesis_content,
        status=synthesis_status,
        error=None if synthesis_result.get("ok") else synthesis_content,
        coverage=snapshot.get("coverage"),
        usage=synthesis_result.get("usage"),
    )
    role_outputs.append(
        {
            "chunk_id": synthesis_chunk_id,
            "role": "首席策略官",
            "status": synthesis_status,
            "content": synthesis_content,
            "usage": synthesis_result.get("usage", {}),
        }
    )
    if not synthesis_result.get("ok"):
        content = llm_failure_report(role_outputs, "首席策略官汇总")
        rid = save_ai_report(
            title=title,
            model=model,
            prompt_hash=prompt_hash(synthesis_prompt),
            content=content,
            status="error",
            error=content,
            coverage=snapshot.get("coverage"),
            usage=synthesis_result.get("usage"),
        )
        synthesis_result["report_id"] = rid
        synthesis_result["role_outputs"] = role_outputs
        synthesis_result["report_run_id"] = report_run_id
        synthesis_result["model_base_url"] = os.getenv("OPENAI_BASE_URL", "")
        return synthesis_result

    review_inputs = summarize_role_outputs_for_synthesis(role_outputs, max_chars=450)
    review_draft = synthesis_content
    prompt = build_quality_review_prompt(review_draft[:6000], review_inputs, snapshot.get("coverage", [])[:25])
    result = call_llm_with_meta(prompt, model=model)
    status = "ok" if result.get("ok") else "error"
    content = result.get("content", "")
    if not result.get("ok"):
        content = llm_failure_report(role_outputs, "投研质控总监最终输出")
        status = "error"
    review_chunk_id = save_ai_chunk(
        report_run_id=report_run_id,
        chunk_name="投研质控总监",
        model=model,
        prompt_hash=prompt_hash(prompt),
        content=content,
        status=status,
        error=None if result.get("ok") else content,
        coverage=snapshot.get("coverage"),
        usage=result.get("usage"),
    )
    role_outputs.append(
        {
            "chunk_id": review_chunk_id,
            "role": "投研质控总监",
            "status": status,
            "content": content,
            "usage": result.get("usage", {}),
        }
    )
    rid = save_ai_report(
        title=title,
        model=model,
        prompt_hash=prompt_hash(prompt),
        content=content,
        status=status,
        error=None if result.get("ok") else content,
        coverage=snapshot.get("coverage"),
        usage=result.get("usage"),
    )
    result["report_id"] = rid
    result["role_outputs"] = role_outputs
    result["report_run_id"] = report_run_id
    return result


def datetime_now_id() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d%H%M%S")
