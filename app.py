from __future__ import annotations

import os
import subprocess
import sys
import hashlib
import json
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data_sources import (
    DataResult,
)
from llm import build_prompt, call_llm_with_meta
from reporting import generate_ai_report
from scoring import classify_environment, latest_date, latest_number, trend
from storage import latest_ai_chunks, latest_ai_reports, latest_refresh, load_ai_chunks, load_ai_report, load_dataset, load_dataset_recent, save_ai_report

load_dotenv(".env")
st.set_page_config(page_title="A股宏观环境分析模板", page_icon="📊", layout="wide")

REFRESH_SECONDS = 30 * 60
AI_REFRESH_SECONDS = 12 * 60 * 60
HISTORY_DAYS = 365 * 5 + 2
DATA_UPDATE_LOCK = Path("data/update.lock")
AI_REPORT_LOCK = Path("data/ai_report.lock")


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def cached_llm(prompt: str, api_key: str, base_url: str, model: str, request_id: int):
    return call_llm_with_meta(prompt, api_key=api_key, base_url=base_url, model=model)


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
        "unemployment": ("城镇调查失业率", "AKShare / 宏观"),
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


@st.cache_data(ttl=60, show_spinner=False)
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


def trigger_background_update() -> None:
    subprocess.Popen(
        [sys.executable, "updater.py", "--once"],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def lock_is_active(path: Path, max_age_seconds: int) -> bool:
    if not path.exists():
        return False
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
    except OSError:
        return False
    if age > max_age_seconds:
        path.unlink(missing_ok=True)
        return False
    return True


def touch_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def latest_log_time(refresh_log: pd.DataFrame) -> datetime | None:
    if refresh_log.empty:
        return None
    latest = pd.to_datetime(refresh_log["created_at"].iloc[0], errors="coerce")
    if pd.isna(latest):
        return None
    return latest.to_pydatetime()


def maybe_auto_update_data(refresh_log: pd.DataFrame) -> None:
    latest = latest_log_time(refresh_log)
    due = latest is None or datetime.now() - latest >= timedelta(seconds=REFRESH_SECONDS)
    if due and not lock_is_active(DATA_UPDATE_LOCK, 20 * 60):
        touch_lock(DATA_UPDATE_LOCK)
        trigger_background_update()


def trigger_background_ai_report() -> None:
    subprocess.Popen(
        [sys.executable, "report_scheduler.py", "--once"],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def maybe_auto_ai_report() -> None:
    reports = latest_ai_reports(1)
    if reports.empty:
        return
    latest = pd.to_datetime(reports["created_at"].iloc[0], errors="coerce")
    if pd.isna(latest):
        return
    due = datetime.now() - latest.to_pydatetime() >= timedelta(seconds=AI_REFRESH_SECONDS)
    if due and not lock_is_active(AI_REPORT_LOCK, 2 * 60 * 60):
        touch_lock(AI_REPORT_LOCK)
        trigger_background_ai_report()


st.markdown(
    """
    <style>
      :root { --bg:#07111f; --panel:#0d1a2b; --panel2:#0f2237; --line:#1c3f5f; --text:#d9e7ff; --muted:#7f9bb8; --cyan:#26d9ff; --green:#2df5a8; --amber:#ffd166; }
      .stApp { background:
        radial-gradient(circle at 20% 0%, rgba(38,217,255,.18), transparent 28rem),
        radial-gradient(circle at 80% 10%, rgba(45,245,168,.11), transparent 30rem),
        linear-gradient(180deg, #06101d 0%, #081321 100%);
        color: var(--text);
      }
      html, body, [class*="css"] { font-size: 15px; color: var(--text); }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1500px; }
      h1 { font-size: 2.15rem !important; letter-spacing: 0 !important; margin-bottom: .2rem; color: #eef7ff; }
      h2 { font-size: 1.45rem !important; margin-top: .7rem; color: #eaf6ff; }
      h3 { font-size: 1.12rem !important; margin-top: .45rem; color: #cde8ff; }
      p, li, .stMarkdown, .stCaption { color: var(--text); }
      section[data-testid="stSidebar"] { background: #07101c; border-right: 1px solid #17334d; min-width: 250px !important; width: 250px !important; }
      section[data-testid="stSidebar"] * { color: #cfe6ff; }
      [data-testid="stSidebarCollapsedControl"] { display: none; }
      div[role="radiogroup"] label {
        background: rgba(15,34,55,.68);
        border: 1px solid rgba(38,217,255,.16);
        border-radius: 8px;
        padding: .28rem .45rem;
        margin-bottom: .22rem;
      }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15,34,55,.96), rgba(9,22,37,.96));
        border: 1px solid rgba(38,217,255,.28);
        box-shadow: 0 0 24px rgba(38,217,255,.08), inset 0 1px 0 rgba(255,255,255,.05);
        border-radius: 10px;
        padding: .78rem .85rem;
        min-height: 104px;
      }
      [data-testid="stMetricLabel"] { font-size: .88rem; color: var(--muted); }
      [data-testid="stMetricValue"] { font-size: 1.35rem; color: #f3fbff; }
      [data-testid="stMetricDelta"] { font-size: .8rem; color: var(--green); }
      div[data-testid="stDataFrame"] {
        border: 1px solid rgba(38,217,255,.22);
        border-radius: 10px;
        overflow: hidden;
        background: rgba(13,26,43,.86);
      }
      div[data-testid="stAlert"] { background: rgba(13,34,56,.92); border: 1px solid rgba(38,217,255,.25); color: var(--text); }
      .stButton>button {
        background: linear-gradient(90deg, #0fb5d8, #18d198);
        color: #03111c;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
      }
      .tech-card {
        background: linear-gradient(145deg, rgba(15,34,55,.92), rgba(8,18,31,.92));
        border: 1px solid rgba(38,217,255,.24);
        border-radius: 12px;
        padding: .85rem 1rem;
        box-shadow: 0 0 28px rgba(38,217,255,.08);
      }
      .muted { color: var(--muted); }
      .chip { display:inline-block; padding:.18rem .5rem; border:1px solid rgba(45,245,168,.35); color:#bafbe0; border-radius:999px; font-size:.82rem; margin-right:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_metric(result, value_col: str = "value") -> tuple[float | None, str]:
    if not result or result.data.empty:
        return None, "暂无"
    return latest_number(result.data, value_col), latest_date(result.data)


def chart_line(title: str, df: pd.DataFrame, threshold: float | None = None):
    fig = go.Figure()
    if not df.empty and {"date", "value"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines+markers", name=title, line=dict(color="#26d9ff", width=2), marker=dict(size=5, color="#2df5a8")))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#ffd166")
    fig.update_layout(
        height=270,
        title=dict(text=title, font=dict(size=16, color="#d9e7ff")),
        margin=dict(l=8, r=8, t=38, b=8),
        font=dict(size=12, color="#9fb7d5"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,31,.76)",
        xaxis=dict(gridcolor="rgba(127,155,184,.14)"),
        yaxis=dict(gridcolor="rgba(127,155,184,.14)"),
    )
    return fig


def dataframe_preview(result, rows: int = 12, height: int = 260):
    if not result:
        return False
    if result.data.empty:
        return False
    else:
        df = result.data.tail(rows).copy()
        preferred = [
            "date", "日期", "抓取日期", "标题", "title", "内容", "name", "名称", "代码", "symbol",
            "value", "今值", "最新价", "涨跌幅", "涨跌额", "成交额", "市盈率", "市净率", "source",
        ]
        cols = [col for col in preferred if col in df.columns]
        if len(cols) >= 2:
            df = df[cols[:8]]
        else:
            df = df.iloc[:, :8]
        st.dataframe(df, width="stretch", hide_index=True, height=height)
        return True


def metric_card(label: str, result, suffix: str = ""):
    val, date = normalize_metric(result)
    st.metric(label, "暂无" if val is None else f"{val:.2f}{suffix}", date)


def fmt_value(value: float | None, suffix: str = "") -> str:
    return "暂无" if value is None else f"{value:.2f}{suffix}"


def refresh_times(refresh_log: pd.DataFrame) -> tuple[str, str]:
    if refresh_log.empty:
        return "暂无", "等待首次更新"
    latest = pd.to_datetime(refresh_log["created_at"].iloc[0], errors="coerce")
    if pd.isna(latest):
        return str(refresh_log["created_at"].iloc[0]), "约半小时后"
    return latest.strftime("%Y-%m-%d %H:%M:%S"), (latest + timedelta(seconds=REFRESH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")


def ai_report_times() -> tuple[str, str, str]:
    reports = latest_ai_reports(1)
    if reports.empty:
        return "暂无", "等待首次报告", "暂无"
    latest = pd.to_datetime(reports["created_at"].iloc[0], errors="coerce")
    status = str(reports["status"].iloc[0])
    if pd.isna(latest):
        return str(reports["created_at"].iloc[0]), "约12小时后", status
    return latest.strftime("%Y-%m-%d %H:%M:%S"), (latest + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), status


def render_update_status(refresh_log: pd.DataFrame):
    updated_at, next_at = refresh_times(refresh_log)
    if refresh_log.empty:
        inserted = updated = 0
    else:
        inserted = int(refresh_log.head(50)["rows_inserted"].sum())
        updated = int(refresh_log.head(50)["rows_updated"].sum())
    cols = st.columns(4)
    cols[0].metric("数据更新到", updated_at)
    cols[1].metric("下次自动更新", next_at)
    cols[2].metric("近期新增", inserted)
    cols[3].metric("近期修订", updated)


def render_ai_report_status():
    report_at, next_report_at, status = ai_report_times()
    cols = st.columns(3)
    cols[0].metric("AI报告更新到", report_at)
    cols[1].metric("下次AI报告", next_report_at)
    cols[2].metric("最近报告状态", status)


def overview_reasons(scores, metrics: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        ("政策环境", scores["policy"].label, f"LPR趋势 {'下行' if metrics.get('lpr_1y_down') else '未明显下行'}，政策判断以稳增长、扩内需、防风险和科技创新为主。"),
        ("经济周期", scores["cycle"].label, f"GDP {fmt_value(metrics.get('gdp'), '%')}，PMI {fmt_value(metrics.get('pmi'))}，消费趋势 {fmt_value(metrics.get('retail_trend'))}，反映经济修复斜率。"),
        ("流动性环境", scores["liquidity"].label, f"M2 {fmt_value(metrics.get('m2'), '%')}，社融 {fmt_value(metrics.get('social_financing'))}，资金价格趋势 {fmt_value(metrics.get('rate_trend'))}。"),
        ("企业盈利", scores["earnings"].label, f"PPI {fmt_value(metrics.get('ppi'), '%')}，工业品价格和盈利修复方向是当前核心证据。"),
        ("估值水平", scores["valuation"].label, f"宽基估值口径约 {fmt_value(metrics.get('market_pe'))}，结合股债收益差和风险溢价观察安全边际。"),
        ("市场风险偏好", scores["risk"].label, f"由增长、流动性、估值三项合成判断；同时受北向资金、汇率、美债利率和热点主线影响。"),
    ]


def render_overview_reasons(scores, metrics):
    st.markdown("### 当前A股宏观环境总评")
    for module, label, reason in overview_reasons(scores, metrics):
        st.markdown(
            f"""
            <div class="tech-card" style="margin-bottom:.65rem;">
              <div style="font-size:1.05rem;color:#f3fbff;"><b>{module}：{label}</b></div>
              <div class="muted" style="margin-top:.25rem;line-height:1.65;">原因：{reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def cycle_position(metrics: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    gdp = metrics.get("gdp")
    pmi = metrics.get("pmi")
    ppi = metrics.get("ppi")
    m2 = metrics.get("m2")
    social_financing = metrics.get("social_financing")
    liquidity_score = scores["liquidity"].score
    risk_score = scores["risk"].score
    cycle_label = scores["cycle"].label

    growth_ok = (gdp is not None and gdp >= 5) + (pmi is not None and pmi >= 50)
    price_ok = ppi is not None and ppi >= 0
    money_ok = (m2 is not None and m2 >= 7) and (social_financing is not None and social_financing > 0)

    if (
        growth_ok >= 2
        and price_ok
        and liquidity_score >= 2
        and risk_score >= 4
        and scores["earnings"].label == "改善"
        and scores["valuation"].label == "低估"
        and "弱复苏" not in cycle_label
    ):
        phase = "复苏中段向繁荣过渡"
        x = 3.35
        y = -0.52
        advice = "权益资产可提高进攻性，但仍要验证盈利兑现和资金持续流入。"
    elif growth_ok >= 1 and liquidity_score >= 1:
        phase = "萧条末期 / 复苏初期"
        x = 3.05
        y = -0.82
        advice = "更像从底部向右侧修复的观察期，适合政策科技、高股息和基本面验证方向并行。"
    elif growth_ok <= 0 and liquidity_score < 1:
        phase = "衰退后段 / 萧条前段"
        x = 2.25
        y = -0.15
        advice = "优先控制风险，等待政策、信用和PMI给出更明确的底部确认。"
    elif growth_ok >= 1 and risk_score < 1:
        phase = "衰退中段"
        x = 1.55
        y = 0.62
        advice = "宏观并未完全失速，但风险偏好不足，行情更容易结构分化。"
    else:
        phase = "复苏初期"
        x = 3.15
        y = -0.72
        advice = "处于低位修复阶段，继续观察PMI、PPI、社融、地产和北向资金。"

    reasons = [
        f"GDP {fmt_value(gdp, '%')}、PMI {fmt_value(pmi)}，用于判断增长动能。",
        f"PPI {fmt_value(ppi, '%')}，用于判断价格和盈利修复斜率。",
        f"M2 {fmt_value(m2, '%')}、社融 {fmt_value(social_financing)}，用于判断信用和流动性是否支持风险资产。",
        f"当前综合评分：经济周期 {scores['cycle'].label}，流动性 {scores['liquidity'].label}，风险偏好 {scores['risk'].label}。",
    ]
    return {"phase": phase, "x": x, "y": y, "advice": advice, "reasons": reasons}


def render_cycle_position_curve(metrics: dict[str, Any], scores: dict[str, Any]):
    position = cycle_position(metrics, scores)
    x_values = [i / 100 for i in range(0, 401)]
    y_values = [__import__("math").sin((x / 4) * 2 * __import__("math").pi) for x in x_values]
    phases = [
        (0, 1, "繁荣期"),
        (1, 2, "衰退期"),
        (2, 3, "萧条期"),
        (3, 4, "复苏期"),
    ]
    fig = go.Figure()
    colors = ["rgba(45,245,168,.08)", "rgba(255,209,102,.08)", "rgba(255,99,132,.08)", "rgba(38,217,255,.09)"]
    for idx, (x0, x1, phase) in enumerate(phases):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[idx], line_width=0)
        fig.add_annotation(x=(x0 + x1) / 2, y=1.2, text=f"<b>{phase}</b>", showarrow=False, font=dict(size=15, color="#cde8ff"), align="center")
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", line=dict(color="#26d9ff", width=4), name="经济周期曲线"))
    fig.add_trace(
        go.Scatter(
            x=[position["x"]],
            y=[position["y"]],
            mode="markers+text",
            marker=dict(size=18, color="#ffd166", line=dict(color="#06101d", width=3)),
            text=["当前位置"],
            textposition="top center",
            textfont=dict(size=15, color="#fff2b8"),
            name="当前位置",
        )
    )
    fig.add_annotation(
        x=position["x"],
        y=position["y"] - 0.18,
        text=position["phase"],
        showarrow=False,
        font=dict(size=13, color="#fff2b8"),
        bgcolor="rgba(8,18,31,.72)",
        bordercolor="rgba(255,209,102,.55)",
        borderwidth=1,
        borderpad=4,
    )
    fig.add_hline(y=0, line_color="rgba(207,232,255,.25)", line_width=1)
    fig.update_xaxes(
        range=[0, 4],
        tickvals=[0.5, 1.5, 2.5, 3.5],
        ticktext=["繁荣", "衰退", "萧条", "复苏"],
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(range=[-1.35, 1.45], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=32, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,31,.72)",
        font=dict(color="#d9e7ff", size=13),
        showlegend=False,
    )
    st.markdown("### 经济周期定位")
    st.markdown(
        f"""
        <div class="tech-card" style="margin-bottom:.7rem;">
          <div style="font-size:1.25rem;color:#f3fbff;"><b>当前定位：{position['phase']}</b></div>
          <div class="muted" style="margin-top:.3rem;line-height:1.7;">{position['advice']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch")
    reason_cols = st.columns(2)
    for idx, reason in enumerate(position["reasons"]):
        with reason_cols[idx % 2]:
            st.markdown(f"<div class='tech-card' style='min-height:72px;margin-bottom:.45rem;'>{reason}</div>", unsafe_allow_html=True)


def markdown_to_plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def pdf_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_for_pdf(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue
        current = ""
        for ch in line:
            test = current + ch
            if draw.textlength(test, font=font) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def build_pdf_bytes(title: str, report_content: str, scores, metrics: dict[str, Any], refresh_log: pd.DataFrame) -> bytes:
    from PIL import Image, ImageDraw

    width, height = 1240, 1754
    margin = 90
    bg = (248, 251, 255)
    ink = (22, 35, 52)
    muted = (91, 112, 135)
    accent = (10, 132, 180)
    card = (232, 242, 250)
    title_font = pdf_font(46)
    h_font = pdf_font(30)
    body_font = pdf_font(24)
    small_font = pdf_font(20)

    pages = []

    def new_page():
        page = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(page)
        draw.rectangle((0, 0, width, 26), fill=(8, 24, 44))
        return page, draw, margin

    page, draw, y = new_page()
    draw.text((margin, y), title, font=title_font, fill=ink)
    y += 68
    updated_at, next_at = refresh_times(refresh_log)
    draw.text((margin, y), f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    数据更新到：{updated_at}    下次更新：{next_at}", font=small_font, fill=muted)
    y += 54

    draw.text((margin, y), "一、总览", font=h_font, fill=accent)
    y += 46
    overview_items = overview_reasons(scores, metrics)
    col_w = (width - margin * 2 - 24) // 2
    card_h = 118
    for idx, (module, label, reason) in enumerate(overview_items):
        x = margin + (idx % 2) * (col_w + 24)
        yy = y + (idx // 2) * (card_h + 18)
        draw.rounded_rectangle((x, yy, x + col_w, yy + card_h), radius=18, fill=card, outline=(190, 215, 230))
        draw.text((x + 22, yy + 16), f"{module}：{label}", font=body_font, fill=ink)
        wrapped = wrap_for_pdf(draw, "原因：" + reason, small_font, col_w - 44)[:2]
        ty = yy + 54
        for line in wrapped:
            draw.text((x + 22, ty), line, font=small_font, fill=muted)
            ty += 26
    y += 3 * (card_h + 18) + 40

    key_metrics = [
        ("GDP同比", metrics.get("gdp"), "%"),
        ("PMI", metrics.get("pmi"), ""),
        ("CPI同比", metrics.get("cpi"), "%"),
        ("PPI同比", metrics.get("ppi"), "%"),
        ("M2同比", metrics.get("m2"), "%"),
        ("社融", metrics.get("social_financing"), ""),
        ("估值PE", metrics.get("market_pe"), ""),
    ]
    draw.text((margin, y), "关键指标", font=h_font, fill=accent)
    y += 42
    metric_text = "    ".join(f"{name}: {fmt_value(value, suffix)}" for name, value, suffix in key_metrics)
    for line in wrap_for_pdf(draw, metric_text, body_font, width - 2 * margin):
        draw.text((margin, y), line, font=body_font, fill=ink)
        y += 34

    pages.append(page)

    plain_report = markdown_to_plain(report_content)
    page, draw, y = new_page()
    draw.text((margin, y), "二、AI宏观策略报告", font=title_font, fill=ink)
    y += 70
    max_width = width - 2 * margin
    line_height = 34
    for line in wrap_for_pdf(draw, plain_report, body_font, max_width):
        if y > height - margin:
            pages.append(page)
            page, draw, y = new_page()
        if re.match(r"^\d+\.\s", line):
            y += 10
            draw.text((margin, y), line, font=h_font, fill=accent)
            y += 44
        else:
            draw.text((margin, y), line, font=body_font, fill=ink if line else muted)
            y += line_height if line else 18
    pages.append(page)

    out = io.BytesIO()
    pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    return out.getvalue()


def render_pdf_download(label: str, title: str, report_content: str, file_prefix: str):
    if not report_content:
        st.info("暂无可导出的报告内容。")
        return
    pdf_bytes = build_pdf_bytes(title, report_content, scores, metrics, refresh_log)
    st.download_button(
        label,
        data=pdf_bytes,
        file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        width="stretch",
    )


def compact_latest(result) -> dict[str, Any]:
    if not result or result.data.empty:
        return {"source": getattr(result, "source", ""), "latest": None, "error": getattr(result, "error", "无数据")}
    df = result.data
    if {"date", "value"}.issubset(df.columns):
        return {
            "source": result.source,
            "latest_date": latest_date(df),
            "latest_value": latest_number(df),
            "trend": trend(df),
        }
    return {
        "source": result.source,
        "columns": list(df.columns)[:10],
        "sample": df.head(5).to_dict(orient="records"),
    }


def dataset_summary(result, full_series: bool = False, recent_rows: int = 8, series_tail: int = 18, max_text_chars: int = 260) -> dict[str, Any]:
    if not result or result.data.empty:
        return {"name": getattr(result, "name", ""), "source": getattr(result, "source", ""), "rows": 0, "status": "missing"}
    df = result.data.copy()
    summary: dict[str, Any] = {
        "name": result.name,
        "source": result.source,
        "rows": len(df),
        "columns": list(df.columns)[:12],
    }
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
                "latest_date": latest_date(df),
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
    keywords = [
        "稳增长", "扩内需", "消费", "投资", "房地产", "科技创新", "人工智能", "半导体", "新能源", "高端装备",
        "新质生产力", "央国企", "资本市场", "金融", "外贸", "出口", "乡村振兴", "区域协调", "军工", "医药",
    ]
    for _, row in df.tail(240).iterrows():
        text = " ".join(str(row.get(col, "")) for col in text_cols)
        for keyword in keywords:
            if keyword in text:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    summary["policy_keyword_counts"] = dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:15])
    return summary


ROLE_DATA_MAP = [
    {"数据范围": "GDP/PMI/CPI/PPI/社零/投资/进出口/地产/失业率/工业增加值", "使用角色": "宏观经济学家", "用途": "判断经济周期、过去五年阶段切换和弱复苏/下行风险"},
    {"数据范围": "财政收入/LPR/新闻联播/宏观热点", "使用角色": "政策研究员", "用途": "判断财政、货币、产业政策力度和政策主线"},
    {"数据范围": "M2/社融/LPR/DR007/Tushare北向/东方财富北向/人民币汇率", "使用角色": "流动性与利率交易员", "用途": "判断银行间流动性、股市可用资金、外资风险偏好和汇率压力"},
    {"数据范围": "PPI/指数估值/中证估值/主要指数/A股行情", "使用角色": "权益估值分析师", "用途": "判断盈利修复、宽基估值、安全边际和估值陷阱"},
    {"数据范围": "美元指数/美债利率/美国通胀就业/中美利差/人民币长序列/全球商品", "使用角色": "全球宏观策略师", "用途": "判断外部环境对A股估值、外资和行业风格的约束"},
    {"数据范围": "行业资金流/人气榜/飙升榜/东方财富妙想/新闻联播/热点新闻", "使用角色": "行业配置策略师", "用途": "交叉验证热点是否有政策、资金和产业证据，筛出4-5个方向"},
    {"数据范围": "地产/新房价格/Tushare北向/东方财富北向/人民币汇率/外储/美债/中美利差/估值", "使用角色": "投资风险管理官", "用途": "形成主要风险数据库，检查外资、汇率、美债、地产、估值和盈利风险"},
]


def risk_source_items(macro, tushare_data, market, global_data) -> list[tuple[str, Any]]:
    return [
        ("地产：房地产景气", macro.get("real_estate")),
        ("地产：新房价格", macro.get("new_house_price")),
        ("北向：Tushare沪深港通资金流", tushare_data.get("tushare_hsgt_moneyflow")),
        ("北向：东方财富沪深港通资金", market.get("north")),
        ("外汇：人民币主要汇率长序列", market.get("fx_boc_safe")),
        ("外汇：实时人民币汇率", market.get("fx")),
        ("外汇：外汇储备", macro.get("fx_reserves")),
        ("海外：美国利率", global_data.get("us_rate")),
        ("海外：中美国债收益率", global_data.get("cn_us_rate_spread")),
        ("海外：美元指数/全球指数", global_data.get("dxy")),
        ("估值：中证指数估值", market.get("csindex_valuation")),
        ("估值：宽基指数估值", market.get("index_pe")),
        ("盈利：PPI", macro.get("ppi")),
        ("盈利：工业增加值", macro.get("industrial_value_added")),
    ]


def risk_evidence_database(macro, tushare_data, market, global_data) -> dict[str, Any]:
    groups = {
        "地产拖累": [("real_estate", macro.get("real_estate")), ("new_house_price", macro.get("new_house_price"))],
        "外资与北向资金": [("tushare_hsgt_moneyflow", tushare_data.get("tushare_hsgt_moneyflow")), ("north", market.get("north"))],
        "人民币与外汇": [("fx_boc_safe", market.get("fx_boc_safe")), ("fx", market.get("fx")), ("fx_reserves", macro.get("fx_reserves"))],
        "海外利率与中美利差": [("us_rate", global_data.get("us_rate")), ("cn_us_rate_spread", global_data.get("cn_us_rate_spread")), ("dxy", global_data.get("dxy"))],
        "估值与盈利下修": [("csindex_valuation", market.get("csindex_valuation")), ("index_pe", market.get("index_pe")), ("ppi", macro.get("ppi")), ("industrial_value_added", macro.get("industrial_value_added"))],
    }
    return {
        risk_name: [
            {"dataset": dataset, **dataset_summary(result, recent_rows=10, series_tail=18, max_text_chars=180)}
            for dataset, result in datasets
            if result is not None
        ]
        for risk_name, datasets in groups.items()
    }


def build_ai_snapshot(scores, macro, tushare_data, market, global_data, news, xinwen_lianbo, refresh_log) -> dict[str, Any]:
    groups = {"macro": macro, "tushare": tushare_data, "market": market, "global": global_data}
    local_mx = globals().get("mx_data", {})
    coverage = []
    for group_name, group in groups.items():
        for dataset, result in group.items():
            coverage.append({"group": group_name, "dataset": dataset, "name": result.name, "source": result.source, "rows": len(result.data)})
    coverage.append({"group": "news", "dataset": "news", "name": news.name, "source": news.source, "rows": len(news.data)})
    coverage.append({"group": "policy_news", "dataset": "xinwen_lianbo", "name": xinwen_lianbo.name, "source": xinwen_lianbo.source, "rows": len(xinwen_lianbo.data)})
    for dataset, result in local_mx.items():
        coverage.append({"group": "eastmoney_mx", "dataset": dataset, "name": result.name, "source": result.source, "rows": len(result.data)})
    important_macro = [
        "gdp", "pmi", "cpi", "ppi", "retail", "investment", "exports", "imports",
        "m2", "social_financing", "lpr", "dr007", "real_estate", "fx_reserves",
        "new_house_price", "fiscal_revenue", "industrial_value_added",
    ]
    important_tushare = ["tushare_gdp", "tushare_cpi", "tushare_ppi", "tushare_m2", "tushare_social_financing", "tushare_pmi", "tushare_hsgt_moneyflow"]
    important_market = ["index_spot", "index_pe", "csindex_valuation", "north", "fund_flow", "hot_rank", "hot_up", "fx", "fx_boc_safe"]
    important_global = ["dxy", "commodity", "us_rate", "us_cpi", "us_core_cpi", "us_nonfarm", "us_unemployment", "us_ism_pmi", "cn_us_rate_spread"]
    return {
        "overview_scores": {key: value.__dict__ for key, value in scores.items()},
        "coverage": coverage,
        "analysis_logic": {
            "step_1_policy": "先看政策环境：财政、货币、产业政策是否偏宽松，决定市场底部托底和结构方向；新闻联播作为政策重心和产业主线的重要确认源。",
            "step_2_cycle": "再看经济周期：GDP、PMI、社零、投资、进出口、地产，判断复苏、弱复苏、下行或过热。",
            "step_3_liquidity": "再看流动性：M2、社融、LPR、DR007、北向资金、汇率，判断资金是否支持估值扩张。",
            "step_4_earnings": "再看企业盈利：PPI、工业品价格、需求指标和宽基表现，判断盈利处于改善、见底或承压。",
            "step_5_valuation": "再看估值：宽基指数PE/PB、股债收益差、风险溢价，判断安全边际。",
            "step_6_external": "再看外部环境：美债、美元、人民币、中美利差、美国通胀/就业/PMI，判断外部约束。",
            "step_7_industry": "最后映射行业：把宏观组合映射到高股息、科技成长、消费、周期、出口链、地产链等方向，并用资金流和热点验证。",
        },
        "macro": {key: dataset_summary(macro[key], full_series=True) for key in important_macro if key in macro},
        "tushare": {key: dataset_summary(tushare_data[key], full_series=True) for key in important_tushare if key in tushare_data},
        "market": {key: dataset_summary(market[key], recent_rows=8, series_tail=16, max_text_chars=180) for key in important_market if key in market},
        "global": {key: dataset_summary(global_data[key], recent_rows=8, series_tail=16, max_text_chars=180) for key in important_global if key in global_data},
        "xinwen_lianbo_policy_signal": policy_signal_summary(xinwen_lianbo, limit=80),
        "eastmoney_mx": {key: dataset_summary(result, recent_rows=24, series_tail=16, max_text_chars=220) for key, result in local_mx.items()},
        "hot_news": dataset_summary(news, recent_rows=18, max_text_chars=220),
        "risk_evidence_database": risk_evidence_database(macro, tushare_data, market, global_data),
        "role_data_map": ROLE_DATA_MAP,
        "refresh_log_latest": refresh_log.head(12).to_dict(orient="records") if not refresh_log.empty else [],
    }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def first_latest(groups: list[dict[str, Any]], keys: list[str]) -> float | None:
    for group in groups:
        for key in keys:
            result = group.get(key)
            if result is not None and not result.data.empty:
                value = latest_number(result.data)
                if value is not None:
                    return value
    return None


def first_non_empty_result(*results):
    for result in results:
        if result is not None and not result.data.empty:
            return result
    return next((result for result in results if result is not None), None)


FRESHNESS_PROFILES = {
    "daily": (10, "日频/交易日数据，正常应接近最近交易日"),
    "weekly": (21, "周频或不定期高频数据，允许几周滞后"),
    "monthly": (75, "月度宏观数据，通常滞后 1-2 个月发布"),
    "quarterly": (150, "季度数据，通常滞后 1 个季度左右发布"),
    "event": (365, "事件/报告类数据，不要求每日更新"),
    "snapshot": (30, "实时快照类数据，接口可能不带日期字段"),
}

DATASET_FREQUENCY = {
    "gdp": "quarterly",
    "tushare_gdp": "quarterly",
    "us_gdp": "quarterly",
    "pmi": "monthly",
    "cpi": "monthly",
    "ppi": "monthly",
    "retail": "monthly",
    "investment": "monthly",
    "exports": "monthly",
    "imports": "monthly",
    "m2": "monthly",
    "social_financing": "monthly",
    "tushare_cpi": "monthly",
    "tushare_ppi": "monthly",
    "tushare_m2": "monthly",
    "tushare_social_financing": "monthly",
    "tushare_pmi": "monthly",
    "real_estate": "monthly",
    "unemployment": "monthly",
    "fx_reserves": "monthly",
    "new_house_price": "monthly",
    "fiscal_revenue": "monthly",
    "industrial_value_added": "monthly",
    "us_cpi": "monthly",
    "us_core_cpi": "monthly",
    "us_nonfarm": "monthly",
    "us_unemployment": "monthly",
    "us_retail": "monthly",
    "us_ism_pmi": "monthly",
    "us_trade": "monthly",
    "lpr": "daily",
    "dr007": "daily",
    "tushare_hsgt_moneyflow": "daily",
    "north": "daily",
    "index_spot": "daily",
    "a_spot": "snapshot",
    "hot_rank": "daily",
    "hot_up": "daily",
    "fund_flow": "daily",
    "index_pe": "daily",
    "csindex_valuation": "daily",
    "fx": "snapshot",
    "fx_boc_safe": "daily",
    "dxy": "daily",
    "commodity": "daily",
    "us_rate": "event",
    "cn_us_rate_spread": "daily",
    "commodity_price": "weekly",
    "au_report": "event",
    "news": "daily",
    "xinwen_lianbo": "daily",
    "mx_search": "daily",
    "mx_finance": "daily",
}

DATASET_FALLBACK_ACTIONS = {
    "pmi": "AKShare 源曾停在旧月份；优先用 Tushare PMI 核对，保留 AKShare 作历史对照。",
    "cpi": "AKShare 源曾停在旧月份；优先用 Tushare CPI 核对，保留 AKShare 作历史对照。",
    "ppi": "AKShare 源曾停在旧月份；优先用 Tushare PPI 核对，保留 AKShare 作历史对照。",
    "m2": "AKShare 源曾停在旧月份；优先用 Tushare M2 核对，保留 AKShare 作历史对照。",
    "exports": "海关月度数据可能接口滞后；若 AKShare 旧，用新闻/政策和 Tushare 同类宏观口径辅助判断外需。",
    "imports": "海关月度数据可能接口滞后；若 AKShare 旧，用新闻/政策和 Tushare 同类宏观口径辅助判断内需。",
    "real_estate": "地产景气指数发布和接口同步较慢；同时参考新房价格、财政收入、政策新闻和风险库。",
    "fx_reserves": "外储为月度发布；若 AKShare 外储滞后，用人民币汇率长序列和美元指数作为外部压力替代验证。",
    "retail": "已在抓取层补 date/value；若接口列结构变化，页面和 AI 仍读取原始表格。",
    "investment": "已在抓取层补 date/value；若接口列结构变化，页面和 AI 仍读取原始表格。",
    "social_financing": "已在抓取层补 date/value；同时优先使用 Tushare 社融月度数据作为流动性核心口径。",
    "lpr": "已在抓取层补 date/value；日频展示按最近公布日判断，不要求每天变化。",
    "commodity_price": "商务部大宗价格指数常无标准日期列；已在抓取层补抓取日期，作为辅助指标。",
    "a_spot": "行情快照可能无日期列；刷新成功时按最近抓取时间判断。",
    "index_spot": "行情快照可能无日期列；刷新成功时按最近抓取时间判断。",
    "hot_rank": "热度榜常无日期列；刷新成功时按最近抓取时间判断。",
    "hot_up": "热度榜常无日期列；若东方财富/雪球失败，保留历史缓存并用新闻联播/妙想资讯核对热点。",
    "fund_flow": "东方财富资金流接口偶发 502；失败时保留历史缓存，并用妙想金融数据/热点榜交叉验证行业方向。",
    "commodity": "东方财富全球商品接口偶发超时；失败时保留历史缓存，并用国内大宗价格指数和美元/美债辅助验证。",
    "dxy": "美元指数源可能只返回快照；刷新成功时按最近抓取时间判断。",
}


def dataset_latest_timestamp(df: pd.DataFrame) -> tuple[pd.Timestamp | None, str]:
    if df is None or df.empty:
        return None, ""
    for col in ("date", "日期", "trade_date", "ann_date", "end_date", "quarter", "报告期", "抓取日期", "created_at"):
        if col not in df.columns:
            continue
        raw = df[col].astype(str)
        normalized = raw.str.replace("Q1", "-03-01", regex=False).str.replace("Q2", "-06-01", regex=False).str.replace("Q3", "-09-01", regex=False).str.replace("Q4", "-12-01", regex=False)
        zh_month = normalized.str.extract(r"(?P<year>\d{4})年(?P<month>\d{1,2})月")
        has_zh_month = zh_month["year"].notna()
        normalized.loc[has_zh_month] = zh_month.loc[has_zh_month, "year"] + "-" + zh_month.loc[has_zh_month, "month"].str.zfill(2) + "-01"
        mask8 = normalized.str.fullmatch(r"\d{8}")
        normalized.loc[mask8] = normalized.loc[mask8].str[:4] + "-" + normalized.loc[mask8].str[4:6] + "-" + normalized.loc[mask8].str[6:8]
        mask6 = normalized.str.fullmatch(r"\d{6}")
        normalized.loc[mask6] = normalized.loc[mask6].str[:4] + "-" + normalized.loc[mask6].str[4:6] + "-01"
        parsed = pd.to_datetime(normalized, errors="coerce")
        if parsed.notna().sum():
            latest = parsed.max()
            return latest, latest.strftime("%Y-%m-%d")
    return None, ""


def latest_refresh_row(refresh_log: pd.DataFrame, dataset: str) -> dict[str, Any]:
    if refresh_log.empty or "dataset" not in refresh_log.columns:
        return {}
    part = refresh_log[refresh_log["dataset"] == dataset]
    if part.empty:
        return {}
    return part.iloc[0].to_dict()


def freshness_status(dataset: str, result: DataResult, refresh_row: dict[str, Any]) -> tuple[str, int | None, str]:
    rows = 0 if result is None or result.data.empty else len(result.data)
    error = str(refresh_row.get("error") or "")
    refresh_status_value = str(refresh_row.get("status") or "")
    if rows == 0:
        if error or refresh_status_value == "error":
            return "无数据 / 需要替代源", None, error[:180] or "最近刷新失败"
        return "无数据 / 等待首次落库", None, "本地库暂无记录"
    latest_ts, _ = dataset_latest_timestamp(result.data)
    freq = DATASET_FREQUENCY.get(dataset, "monthly")
    max_days, note = FRESHNESS_PROFILES[freq]
    if latest_ts is None:
        if freq == "snapshot":
            if refresh_status_value == "ok":
                return "正常", None, f"{note}；最近刷新成功，按刷新时间判断"
            return "实时快照 / 无日期字段", None, note
        return "有数据但无日期字段", None, "无法判断数据新鲜度，需检查字段结构"
    lag_days = int((pd.Timestamp(datetime.now().date()) - latest_ts.normalize()).days)
    if error or refresh_status_value == "error":
        return "使用历史缓存 / 源异常", lag_days, error[:180] or "最近刷新失败，当前展示本地缓存"
    if lag_days <= max_days:
        return "正常", lag_days, note
    if lag_days <= max_days * 2:
        return "偏旧 / 需关注", lag_days, f"{note}；已超过预期窗口"
    return "明显滞后 / 需要替代源", lag_days, f"{note}；滞后时间过长，建议补备用接口"


def freshness_action(dataset: str, status_text: str, refresh_row: dict[str, Any]) -> str:
    if status_text == "正常":
        return "无需处理，继续半小时增量更新。"
    if status_text == "使用历史缓存 / 源异常":
        return DATASET_FALLBACK_ACTIONS.get(dataset, "接口临时失败，保留历史缓存；下轮自动重试，必要时补替代源。")
    return DATASET_FALLBACK_ACTIONS.get(dataset, "继续自动重试；若连续滞后，优先补同口径备用接口并在 AI 输入中标注口径。")


def build_freshness_table(macro, tushare_data, market, global_data, mx_data, news, xinwen_lianbo, refresh_log: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, dict[str, DataResult]]] = [
        ("宏观", macro),
        ("Tushare", tushare_data),
        ("市场", market),
        ("外部", global_data),
        ("东方财富妙想", mx_data),
        ("新闻", {"news": news, "xinwen_lianbo": xinwen_lianbo}),
    ]
    rows = []
    for group_name, group in groups:
        for dataset, result in group.items():
            refresh_row = latest_refresh_row(refresh_log, dataset)
            latest_ts, latest_text = dataset_latest_timestamp(result.data if result else pd.DataFrame())
            status_text, lag_days, note = freshness_status(dataset, result, refresh_row)
            rows.append(
                {
                    "分组": group_name,
                    "数据集": dataset,
                    "指标": getattr(result, "name", ""),
                    "本地行数": 0 if result is None or result.data.empty else len(result.data),
                    "最新数据日期": latest_text or "无日期字段",
                    "滞后天数": "" if lag_days is None else lag_days,
                    "新鲜度状态": status_text,
                    "预期频率": DATASET_FREQUENCY.get(dataset, "monthly"),
                    "最近刷新时间": refresh_row.get("created_at", ""),
                    "最近刷新状态": refresh_row.get("status", ""),
                    "说明/错误": (str(refresh_row.get("error") or note))[:240],
                    "处理方案": freshness_action(dataset, status_text, refresh_row),
                }
            )
    return pd.DataFrame(rows)


def render_freshness_page(freshness_df: pd.DataFrame):
    st.markdown("### 数据新鲜度")
    st.caption("按每个指标的正常发布周期判断：月度/季度宏观数据不要求更新到今天；日频行情、汇率、北向、新闻应尽量接近最近交易日。")
    if freshness_df.empty:
        st.info("暂无数据新鲜度记录。")
        return
    counts = freshness_df["新鲜度状态"].value_counts()
    cols = st.columns(5)
    for col, label in zip(cols, ["正常", "偏旧 / 需关注", "明显滞后 / 需要替代源", "无数据 / 需要替代源", "实时快照 / 无日期字段"]):
        col.metric(label, int(counts.get(label, 0)))
    status_filter = st.multiselect(
        "按状态筛选",
        options=sorted(freshness_df["新鲜度状态"].unique().tolist()),
        default=sorted(freshness_df["新鲜度状态"].unique().tolist()),
    )
    filtered = freshness_df[freshness_df["新鲜度状态"].isin(status_filter)].copy()
    status_order = {
        "无数据 / 需要替代源": 0,
        "明显滞后 / 需要替代源": 1,
        "偏旧 / 需关注": 2,
        "有数据但无日期字段": 3,
        "使用历史缓存 / 源异常": 4,
        "实时快照 / 无日期字段": 5,
        "无数据 / 等待首次落库": 6,
        "正常": 7,
    }
    filtered["_order"] = filtered["新鲜度状态"].map(status_order).fillna(9)
    filtered = filtered.sort_values(["_order", "分组", "数据集"]).drop(columns=["_order"])
    st.dataframe(filtered, width="stretch", hide_index=True, height=620)


def render_source_grid(items: list[tuple[str, Any]], rows: int = 8):
    available = [(title, result) for title, result in items if result is not None and not result.data.empty]
    if not available:
        st.info("当前模块暂无可展示数据，后台会继续尝试增量更新。")
        return
    cols = st.columns(2)
    for idx, (title, result) in enumerate(available):
        with cols[idx % 2]:
            st.markdown(f"<div class='tech-card'><span class='chip'>{title}</span><span class='muted'> {result.source}</span></div>", unsafe_allow_html=True)
            dataframe_preview(result, rows=rows, height=230)


def render_status(group_title: str, group: dict[str, DataResult]):
    st.markdown(f"### {group_title}")
    rows = []
    for dataset, result in group.items():
        rows.append(
            {
                "数据集": dataset,
                "指标": result.name,
                "来源": result.source,
                "本地行数": len(result.data),
                "状态": "有数据" if not result.data.empty else "暂无",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)


st.title("A股宏观环境分析终端")
st.caption("本地近五年数据库 + 半小时增量更新 + 多源核对。最终报告在左侧目录最后一项「最终结论」。")

with st.sidebar:
    st.header("目录")
    page = st.radio(
        "选择页面",
        ["总览", "政策与经济", "流动性", "盈利与估值", "外部环境", "行业热点", "主要风险", "数据新鲜度", "数据源状态", "AI分析过程", "报告日志", "最终结论"],
        label_visibility="collapsed",
    )
    st.divider()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://www.bytego.team/v1")
    openai_model = os.getenv("OPENAI_MODEL", "chatgpt-5.5")
    st.caption("后台每 30 分钟增量更新，不删除历史。")
    if st.button("立即更新数据", type="primary", width="stretch"):
        if lock_is_active(DATA_UPDATE_LOCK, 20 * 60):
            st.toast("数据更新已经在后台运行。")
        else:
            touch_lock(DATA_UPDATE_LOCK)
            trigger_background_update()
            st.cache_data.clear()
            st.toast("已触发后台增量更新，页面不会阻塞；稍后自动刷新展示新数据。")

if "llm_request_id" not in st.session_state:
    st.session_state.llm_request_id = 0

data_bundle = load_local_bundle()
macro = data_bundle["macro"]
tushare_data = data_bundle["tushare"]
market = data_bundle["market"]
global_data = data_bundle["global"]
news = data_bundle["news"]
xinwen_lianbo = data_bundle["xinwen_lianbo"]
mx_data = data_bundle["mx"]
refresh_log = data_bundle["refresh_log"]
maybe_auto_update_data(refresh_log)
maybe_auto_ai_report()

gdp = first_latest([tushare_data, macro], ["tushare_gdp", "gdp"])
pmi = first_latest([tushare_data, macro], ["tushare_pmi", "pmi"])
cpi = first_latest([tushare_data, macro], ["tushare_cpi", "cpi"])
ppi = first_latest([tushare_data, macro], ["tushare_ppi", "ppi"])
m2 = first_latest([tushare_data, macro], ["tushare_m2", "m2"])
social_financing = None
for result in (tushare_data.get("tushare_social_financing"), macro.get("social_financing")):
    if result and not result.data.empty:
        value = latest_number(result.data)
        if value is not None:
            social_financing = value
            break
        nums = result.data.select_dtypes(include="number")
        if not nums.empty:
            social_financing = float(nums.iloc[-1].dropna().iloc[-1])
            break

market_pe = None
if market.get("index_pe") and not market.get("index_pe").data.empty:
    nums = market.get("index_pe").data.select_dtypes(include="number")
    if not nums.empty:
        market_pe = float(nums.iloc[-1].dropna().iloc[0])

metrics = {
    "gdp": gdp,
    "pmi": pmi,
    "cpi": cpi,
    "ppi": ppi,
    "m2": m2,
    "social_financing": social_financing,
    "market_pe": market_pe,
    "equity_risk_premium": None,
    "retail_trend": trend(macro.get("retail").data) if macro.get("retail") else 0,
    "industrial_profit_trend": trend(macro.get("ppi").data) if macro.get("ppi") else 0,
    "rate_trend": trend(macro.get("dr007").data) if macro.get("dr007") else 0,
    "lpr_1y_down": trend(macro.get("lpr").data) < 0 if macro.get("lpr") else False,
    "rrr_down": False,
}
scores = classify_environment(metrics)

if page == "总览":
    render_cycle_position_curve(metrics, scores)
    render_update_status(refresh_log)
    overview_cols = st.columns(6)
    for col, key, title in zip(
        overview_cols,
        ["policy", "cycle", "liquidity", "earnings", "valuation", "risk"],
        ["政策环境", "经济周期", "流动性", "企业盈利", "估值水平", "风险偏好"],
    ):
        with col:
            st.metric(title, scores[key].label, scores[key].score)

    render_overview_reasons(scores, metrics)

    cols = st.columns(6)
    with cols[0]:
        metric_card("GDP同比", first_non_empty_result(tushare_data.get("tushare_gdp"), macro.get("gdp")), "%")
    with cols[1]:
        metric_card("制造业PMI", first_non_empty_result(tushare_data.get("tushare_pmi"), macro.get("pmi")))
    with cols[2]:
        metric_card("CPI同比", first_non_empty_result(tushare_data.get("tushare_cpi"), macro.get("cpi")), "%")
    with cols[3]:
        metric_card("PPI同比", first_non_empty_result(tushare_data.get("tushare_ppi"), macro.get("ppi")), "%")
    with cols[4]:
        metric_card("M2同比", first_non_empty_result(tushare_data.get("tushare_m2"), macro.get("m2")), "%")
    with cols[5]:
        metric_card("社融", first_non_empty_result(tushare_data.get("tushare_social_financing"), macro.get("social_financing")))

    if not refresh_log.empty:
        updated_at, next_at = refresh_times(refresh_log)
        st.caption(f"数据更新到：{updated_at}；下一次自动更新：{next_at}。历史数据保存在本地 SQLite，不会因刷新删除。")
    else:
        st.warning("本地库还没有数据，请点击侧边栏“立即更新数据”。")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_line("GDP同比", tushare_data.get("tushare_gdp").data if tushare_data.get("tushare_gdp") else pd.DataFrame()), width="stretch")
        st.plotly_chart(chart_line("制造业PMI", tushare_data.get("tushare_pmi").data if tushare_data.get("tushare_pmi") else pd.DataFrame(), threshold=50), width="stretch")
    with c2:
        st.markdown("### 新闻联播政策信号")
        dataframe_preview(xinwen_lianbo, rows=8, height=260)
        st.markdown("### 最新宏观热点")
        dataframe_preview(news, rows=8, height=260)

elif page == "政策与经济":
    st.markdown("### 政策环境：财政、货币、产业政策")
    st.write(scores["policy"].conclusion)
    st.markdown("### 经济基础")
    st.write(scores["cycle"].conclusion)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_line("GDP同比", macro.get("gdp").data if macro.get("gdp") else pd.DataFrame()), width="stretch")
        st.plotly_chart(chart_line("CPI同比", macro.get("cpi").data if macro.get("cpi") else pd.DataFrame()), width="stretch")
    with c2:
        st.plotly_chart(chart_line("制造业PMI", macro.get("pmi").data if macro.get("pmi") else pd.DataFrame(), threshold=50), width="stretch")
        st.plotly_chart(chart_line("PPI同比", macro.get("ppi").data if macro.get("ppi") else pd.DataFrame(), threshold=0), width="stretch")
    render_source_grid(
        [
            ("新闻联播政策信号", xinwen_lianbo),
            ("东方财富妙想资讯", mx_data.get("mx_search")),
            ("LPR", macro.get("lpr")),
            ("社零", macro.get("retail")),
            ("固定资产投资", macro.get("investment")),
            ("出口", macro.get("exports")),
            ("进口", macro.get("imports")),
            ("房地产景气", macro.get("real_estate")),
            ("调查失业率", macro.get("unemployment")),
            ("外汇储备", macro.get("fx_reserves")),
            ("新房价格", macro.get("new_house_price")),
            ("大宗商品价格", macro.get("commodity_price")),
        ],
        rows=8,
    )

elif page == "流动性":
    st.markdown("### 流动性环境")
    st.write(scores["liquidity"].conclusion)
    render_source_grid(
        [
            ("M2 同比", first_non_empty_result(tushare_data.get("tushare_m2"), macro.get("m2"))),
            ("社融规模", first_non_empty_result(tushare_data.get("tushare_social_financing"), macro.get("social_financing"))),
            ("DR/SHIBOR", macro.get("dr007")),
            ("LPR", macro.get("lpr")),
            ("Tushare北向资金", tushare_data.get("tushare_hsgt_moneyflow")),
            ("北向资金", market.get("north")),
            ("人民币主要汇率长序列", market.get("fx_boc_safe")),
            ("实时人民币汇率", market.get("fx")),
        ],
        rows=10,
    )

elif page == "盈利与估值":
    st.markdown("### 企业盈利")
    st.write(scores["earnings"].conclusion)
    st.markdown("### 估值水平")
    st.write(scores["valuation"].conclusion)
    render_source_grid(
        [
            ("PPI 同比", first_non_empty_result(tushare_data.get("tushare_ppi"), macro.get("ppi"))),
            ("指数估值", market.get("index_pe")),
            ("主要指数", market.get("index_spot")),
            ("A股行情", market.get("a_spot")),
        ],
        rows=12,
    )

elif page == "外部环境":
    st.markdown("### 外部环境")
    render_source_grid(
        [
            ("美元指数/全球指数", global_data.get("dxy")),
            ("美国利率", global_data.get("us_rate")),
            ("中美国债收益率", global_data.get("cn_us_rate_spread")),
            ("美国CPI", global_data.get("us_cpi")),
            ("美国核心CPI", global_data.get("us_core_cpi")),
            ("美国非农", global_data.get("us_nonfarm")),
            ("美国失业率", global_data.get("us_unemployment")),
            ("美国ISM PMI", global_data.get("us_ism_pmi")),
            ("全球商品", global_data.get("commodity")),
            ("人民币主要汇率长序列", market.get("fx_boc_safe")),
            ("人民币汇率", market.get("fx")),
            ("外汇储备", macro.get("fx_reserves")),
        ],
        rows=10,
    )

elif page == "行业热点":
    st.markdown("### 行业方向")
    st.write("根据模板映射：弱复苏 + 政策支撑优先关注高股息、央国企、基建、科技主题；风险偏好改善时关注 AI、半导体、创新药、军工。")
    render_source_grid(
        [
            ("东方财富妙想资讯", mx_data.get("mx_search")),
            ("东方财富妙想金融数据", mx_data.get("mx_finance")),
            ("行业资金流", market.get("fund_flow")),
            ("东方财富人气榜", market.get("hot_rank")),
            ("东方财富飙升榜", market.get("hot_up")),
            ("新闻联播政策信号", xinwen_lianbo),
            ("宏观热点", news),
        ],
        rows=16,
    )

elif page == "主要风险":
    st.markdown("### 主要风险")
    st.write("这里是给投资风险管理官和最终报告使用的主要风险数据库。北向资金、外汇、美债、中美利差、地产、估值和盈利数据都会进入 AI 风险判断，不再只是缓存里有数据。")
    risk_cards = [
        ("地产拖累", "看房地产景气和新房价格，判断地产链是否继续压制信用、消费和银行资产质量。"),
        ("外资与北向资金", "优先看 Tushare 沪深港通资金流；东方财富北向数据若净买额为空，只作为行情辅助。"),
        ("人民币与外汇", "看人民币主要汇率长序列、实时汇率和外储，判断汇率压力和外资风险偏好。"),
        ("海外利率与中美利差", "看美国利率、中美国债收益率和美元指数，判断A股估值扩张的外部约束。"),
        ("估值与盈利下修", "看中证指数估值、宽基估值、PPI和工业增加值，判断低估是否伴随盈利下修风险。"),
    ]
    for title, body in risk_cards:
        st.markdown(f"<div class='tech-card' style='margin-bottom:.55rem;'><b>{title}</b><br><span class='muted'>{body}</span></div>", unsafe_allow_html=True)
    render_source_grid(risk_source_items(macro, tushare_data, market, global_data), rows=10)

snapshot = build_ai_snapshot(scores, macro, tushare_data, market, global_data, news, xinwen_lianbo, refresh_log)
prompt = build_prompt(snapshot)

if page == "最终结论":
    st.markdown("### AI 总结报告")
    render_ai_report_status()
    render_cycle_position_curve(metrics, scores)
    with st.expander("基础数据更新时间", expanded=False):
        render_update_status(refresh_log)
    with st.expander("分析逻辑：这份报告怎么推导结论", expanded=True):
        for title, body in [
                ("1. 政策环境", "先判断财政、货币、产业政策是否偏宽松；新闻联播作为政策重心、产业主线和主线切换的重要确认源。"),
            ("2. 经济周期", "用 GDP、PMI、社零、投资、进出口、地产判断经济处于复苏、弱复苏、下行还是过热阶段。"),
            ("3. 流动性", "用 M2、社融、LPR、DR007、北向资金、汇率判断资金条件是否支持估值扩张。"),
            ("4. 企业盈利", "用 PPI、需求指标、地产和工业品价格判断上市公司盈利处于改善、见底或承压。"),
            ("5. 估值水平", "用宽基指数估值、股债收益差和风险溢价判断市场安全边际。"),
            ("6. 外部环境", "用美债、美元、人民币、中美利差、美国通胀/就业/PMI判断外部压力。"),
                ("7. 行业映射", "把宏观组合映射到行业：弱复苏+政策支撑偏高股息/央国企/科技主题，流动性改善偏AI/半导体/创新药，出口改善偏家电/汽车/机械/电子；再用东方财富妙想资讯、板块资金和热点榜核对。"),
        ]:
            st.markdown(f"**{title}**：{body}")
    coverage_df = pd.DataFrame(snapshot["coverage"])
    total_rows = int(coverage_df["rows"].sum()) if not coverage_df.empty else 0
    available_sets = int((coverage_df["rows"] > 0).sum()) if not coverage_df.empty else 0
    proof_cols = st.columns(4)
    proof_cols[0].metric("已传入数据集", available_sets)
    proof_cols[1].metric("传入总行数", total_rows)
    proof_cols[2].metric("分块角色", "7个")
    proof_cols[3].metric("ChatGPT分析", "12小时自动更新")

    st.caption("AI报告全局每12小时自动更新一次；网页刷新不会触发AI分析。后台数据仍每半小时增量更新。")

    if st.button("手动触发一次AI分析", type="primary", width="stretch"):
        if st.session_state.get("ai_generation_running"):
            st.warning("AI分析已经在运行中，请等待当前任务完成。")
            st.stop()
        st.session_state.ai_generation_running = True
        with st.spinner("正在手动触发多角色AI分析。本次完成后会写入报告日志和AI分析过程。"):
            try:
                llm_result = generate_ai_report(title=f"手动触发AI报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                if llm_result.get("report_run_id"):
                    st.session_state.last_report_run_id = llm_result.get("report_run_id")
                if llm_result.get("ok"):
                    st.success(f"手动AI报告生成成功，报告ID：{llm_result.get('report_id')}")
                else:
                    st.warning("手动AI报告生成完成，但最终响应失败，已记录到报告日志。")
                    st.warning(llm_result.get("content", ""))
            except Exception as exc:
                st.error(f"手动AI分析失败：{exc}")
            finally:
                st.session_state.ai_generation_running = False
        st.rerun()

    latest_reports = latest_ai_reports(1)
    if latest_reports.empty:
        st.info("还没有自动AI报告。下方先显示规则引擎汇总；后台12小时调度会生成正式报告。")
        render_overview_reasons(scores, metrics)
    else:
        latest_report_id = int(latest_reports["id"].iloc[0])
        report = load_ai_report(latest_report_id)
        meta_cols = st.columns(4)
        meta_cols[0].metric("报告ID", latest_report_id)
        meta_cols[1].metric("报告状态", report.get("status", ""))
        meta_cols[2].metric("生成时间", report.get("created_at", ""))
        meta_cols[3].metric("模型", report.get("model", ""))
        if report.get("error"):
            st.warning(report["error"])
        st.markdown(report.get("content", ""))
        render_pdf_download(
            "导出当前自动报告为PDF",
            report.get("title", "A股宏观AI报告"),
            report.get("content", ""),
            f"a股宏观AI报告_{latest_report_id}",
        )

    with st.expander("确认：传给 ChatGPT 的数据覆盖清单", expanded=False):
        st.dataframe(coverage_df, width="stretch", hide_index=True, height=420)
    with st.expander("确认：数据被哪些职业角色使用", expanded=False):
        st.dataframe(pd.DataFrame(ROLE_DATA_MAP), width="stretch", hide_index=True, height=300)
    with st.expander("确认：主要风险数据库已传给AI", expanded=False):
        st.json(snapshot.get("risk_evidence_database", {}), expanded=False)
    with st.expander("确认：旧版单次 Prompt 预览（现在实际使用多角色分块）", expanded=False):
        st.code(prompt, language="markdown")

if page == "数据源状态":
    st.markdown("### 数据源状态")
    render_status("宏观", macro)
    render_status("Tushare", tushare_data)
    render_status("市场", market)
    render_status("外部", global_data)
    render_status("东方财富妙想", mx_data)
    st.markdown("### 新闻联播政策信号")
    dataframe_preview(xinwen_lianbo, rows=20, height=360)
    st.markdown("### 热点")
    dataframe_preview(news, rows=20, height=360)
    st.markdown("### 增量更新记录")
    st.dataframe(refresh_log, width="stretch", hide_index=True, height=420)

if page == "数据新鲜度":
    freshness_df = build_freshness_table(macro, tushare_data, market, global_data, mx_data, news, xinwen_lianbo, refresh_log)
    render_freshness_page(freshness_df)

if page == "AI分析过程":
    st.markdown("### AI分析过程")
    st.caption("这里显示最近一次12小时自动分析或手动触发分析的多职业角色分块结果。本页面不自动刷新，也不会触发AI分析。")

    chunks_index = latest_ai_chunks(200)
    if chunks_index.empty:
        st.info("还没有分块分析记录。系统会每12小时自动生成，也可以到「最终结论」手动触发一次。")
    else:
        run_ids = chunks_index["report_run_id"].drop_duplicates().tolist()
        default_run = st.session_state.get("last_report_run_id")
        default_index = run_ids.index(default_run) if default_run in run_ids else 0
        selected_run = st.selectbox("选择分析批次", run_ids, index=default_index)
        chunks = load_ai_chunks(selected_run, limit=50)
        completed = int((chunks["status"] == "ok").sum()) if not chunks.empty else 0
        failed = int((chunks["status"] == "error").sum()) if not chunks.empty else 0
        expected_roles = 9
        cols = st.columns(4)
        cols[0].metric("分析批次", selected_run)
        cols[1].metric("已完成角色", f"{completed}/{expected_roles}")
        cols[2].metric("失败角色", failed)
        cols[3].metric("最新更新时间", chunks["created_at"].max() if not chunks.empty else "暂无")

        if not chunks.empty:
            st.dataframe(
                chunks[["id", "created_at", "chunk_name", "status", "prompt_hash", "error"]],
                width="stretch",
                hide_index=True,
                height=300,
            )
            st.markdown("### 角色输出")
            for _, row in chunks.iterrows():
                title = f"#{row['id']} {row['chunk_name']}｜{row['status']}｜{row['created_at']}"
                with st.expander(title, expanded=row["chunk_name"] in ("首席策略官", "投研质控总监")):
                    if row.get("error"):
                        st.warning(row["error"])
                    st.markdown(row.get("content") or "暂无输出")

if page == "报告日志":
    st.markdown("### AI报告日志")
    reports = latest_ai_reports(500)
    if reports.empty:
        st.info("还没有保存过AI报告。系统会每12小时自动生成。")
    else:
        st.dataframe(reports, width="stretch", hide_index=True, height=320)
        ids = reports["id"].tolist()
        selected = st.selectbox("查看报告", ids, format_func=lambda rid: f"#{rid} - {reports.loc[reports['id'] == rid, 'title'].iloc[0]}")
        report = load_ai_report(int(selected))
        if report:
            st.markdown(f"### {report['title']}")
            st.caption(f"生成时间：{report['created_at']}｜模型：{report['model']}｜状态：{report['status']}｜Prompt Hash：{report['prompt_hash']}")
            if report.get("error"):
                st.warning(report["error"])
            st.markdown(report.get("content", ""))
            render_pdf_download(
                "导出这份报告为PDF",
                report.get("title", "A股宏观AI报告"),
                report.get("content", ""),
                f"a股宏观AI报告_{report.get('id')}",
            )
