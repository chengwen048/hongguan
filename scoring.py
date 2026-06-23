from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ModuleScore:
    label: str
    score: int
    conclusion: str


def latest_number(df: pd.DataFrame, col: str = "value") -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def latest_date(df: pd.DataFrame, col: str = "date") -> str:
    if df is None or df.empty or col not in df.columns:
        return "暂无"
    value = df[col].iloc[-1]
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def trend(df: pd.DataFrame, col: str = "value", window: int = 3) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < window + 1:
        return 0.0
    return float(s.iloc[-window:].mean() - s.iloc[-window - 1 : -1].mean())


def classify_environment(metrics: dict[str, Any]) -> dict[str, ModuleScore]:
    gdp = metrics.get("gdp")
    pmi = metrics.get("pmi")
    cpi = metrics.get("cpi")
    ppi = metrics.get("ppi")
    m2 = metrics.get("m2")
    social_financing = metrics.get("social_financing")
    pe = metrics.get("market_pe")
    equity_risk_premium = metrics.get("equity_risk_premium")

    policy_score = 1
    if metrics.get("lpr_1y_down"):
        policy_score += 1
    if metrics.get("rrr_down"):
        policy_score += 1
    policy_label = "偏宽松" if policy_score >= 2 else "中性偏宽"

    growth_score = 0
    growth_score += 1 if gdp is not None and gdp >= 5 else -1
    growth_score += 1 if pmi is not None and pmi >= 50 else -1
    growth_score += 1 if metrics.get("retail_trend", 0) >= 0 else -1
    cycle_label = "复苏" if growth_score >= 1 else "弱复苏 / 筑底"

    liquidity_score = 0
    liquidity_score += 1 if m2 is not None and m2 >= 7 else -1
    liquidity_score += 1 if social_financing is not None and social_financing >= 8 else -1
    liquidity_score += 1 if metrics.get("rate_trend", 0) <= 0 else -1
    liquidity_label = "宽松" if liquidity_score >= 1 else "中性"

    earnings_score = 0
    earnings_score += 1 if ppi is not None and ppi >= -1 else -1
    earnings_score += 1 if metrics.get("industrial_profit_trend", 0) >= 0 else -1
    earnings_label = "改善" if earnings_score >= 1 else "承压"

    valuation_score = 0
    valuation_score += 1 if pe is not None and pe <= 18 else 0
    valuation_score += 1 if equity_risk_premium is not None and equity_risk_premium >= 3 else 0
    valuation_label = "低估" if valuation_score >= 1 else "中性"

    risk_score = growth_score + liquidity_score + valuation_score
    risk_label = "修复中" if risk_score >= 1 else "偏谨慎"

    return {
        "policy": ModuleScore(policy_label, policy_score, f"政策环境{policy_label}，重点仍在稳增长、扩内需、防风险与科技创新。"),
        "cycle": ModuleScore(cycle_label, growth_score, f"经济周期处于{cycle_label}，GDP/PMI 是核心验证项。"),
        "liquidity": ModuleScore(liquidity_label, liquidity_score, f"流动性环境{liquidity_label}，关注 M2、社融、LPR、DR007 和基金发行。"),
        "earnings": ModuleScore(earnings_label, earnings_score, f"企业盈利{earnings_label}，重点观察 PPI、工业利润和财报趋势。"),
        "valuation": ModuleScore(valuation_label, valuation_score, f"估值水平{valuation_label}，结合 PE/PB 和股债收益差判断安全边际。"),
        "risk": ModuleScore(risk_label, risk_score, f"市场风险偏好{risk_label}，受政策预期、外部环境和热点主线共同影响。"),
    }
