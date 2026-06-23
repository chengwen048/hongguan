from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def build_prompt(snapshot: dict[str, Any]) -> str:
    return f"""
你是一名中文A股宏观策略分析师。请严格基于下面的数据快照，按用户提供的“A股宏观环境分析模板”输出A股宏观情况。

要求：
- 不要编造没有给出的数据。
- 必须综合所有模块的数据覆盖情况、近五年趋势、最新值和热点信息。
- 必须重点使用“新闻联播政策信号”和“东方财富妙想”数据：新闻联播用于判断政策重心、产业主线和主线切换，东方财富妙想用于核对行情、板块、资讯与资金证据。
- 必须使用“主要风险数据库”：北向资金、人民币汇率/外汇储备、美债/中美利差、地产、估值和盈利下修证据都要进入第10节风险判断。
- 如果数据源之间冲突，要指出冲突和你采用的判断口径。
- 对每个模块给出：结论、依据、风险、后续观察指标。
- 必须分成“过去怎么看、现在处于什么阶段、未来 1-2 个季度怎么看”。
- 报告最开头必须先给出“经济周期曲线定位”：把当前宏观位置标注为繁荣期/衰退期/萧条期/复苏期中的哪一段，最好细分为“萧条末期/复苏初期”等，并说明为什么。
- 过去部分要写清楚：上一阶段/上一轮行情主线是什么，主线从哪里开始变化，变化背后的宏观、政策、流动性或产业原因是什么。
- 热点行业必须有理有据，但不要给太多方向，最终只保留 4-5 个最值得关注的行业方向。
- 必须在报告最后单独给出“我需要重点关注的行业”，用表格列出 4-5 个方向：行业方向、关注级别、为什么关注、对应数据证据、触发条件、风险。
- 必须单独给出“A股市场主线推演”：上一轮主线是什么、当前主线是什么、当前主线大概能维持多久、未来可能切换到什么主线、主线切换的验证信号。
- 最后给出“当前A股宏观环境总评”和“行业配置方向”。
- 语言要像券商宏观策略报告，中文，条理清晰。
- 输出尽量详细，目标约 3000 字，先给结论，再给证据；不要为了凑字重复空话。

数据快照：
{snapshot}

请按以下结构输出：
0. 经济周期曲线定位：当前处于繁荣/衰退/萧条/复苏的哪一段，并写清为什么
1. 一句话结论
2. 过去：近五年宏观与A股环境怎么演变
3. 过去主线复盘：上一轮主线是什么，何时发生变化，变化原因和当时行情特征
4. 现在：政策、新闻联播政策信号、经济、流动性、盈利、估值、风险偏好分别处于什么阶段，并说明为什么
5. 未来：未来 1-2 个季度的基准情景、乐观情景、风险情景
6. A股市场主线推演：上一轮主线、当前主线、当前主线持续时间判断、未来主线、切换信号
7. 热点行业方向：只列出 4-5 个方向，每个方向说明宏观证据、产业/资金/热点证据、验证指标、风险
8. 我需要重点关注的行业：用表格给出最终清单
9. 主要风险与后续观察清单
10. 最终结论：分行写清楚“当前A股宏观环境总评”“当前主线”“下一阶段关注方向”
""".strip()


def build_role_prompt(role_name: str, role_instruction: str, data: dict[str, Any]) -> str:
    return f"""
你是一名中文A股投研团队成员，当前角色：{role_name}。

任务：
- 只基于给定数据，不编造。
- 输出 350-500 字左右，必须有理有据，不能只给观点。
- 每轮分析必须按“结论 -> 支撑证据 -> 反证/不足 -> 后续观察指标”的逻辑写清楚。
- 支撑证据必须尽量点名数据来源或指标，例如：Tushare、AKShare、新闻联播、东方财富妙想、PMI、社融、PPI、汇率、资金流、热点榜等。
- 如果证据不足，只能给“观察/低置信度”结论，并说明缺少哪些数据。
- 如果数据不足，明确写“数据不足”并说明还需要什么。
- 语言要像券商策略会内部纪要，清晰、直接、有判断。

角色职责：
{role_instruction}

数据：
{data}
""".strip()


def build_synthesis_prompt(role_outputs: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> str:
    return f"""
你是A股宏观策略首席分析师。下面是投研团队多个角色基于本地五年宏观数据库、新闻联播、东方财富妙想、Tushare、AKShare等数据分别形成的分块结论。

要求：
- 只综合这些角色结论，不编造新数据。
- 必须检查主要风险数据库是否被风险管理官使用；第10节必须覆盖北向资金、人民币汇率/外汇储备、美债/中美利差、地产、估值和盈利下修。
- 必须指出不同角色之间是否有冲突，以及最终采用的判断口径。
- 输出 900-1300 字的“首席策略官汇总备忘录”，不要写成长篇完整报告。
- 必须按照用户原始“A股宏观环境分析模板”的栏目输出，不要改成自由发挥格式。
- 必须写清楚：过去主线是什么、何时/因为什么变化；当前主线是什么；大概能维持多久；未来可能主线是什么；切换信号是什么。
- 最开头必须先写“经济周期曲线定位”，判断当前处于繁荣期/衰退期/萧条期/复苏期的哪一段，建议细分为“萧条末期/复苏初期”等，并给出证据。
- 最后必须给出“我需要重点关注的行业”Markdown 表格，必须且只能 4-5 行，列名固定为：行业方向、关注级别、当前主线/未来主线、为什么关注、对应证据、触发条件、风险。
- 最终结论放最后，并分行写清楚“当前A股宏观环境总评”“当前主线”“下一阶段关注方向”。
- 每个小节最多 3 条要点，避免空话和重复。

数据覆盖：
{coverage}

角色结论：
{role_outputs}

请严格按以下模板输出：
0. 经济周期曲线定位
- 当前位于周期曲线的哪个阶段：繁荣期 / 衰退期 / 萧条期 / 复苏期，必要时细分为前段/中段/末段
- 为什么：结合GDP、PMI、CPI/PPI、M2/社融、利率、地产、估值、风险偏好
- 对资产配置的含义

1. 当前A股宏观环境总评
- 政策环境：偏大规模 / 中性 / 偏紧，并写原因
- 经济周期：复苏 / 弱复苏 / 下行 / 过热，并写原因
- 流动性环境：宽松 / 中性 / 偏紧，并写原因
- 企业盈利：改善 / 见底 / 承压，并写原因
- 估值水平：低估 / 合理 / 偏高，并写原因
- 市场风险偏好：提升 / 中性 / 回落，并写原因
- 综合判断：当前A股处于什么宏观阶段

2. 政策环境
- 财政政策
- 货币政策：降准、降息、LPR、MLF、公开市场操作
- 产业政策：人工智能、新质生产力、半导体、高端制造、消费、地产等
- 新闻联播政策信号
- 当前政策重点：稳定增长、扩内需、防风险、促进改革、支持科技创新
- 结论

3. 经济基础
- GDP、PMI、社零、固定资产投资、出口、CPI/PPI、地产销售/投资、失业率
- 过去五年经济阶段变化
- 结论模板：当前经济处于什么阶段，为什么

4. 流动性环境
- M2、社融、LPR、DR007/资金利率、北向资金、人民币汇率、新发基金/资金风险偏好
- 结论

5. 企业盈利
- 全A盈利、沪深300/创业板/周期盈利线索、PPI、工业品价格、毛利率、财报趋势
- 结论

6. 估值水平
- 沪深300、中证500、中证1000、创业板指、科创50、万得全A/风险溢价
- 结论

7. 外部环境
- 美联储、美债收益率、美元指数、人民币汇率、中美关系、大宗商品
- 结论

8. A股市场主线推演
- 上一次主线是什么
- 主线在哪里/什么时候变化了
- 当前主线是什么
- 当前主线大概能维持多久
- 未来可能主线是什么
- 主线切换验证信号

9. 行业配置方向
- 根据宏观环境映射行业方向
- 热点行业证据：东方财富妙想、资金流、人气榜、新闻联播、宏观数据
- 必须用 Markdown 表格列出“我需要重点关注的行业”，表格必须且只能 4-5 行，列名固定如下：
  | 行业方向 | 关注级别 | 当前主线/未来主线 | 为什么关注 | 对应证据 | 触发条件 | 风险 |
- 表格必须体现“上一次主线/当前主线/未来主线”的关系：属于当前主线、未来主线、还是上一轮主线退潮后的观察方向。
- 关注级别只能用：高 / 中高 / 中 / 观察。

10. 主要风险
- 经济修复不及预期、地产、政策落地、外资/北向资金、人民币/外汇储备、美债/中美利差、上市公司盈利、估值回撤等

11. 最终结论
- 当前A股宏观环境总评
- 当前主线
- 下一阶段关注方向
- 需要继续观察的数据
""".strip()


def build_quality_review_prompt(draft_report: str, role_outputs: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> str:
    compact_roles = [
        {
            "role": item.get("role"),
            "status": item.get("status"),
            "content": (str(item.get("content") or "")[:260] + "...") if len(str(item.get("content") or "")) > 260 else item.get("content"),
        }
        for item in role_outputs[:9]
    ]
    compact_coverage = [
        {key: item.get(key) for key in ("dataset", "name", "rows", "latest") if key in item}
        for item in coverage[:18]
    ]
    return f"""
你是投研质控总监，负责检查A股宏观策略报告并输出最终版。

你的任务：
- 检查报告是否只基于角色结论和数据覆盖，不得编造新数据。
- 检查是否覆盖：当前A股宏观环境总评、过去主线、当前主线、当前主线持续时间、未来主线、切换信号、重点关注行业、风险。
- 检查报告最开头是否有“经济周期曲线定位”，并明确当前处于繁荣/衰退/萧条/复苏的哪一段；如缺失，必须在最终版最前面补上。
- 检查第10节是否使用主要风险数据库，至少覆盖：北向资金、人民币汇率/外汇储备、美债/中美利差、地产、估值和盈利下修；缺失则补齐并标注证据不足处。
- 检查第 9 节是否包含 Markdown 行业配置推荐表。如果没有表格，或表格不是 4-5 行，或列名不是“行业方向、关注级别、当前主线/未来主线、为什么关注、对应证据、触发条件、风险”，必须在最终版中重写第 9 节。
- 如发现角色之间结论冲突，要在最终版中明确说明并给出采用口径。
- 如发现数据不足，要在最终版中标注“数据不足/需继续观察”，不要硬下结论。
- 输出最终可读报告，目标约1800-2300字，结构清楚，最终结论放最后。
- 最终版必须保持用户原始“A股宏观环境分析模板”结构：总评、政策、经济基础、流动性、盈利、估值、外部环境、市场主线、行业配置、主要风险、最终结论。
- 第 9 节“行业配置方向”必须包含且只能包含 4-5 行 Markdown 表格：
  | 行业方向 | 关注级别 | 当前主线/未来主线 | 为什么关注 | 对应证据 | 触发条件 | 风险 |
- 表格中必须写清每个行业为什么值得关注、对应的宏观/政策/资金/新闻联播或东方财富妙想证据、触发条件和风险，不允许只写口号。
- 不允许把最终版改写成聊天式回答或纯摘要。

数据覆盖摘要：
{compact_coverage}

各职业角色核心结论摘要：
{compact_roles}

首席策略官汇总草稿：
{draft_report}

请直接输出最终版报告，并严格使用以下一级标题：
0. 经济周期曲线定位
1. 当前A股宏观环境总评
2. 政策环境
3. 经济基础
4. 流动性环境
5. 企业盈利
6. 估值水平
7. 外部环境
8. A股市场主线推演
9. 行业配置方向
10. 主要风险
11. 最终结论
""".strip()


def call_llm_with_meta(prompt: str, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or not base_url:
        return {"ok": False, "content": "未配置大语言模型接口，请在 .env 中设置 OPENAI_API_KEY 与 OPENAI_BASE_URL。", "model": model, "base_url": base_url}

    started_at = datetime.now().isoformat(timespec="seconds")
    import requests

    timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "300"))
    retry_attempts = max(1, int(os.getenv("OPENAI_RETRY_ATTEMPTS", "4")))
    retry_seconds = float(os.getenv("OPENAI_RETRY_SECONDS", "8"))
    min_interval = float(os.getenv("OPENAI_MIN_INTERVAL_SECONDS", "2.5"))
    base_urls = llm_base_urls(base_url)
    payload = {
        "model": model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": "你是严谨的中文A股宏观策略分析师，只基于给定数据做推理。"},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 macro-dashboard/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last_error = ""
    for candidate_base in base_urls:
        endpoint = f"{candidate_base}/chat/completions"
        for attempt in range(1, retry_attempts + 1):
            if min_interval > 0:
                time.sleep(min_interval)
            session = requests.Session()
            if os.getenv("OPENAI_DISABLE_PROXY", "1").lower() not in ("0", "false", "no"):
                session.trust_env = False
            try:
                response = session.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
            except requests.RequestException as exc:
                last_error = f"LLM接口请求失败：{exc}"
                if attempt < retry_attempts:
                    time.sleep(retry_seconds * attempt)
                    continue
                break
            if response.status_code >= 400:
                last_error = f"LLM接口返回 {response.status_code}: {clean_llm_error(response.text)}"
                if response.status_code in (403, 408, 429, 500, 502, 503, 504) and attempt < retry_attempts:
                    time.sleep(retry_seconds * attempt)
                    continue
                break
            try:
                data = response.json()
            except ValueError:
                last_error = f"LLM接口未返回JSON: {clean_llm_error(response.text)}"
                if attempt < retry_attempts:
                    time.sleep(retry_seconds * attempt)
                    continue
                break
            if "error" in data:
                last_error = str(data["error"])
                if attempt < retry_attempts:
                    time.sleep(retry_seconds * attempt)
                    continue
                break
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content.strip():
                last_error = "LLM接口返回空内容"
                if attempt < retry_attempts:
                    time.sleep(retry_seconds * attempt)
                    continue
                break
            return {
                "ok": True,
                "content": content,
                "model": model,
                "base_url": candidate_base,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "response_id": data.get("id", ""),
                "usage": data.get("usage", {}),
                "attempts": attempt,
            }
    return {
        "ok": False,
        "content": last_error or "LLM接口请求失败：未知错误",
        "model": model,
        "base_url": ",".join(base_urls),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def call_llm(prompt: str, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> str:
    return call_llm_with_meta(prompt, api_key=api_key, base_url=base_url, model=model)["content"]


def clean_llm_error(text: str, limit: int = 360) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "空响应"
    if "Attention Required" in raw and "Cloudflare" in raw:
        return "Cloudflare 拦截了服务器请求。通常是当前云服务器 IP 被接口站点风控，需要更换接口出口、让服务商放行服务器 IP，或改用不会拦截服务器请求的 OpenAI 兼容接口。"
    if "<html" in raw.lower() or "<!doctype" in raw.lower():
        title = re.search(r"<title>(.*?)</title>", raw, flags=re.I | re.S)
        if title:
            return re.sub(r"\s+", " ", title.group(1)).strip()
        return "接口返回HTML页面，不是JSON API响应。请检查 OPENAI_BASE_URL 是否为 /v1 API 地址，或接口站点是否拦截服务器请求。"
    return raw[:limit]


def llm_base_urls(primary: str) -> list[str]:
    values = [primary]
    extra = os.getenv("OPENAI_BASE_URLS", "")
    if extra:
        values.extend(part.strip() for part in extra.split(",") if part.strip())
    out = []
    for value in values:
        cleaned = value.rstrip("/")
        if not cleaned:
            continue
        if cleaned.endswith("/chat/completions"):
            cleaned = cleaned[: -len("/chat/completions")]
        candidates = [cleaned]
        if not cleaned.endswith("/v1"):
            candidates.insert(0, f"{cleaned}/v1")
        for candidate in candidates:
            if candidate not in out:
                out.append(candidate)
    return out
