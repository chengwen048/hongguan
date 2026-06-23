from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


DB_PATH = Path("data/macro_history.sqlite3")
OUT_DIR = Path("output/pdf")
ROLE_ORDER = [
    "政策研究员",
    "宏观经济学家",
    "流动性与利率交易员",
    "权益估值分析师",
    "全球宏观策略师",
    "行业配置策略师",
    "投资风险管理官",
    "首席策略官",
]
KEY_DATASETS = [
    ("tushare_gdp", "Tushare GDP"),
    ("tushare_pmi", "Tushare PMI"),
    ("tushare_cpi", "Tushare CPI"),
    ("tushare_ppi", "Tushare PPI"),
    ("tushare_m2", "Tushare M2"),
    ("tushare_social_financing", "Tushare 社融"),
    ("tushare_hsgt_moneyflow", "Tushare 北向资金"),
    ("fx_boc_safe", "人民币主要汇率长序列"),
    ("fx_reserves", "外汇储备"),
    ("cn_us_rate_spread", "中美国债收益率"),
    ("csindex_valuation", "中证指数估值"),
    ("real_estate", "房地产景气"),
    ("new_house_price", "新房价格"),
    ("xinwen_lianbo", "新闻联播政策信号"),
    ("mx_search", "东方财富妙想资讯"),
]


def html_escape(text: object) -> str:
    value = "" if text is None else str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def clean_markdown_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return html_escape(text).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")


def split_cjk_text(text: object) -> str:
    value = "" if text is None else str(text)
    if len(value) <= 18:
        return value
    chunks = []
    current = ""
    for ch in value:
        current += ch
        if len(current) >= 18:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return "\n".join(chunks)


def register_fonts() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("ReportCN", path))
            return "ReportCN"
    raise SystemExit("未找到可嵌入的中文字体")


def build_styles(font_name: str):
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=22,
            leading=30,
            textColor=colors.HexColor("#0B1F33"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#075985"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#0F766E"),
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#1F2937"),
            alignment=TA_LEFT,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#475569"),
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def fetch_one(conn: sqlite3.Connection, query: str, params: tuple = ()) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    row = conn.execute(query, params).fetchone()
    if row is None:
        raise SystemExit(f"未找到数据：{query} {params}")
    return row


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(query, params).fetchall()


def latest_dataset_meta(conn: sqlite3.Connection, dataset: str) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS rows_count, MAX(updated_at) AS updated_at
        FROM observations
        WHERE dataset = ?
        """,
        (dataset,),
    ).fetchone()
    sample = conn.execute(
        """
        SELECT payload_json
        FROM observations
        WHERE dataset = ?
        ORDER BY row_key DESC
        LIMIT 1
        """,
        (dataset,),
    ).fetchone()
    payload = json.loads(sample["payload_json"]) if sample else {}
    date_value = ""
    for key in ("date", "日期", "trade_date", "抓取日期", "quarter", "报告期"):
        if payload.get(key):
            date_value = str(payload.get(key))
            break
    latest_value = ""
    for key in ("value", "今值", "最新值", "north_money", "美国国债收益率10年", "市盈率1"):
        if payload.get(key) not in (None, ""):
            latest_value = str(payload.get(key))
            break
    return {
        "rows": row["rows_count"] if row else 0,
        "updated_at": row["updated_at"] if row else "",
        "latest_date": date_value,
        "latest_value": latest_value,
    }


def make_table(data: list[list[object]], styles: dict, col_widths: list[float] | None = None) -> Table:
    cells = []
    for r, row in enumerate(data):
        style = styles["table_header"] if r == 0 else styles["table"]
        cells.append([Paragraph(clean_markdown_inline(split_cjk_text(cell)), style) for cell in row])
    table = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C81")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def markdown_blocks(markdown: str, styles: dict) -> Iterable:
    lines = markdown.splitlines()
    i = 0
    paragraph: list[str] = []

    def flush_para():
        nonlocal paragraph
        if not paragraph:
            return []
        text = "<br/>".join(clean_markdown_inline(line.strip()) for line in paragraph if line.strip())
        paragraph = []
        return [Paragraph(text, styles["body"])] if text else []

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            for item in flush_para():
                yield item
            yield Spacer(1, 3)
            i += 1
            continue
        if line.lstrip().startswith("|") and "|" in line:
            for item in flush_para():
                yield item
            table_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tline in table_lines:
                parts = [part.strip() for part in tline.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", part or "") for part in parts):
                    continue
                rows.append(parts)
            if rows:
                col_count = max(len(row) for row in rows)
                rows = [row + [""] * (col_count - len(row)) for row in rows]
                page_width = A4[0] - 30 * mm
                yield make_table(rows, styles, [page_width / col_count] * col_count)
                yield Spacer(1, 8)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        numbered_heading = re.match(r"^(\d+\.\s+.+)$", line)
        if heading or numbered_heading:
            for item in flush_para():
                yield item
            text = heading.group(2) if heading else numbered_heading.group(1)
            yield Paragraph(clean_markdown_inline(text), styles["h1"] if heading and len(heading.group(1)) <= 2 or numbered_heading else styles["h2"])
        else:
            paragraph.append(line)
        i += 1
    for item in flush_para():
        yield item


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("ReportCN", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(15 * mm, 10 * mm, "A股宏观环境分析报告")
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf(report_id: int, output: Path | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    report = fetch_one(
        conn,
        "SELECT * FROM ai_reports WHERE id = ?",
        (report_id,),
    )
    run_row = fetch_one(
        conn,
        """
        SELECT report_run_id
        FROM ai_chunks
        WHERE prompt_hash = ? AND chunk_name = '投研质控总监'
        ORDER BY id DESC
        LIMIT 1
        """,
        (report["prompt_hash"],),
    )
    run_id = run_row["report_run_id"]
    chunks = fetch_all(
        conn,
        """
        SELECT *
        FROM ai_chunks
        WHERE report_run_id = ?
        ORDER BY id
        """,
        (run_id,),
    )
    chunk_by_name = {row["chunk_name"]: row for row in chunks}
    coverage = json.loads(report["coverage_json"] or "[]")
    output = output or OUT_DIR / f"a股宏观分析完整报告_{report_id}_{run_id}.pdf"

    font = register_fonts()
    styles = build_styles(font)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"A股宏观分析完整报告 {report_id}",
        author="本地A股宏观分析终端",
    )
    story = []
    story.append(Paragraph("A股宏观环境分析完整PDF", styles["title"]))
    story.append(Paragraph(f"报告ID：{report['id']}｜分析批次：{run_id}", styles["body"]))
    story.append(Paragraph(f"生成时间：{report['created_at']}｜模型：{report['model']}｜状态：{report['status']}", styles["body"]))
    story.append(Paragraph(f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("一、总览", styles["h1"]))
    latest_refresh = fetch_all(
        conn,
        """
        SELECT created_at, group_name, dataset, status, rows_seen, rows_inserted, rows_updated
        FROM refresh_log
        ORDER BY id DESC
        LIMIT 12
        """,
    )
    total_rows = sum(int(item.get("rows", 0)) for item in coverage)
    available_sets = sum(1 for item in coverage if int(item.get("rows", 0)) > 0)
    meta_table = [
        ["项目", "内容"],
        ["报告标题", report["title"]],
        ["数据覆盖", f"{available_sets} 个有效数据集 / 传入总行数 {total_rows}"],
        ["角色输出", f"{len(chunks)} 个分块输出，其中本PDF采用 {len([x for x in ROLE_ORDER if x in chunk_by_name])} 个分析/汇总角色"],
        ["最近数据刷新", latest_refresh[0]["created_at"] if latest_refresh else "暂无"],
    ]
    story.append(make_table(meta_table, styles, [36 * mm, A4[0] - 66 * mm]))
    story.append(Spacer(1, 8))

    key_rows = [["数据集", "本地行数", "最近数据日期", "最新值/样本", "最近入库时间"]]
    for dataset, label in KEY_DATASETS:
        meta = latest_dataset_meta(conn, dataset)
        key_rows.append([label, meta["rows"], meta["latest_date"], meta["latest_value"], meta["updated_at"]])
    story.append(Paragraph("关键数据快照", styles["h2"]))
    story.append(make_table(key_rows, styles, [34 * mm, 21 * mm, 28 * mm, 33 * mm, 44 * mm]))
    story.append(Spacer(1, 8))

    cov_rows = [["分组", "数据集", "指标", "行数"]]
    for item in coverage[:40]:
        cov_rows.append([item.get("group", ""), item.get("dataset", ""), item.get("name", ""), item.get("rows", "")])
    story.append(Paragraph("传入AI的数据覆盖清单", styles["h2"]))
    story.append(make_table(cov_rows, styles, [24 * mm, 42 * mm, 62 * mm, 18 * mm]))

    story.append(PageBreak())
    story.append(Paragraph("二、所有分析师分块输出", styles["h1"]))
    for role in ROLE_ORDER:
        row = chunk_by_name.get(role)
        if not row:
            continue
        story.append(Paragraph(f"{role}｜{row['status']}｜{row['created_at']}", styles["h2"]))
        for block in markdown_blocks(row["content"] or "", styles):
            story.append(block)
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("三、最终报告", styles["h1"]))
    for block in markdown_blocks(report["content"] or "", styles):
        story.append(block)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = build_pdf(args.report_id, args.output)
    print(path)


if __name__ == "__main__":
    main()
