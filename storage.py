from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

DB_PATH = Path("data/macro_history.sqlite3")

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
    "created_at",
    "observed_at",
    "updated_at",
)

VALUE_CANDIDATES = (
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


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            dataset TEXT NOT NULL,
            source TEXT NOT NULL,
            row_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (dataset, row_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_seen INTEGER NOT NULL,
            rows_inserted INTEGER NOT NULL,
            rows_updated INTEGER NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            coverage_json TEXT,
            usage_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_run_id TEXT NOT NULL,
            chunk_name TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            coverage_json TEXT,
            usage_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _row_key(row: dict) -> str:
    for date_key in ("date", "日期", "trade_date", "ann_date", "end_date", "quarter", "报告期", "抓取日期"):
        if date_key not in row or pd.isna(row[date_key]):
            continue
        for text_key in ("标题", "title", "内容", "content", "name", "名称", "代码", "symbol", "指数", "指数代码", "指数中文简称", "货币对", "pair", "currency"):
            if text_key in row and pd.notna(row[text_key]):
                return f"{row[date_key]}::{row[text_key]}"
    for key in ("date", "日期", "trade_date", "ann_date", "end_date", "quarter", "报告期", "抓取日期"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _records(df: pd.DataFrame) -> Iterable[tuple[str, str]]:
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    for record in clean.to_dict(orient="records"):
        yield _row_key(record), json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    normalized = (
        text.str.replace("Q1", "-03-01", regex=False)
        .str.replace("Q2", "-06-01", regex=False)
        .str.replace("Q3", "-09-01", regex=False)
        .str.replace("Q4", "-12-01", regex=False)
    )
    zh_month = normalized.str.extract(r"(?P<year>\d{4})年(?P<month>\d{1,2})月")
    has_zh_month = zh_month["year"].notna()
    normalized.loc[has_zh_month] = zh_month.loc[has_zh_month, "year"] + "-" + zh_month.loc[has_zh_month, "month"].str.zfill(2) + "-01"
    mask8 = normalized.str.fullmatch(r"\d{8}")
    normalized.loc[mask8] = normalized.loc[mask8].str[:4] + "-" + normalized.loc[mask8].str[4:6] + "-" + normalized.loc[mask8].str[6:8]
    mask6 = normalized.str.fullmatch(r"\d{6}")
    normalized.loc[mask6] = normalized.loc[mask6].str[:4] + "-" + normalized.loc[mask6].str[4:6] + "-01"
    return pd.to_datetime(normalized, errors="coerce")


def _enhance_loaded_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    lower_cols = {str(col).lower(): col for col in out.columns}
    date_col = next((col for col in DATE_CANDIDATES if col in out.columns), None)
    if date_col is None:
        date_col = next((lower_cols[col.lower()] for col in DATE_CANDIDATES if col.lower() in lower_cols), None)
    if date_col:
        parsed = _parse_mixed_dates(out[date_col])
        if parsed.notna().sum():
            if "date" in out.columns:
                out["date"] = parsed
            else:
                out.insert(0, "date", parsed)
    if "value" not in out.columns:
        value_col = next((col for col in VALUE_CANDIDATES if col in out.columns and col != "value"), None)
        if value_col is None:
            numeric_cols = [col for col in out.columns if col != "date" and pd.to_numeric(out[col], errors="coerce").notna().sum() > 0]
            value_col = numeric_cols[0] if numeric_cols else None
        if value_col:
            value = pd.to_numeric(out[value_col], errors="coerce")
            if value.notna().sum():
                out["value"] = value
    return out


def save_result(group_name: str, dataset: str, result) -> dict[str, int]:
    now = datetime.now().isoformat(timespec="seconds")
    rows_seen = 0 if result is None or result.data.empty else len(result.data)
    inserted = 0
    updated = 0
    status = "error" if getattr(result, "error", None) else "ok"
    error = getattr(result, "error", None)

    with connect() as conn:
        if result is not None and not result.data.empty:
            for key, payload in _records(result.data):
                old = conn.execute(
                    "SELECT payload_json FROM observations WHERE dataset = ? AND row_key = ?",
                    (dataset, key),
                ).fetchone()
                if old is None:
                    inserted += 1
                    conn.execute(
                        """
                        INSERT INTO observations
                        (dataset, source, row_key, payload_json, observed_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (dataset, result.source, key, payload, now, now),
                    )
                elif old[0] != payload:
                    updated += 1
                    conn.execute(
                        """
                        UPDATE observations
                        SET source = ?, payload_json = ?, updated_at = ?
                        WHERE dataset = ? AND row_key = ?
                        """,
                        (result.source, payload, now, dataset, key),
                    )
        conn.execute(
            """
            INSERT INTO refresh_log
            (group_name, dataset, source, status, rows_seen, rows_inserted, rows_updated, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_name,
                dataset,
                getattr(result, "source", ""),
                status,
                rows_seen,
                inserted,
                updated,
                error,
                now,
            ),
        )
    return {"seen": rows_seen, "inserted": inserted, "updated": updated}


def load_dataset(dataset: str, limit: int = 5000) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT payload_json, observed_at, updated_at
            FROM observations
            WHERE dataset = ?
            ORDER BY row_key DESC
            LIMIT ?
            """,
            (dataset, limit),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for payload_json, observed_at, updated_at in rows:
        record = json.loads(payload_json)
        record.setdefault("observed_at", observed_at)
        record.setdefault("updated_at", updated_at)
        records.append(record)
    df = _enhance_loaded_dataset(pd.DataFrame(records))
    lower_cols = {str(col).lower(): col for col in df.columns}
    for candidate in DATE_CANDIDATES:
        col = candidate if candidate in df.columns else lower_cols.get(candidate.lower())
        if col:
            parsed = _parse_mixed_dates(df[col])
            if parsed.notna().sum():
                return df.assign(_sort_date=parsed).sort_values("_sort_date").drop(columns=["_sort_date"])
    return df


def load_dataset_recent(dataset: str, days: int = 370, limit: int = 5000) -> pd.DataFrame:
    df = load_dataset(dataset, limit=limit)
    if df.empty:
        return df
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
    lower_cols = {str(col).lower(): col for col in df.columns}
    for candidate in DATE_CANDIDATES:
        col = candidate if candidate in df.columns else lower_cols.get(candidate.lower())
        if not col:
            continue
        parsed = _parse_mixed_dates(df[col])
        if parsed.notna().sum() == 0:
            continue
        recent = df.loc[parsed >= cutoff].copy()
        return recent if not recent.empty else df.head(24)
    return df.head(500)


def latest_refresh(limit: int = 80) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT created_at, group_name, dataset, source, status, rows_seen, rows_inserted, rows_updated, error
            FROM refresh_log
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def save_ai_report(title: str, model: str, prompt_hash: str, content: str, status: str, error: str | None = None, coverage=None, usage=None) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_reports
            (title, model, prompt_hash, content, status, error, coverage_json, usage_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                model,
                prompt_hash,
                content,
                status,
                error,
                json.dumps(coverage or [], ensure_ascii=False, default=_json_default),
                json.dumps(usage or {}, ensure_ascii=False, default=_json_default),
                now,
            ),
        )
        return int(cur.lastrowid)


def keep_latest_ai_report_only() -> None:
    with connect() as conn:
        latest = conn.execute("SELECT id FROM ai_reports ORDER BY id DESC LIMIT 1").fetchone()
        if latest:
            conn.execute("DELETE FROM ai_reports WHERE id <> ?", (latest[0],))
        latest_run = conn.execute("SELECT report_run_id FROM ai_chunks ORDER BY id DESC LIMIT 1").fetchone()
        if latest_run:
            conn.execute("DELETE FROM ai_chunks WHERE report_run_id <> ?", (latest_run[0],))


def save_ai_chunk(report_run_id: str, chunk_name: str, model: str, prompt_hash: str, content: str, status: str, error: str | None = None, coverage=None, usage=None) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_chunks
            (report_run_id, chunk_name, model, prompt_hash, content, status, error, coverage_json, usage_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_run_id,
                chunk_name,
                model,
                prompt_hash,
                content,
                status,
                error,
                json.dumps(coverage or [], ensure_ascii=False, default=_json_default),
                json.dumps(usage or {}, ensure_ascii=False, default=_json_default),
                now,
            ),
        )
        return int(cur.lastrowid)


def latest_ai_chunks(limit: int = 100) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT id, created_at, report_run_id, chunk_name, model, prompt_hash, status, error
            FROM ai_chunks
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def load_ai_chunks(report_run_id: str | None = None, limit: int = 100) -> pd.DataFrame:
    with connect() as conn:
        if report_run_id:
            return pd.read_sql_query(
                """
                SELECT id, created_at, report_run_id, chunk_name, model, prompt_hash, content, status, error, usage_json
                FROM ai_chunks
                WHERE report_run_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                conn,
                params=(report_run_id, limit),
            )
        return pd.read_sql_query(
            """
            SELECT id, created_at, report_run_id, chunk_name, model, prompt_hash, content, status, error, usage_json
            FROM ai_chunks
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def latest_ai_reports(limit: int = 50) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT id, created_at, title, model, prompt_hash, status, error
            FROM ai_reports
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def load_ai_report(report_id: int) -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, title, model, prompt_hash, content, status, error, coverage_json, usage_json
            FROM ai_reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()
    if not row:
        return {}
    keys = ["id", "created_at", "title", "model", "prompt_hash", "content", "status", "error", "coverage_json", "usage_json"]
    return dict(zip(keys, row))
