from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_sources import (
    load_global_data,
    load_hot_news,
    load_macro_data,
    load_market_data,
    load_mx_finance_data,
    load_mx_search_data,
    load_tushare_data,
    load_xinwen_lianbo,
    save_result_groups,
)

load_dotenv(".env")

REFRESH_SECONDS = 30 * 60
GROUP_TIMEOUT_SECONDS = int(os.getenv("UPDATE_GROUP_TIMEOUT_SECONDS", "300"))
UPDATE_LOCK = Path("data/update.lock")


def task_fetcher(group_name: str):
    token = os.getenv("TUSHARE_TOKEN", "")
    http_url = os.getenv("TUSHARE_HTTP_URL", "http://8.163.90.143:8686/")
    tasks = {
        "macro": lambda: load_macro_data(),
        "tushare": lambda: load_tushare_data(token, http_url) if token else {},
        "market": lambda: load_market_data(),
        "global": lambda: load_global_data(),
        "xinwen_lianbo": lambda: load_xinwen_lianbo(),
        "mx": lambda: {"mx_search": load_mx_search_data(), "mx_finance": load_mx_finance_data()},
        "news": lambda: load_hot_news(),
    }
    return tasks[group_name]


def task_names() -> list[str]:
    # Put the fast, high-value sources first so one slow bulk market/macro API
    # cannot delay 新闻联播、Tushare、东方财富妙想的增量落库.
    return ["tushare", "xinwen_lianbo", "mx", "news", "macro", "market", "global"]


def refresh_group(group_name: str) -> dict[str, dict[str, int]]:
    group_start = datetime.now().isoformat(timespec="seconds")
    print(f"[{group_start}] 更新 {group_name}...", flush=True)
    data = task_fetcher(group_name)()
    stats = save_result_groups({group_name: data})
    inserted = sum(item["inserted"] for item in stats.values())
    updated = sum(item["updated"] for item in stats.values())
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {group_name} 完成：新增 {inserted} 行，更新 {updated} 行", flush=True)
    return stats


def clean_subprocess_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def refresh_once() -> dict[str, dict[str, int]]:
    all_stats: dict[str, dict[str, int]] = {}
    for group_name in task_names():
        cmd = [sys.executable, str(Path(__file__).resolve()), "--group", group_name]
        try:
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, text=True, capture_output=True, timeout=GROUP_TIMEOUT_SECONDS)
            if result.stdout:
                print(result.stdout.rstrip(), flush=True)
            if result.stderr:
                print(result.stderr.rstrip(), flush=True)
            if result.returncode != 0:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] {group_name} 子进程失败：退出码 {result.returncode}", flush=True)
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                print(clean_subprocess_text(exc.stdout).rstrip(), flush=True)
            if exc.stderr:
                print(clean_subprocess_text(exc.stderr).rstrip(), flush=True)
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {group_name} 超过 {GROUP_TIMEOUT_SECONDS} 秒，已跳过并继续下一组", flush=True)
        except Exception as exc:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {group_name} 调度失败：{exc}", flush=True)
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只更新一次")
    parser.add_argument("--interval", type=int, default=REFRESH_SECONDS, help="循环更新间隔秒数")
    parser.add_argument("--group", choices=task_names(), help="只更新指定分组，供父进程隔离调用")
    args = parser.parse_args()

    if args.group:
        try:
            refresh_group(args.group)
        except Exception as exc:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {args.group} 失败：{exc}", flush=True)
            raise SystemExit(1)
        return

    try:
        while True:
            start = datetime.now().isoformat(timespec="seconds")
            print(f"[{start}] 开始增量更新", flush=True)
            try:
                stats = refresh_once()
                inserted = sum(item["inserted"] for item in stats.values())
                updated = sum(item["updated"] for item in stats.values())
                print(f"[{datetime.now().isoformat(timespec='seconds')}] 完成：新增 {inserted} 行，更新 {updated} 行", flush=True)
            except Exception as exc:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] 更新失败：{exc}", flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        UPDATE_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
