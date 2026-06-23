from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from reporting import generate_ai_report

load_dotenv(".env")

REPORT_INTERVAL_SECONDS = 12 * 60 * 60
AI_REPORT_LOCK = Path("data/ai_report.lock")


def run_once() -> None:
    title = f"A股宏观环境自动报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] 开始生成AI报告", flush=True)
    try:
        result = generate_ai_report(title)
        status = "成功" if result.get("ok") else "失败"
        print(f"[{datetime.now().isoformat(timespec='seconds')}] AI报告{status} report_id={result.get('report_id')}", flush=True)
    except Exception as exc:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] AI报告异常：{exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=REPORT_INTERVAL_SECONDS)
    parser.add_argument("--wait-first", action="store_true", help="启动后先等待一个周期，再生成报告")
    args = parser.parse_args()
    try:
        if args.wait_first and not args.once:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] AI报告调度器已启动，先等待 {args.interval} 秒", flush=True)
            time.sleep(args.interval)
        while True:
            run_once()
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        AI_REPORT_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
