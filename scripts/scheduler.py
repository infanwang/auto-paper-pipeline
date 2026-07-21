#!/usr/bin/env python3
"""Main scheduler - orchestrates the paper pipeline."""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from paper_crawler import crawl_all_topics
from paper_analyzer import generate_daily_report
from reproducer import generate_repro_summary
from email_briefing import send_daily_briefing, send_weekly_summary

DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")
REPORTS_DIR = Path("/root/git/mimo/paper-pipeline/reports")


def daily_run():
    """Run daily paper pipeline."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"Paper Pipeline Daily Run: {today}")
    print(f"{'='*60}\n")
    
    # Step 1: Crawl papers
    print("[1/4] Crawling papers...")
    papers = crawl_all_topics()
    
    if not papers:
        print("No papers found. Exiting.")
        return
    
    # Step 2: Generate report
    print("\n[2/4] Generating daily report...")
    report_path = generate_daily_report(papers, today)
    
    # Step 3: Generate reproduction guides for top 5
    print("\n[3/4] Generating reproduction guides...")
    repro_summary_path, repro_paths = generate_repro_summary(papers[:5])
    
    # Step 4: Send email
    print("\n[4/4] Sending email briefing...")
    send_daily_briefing(report_path, repro_summary_path)
    
    print(f"\n{'='*60}")
    print(f"Daily run complete! {len(papers)} papers processed.")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")


def weekly_run():
    """Run weekly summary."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nWeekly summary: {today}")
    send_weekly_summary(REPORTS_DIR)


def backup_run():
    """Backup all data."""
    import shutil
    today = datetime.now().strftime("%Y-%m-%d")
    backup_dir = Path(f"/root/git/mimo/paper-pipeline/backups/{today}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup data
    if DATA_DIR.exists():
        shutil.copytree(DATA_DIR, backup_dir / "data", dirs_exist_ok=True)
    
    # Backup reports
    if REPORTS_DIR.exists():
        shutil.copytree(REPORTS_DIR, backup_dir / "reports", dirs_exist_ok=True)
    
    print(f"Backup created: {backup_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: scheduler.py <daily|weekly|backup>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "daily":
        daily_run()
    elif cmd == "weekly":
        weekly_run()
    elif cmd == "backup":
        backup_run()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: scheduler.py <daily|weekly|backup>")
