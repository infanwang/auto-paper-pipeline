#!/usr/bin/env python3
"""Email briefing sender for paper pipeline."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

SMTP_HOST = "smtp.163.com"
SMTP_PORT = 465
SENDER = "cl0udp1k@163.com"
AUTH_CODE = "CKnX5kKGxzU3VkVT"
RECIPIENT = "cl0udp1k@163.com"


def send_daily_briefing(report_path, repro_summary_path=None):
    """Send daily paper briefing email."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    report = Path(report_path).read_text("utf-8") if Path(report_path).exists() else "No report today."
    
    repro_section = ""
    if repro_summary_path and Path(repro_summary_path).exists():
        repro_section = f"\n\n---\n\n{Path(repro_summary_path).read_text('utf-8')}"
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 AI论文日报 {today}"
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    
    plain = f"{report}{repro_section}"
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    
    html = f"""<html><head><meta charset="utf-8"><style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
h2 {{ color: #16213e; }}
h3 {{ color: #0f3460; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f8f9fa; }}
a {{ color: #0f3460; }}
</style></head><body>
<pre style="white-space:pre-wrap;font-family:inherit;">{plain}</pre>
</body></html>"""
    msg.attach(MIMEText(html, "html", "utf-8"))
    
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER, AUTH_CODE)
        server.sendmail(SENDER, RECIPIENT, msg.as_string())
        print(f"Daily briefing sent to {RECIPIENT}")


def send_weekly_summary(reports_dir):
    """Send weekly summary of all daily reports."""
    today = datetime.now().strftime("%Y-%m-%d")
    reports_dir = Path(reports_dir)
    
    weekly = f"# AI论文周报 {today}\n\n"
    total_papers = 0
    
    for report_file in sorted(reports_dir.glob("daily_*.md"))[-7:]:
        content = report_file.read_text("utf-8")
        weekly += f"\n---\n\n{content}\n"
        # Count papers
        lines = content.split("\n")
        for line in lines:
            if line.startswith("| ") and "arxiv.org" in line:
                total_papers += 1
    
    weekly = f"# AI论文周报 {today}\n\n> 本周共收录 {total_papers} 篇论文\n\n" + weekly
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 AI论文周报 {today}"
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    
    msg.attach(MIMEText(weekly, "plain", "utf-8"))
    
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER, AUTH_CODE)
        server.sendmail(SENDER, RECIPIENT, msg.as_string())
        print(f"Weekly summary sent to {RECIPIENT}")


if __name__ == "__main__":
    import sys
    today = datetime.now().strftime("%Y-%m-%d")
    
    if len(sys.argv) > 1 and sys.argv[1] == "weekly":
        send_weekly_summary("/root/git/mimo/paper-pipeline/reports")
    else:
        report = f"/root/git/mimo/paper-pipeline/reports/daily_{today}.md"
        repro = f"/root/git/mimo/paper-pipeline/reports/{today}/reproduction/repro_summary.md"
        send_daily_briefing(report, repro)
