"""邮件通知器 - V2.0"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Dict


class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self):
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.163.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "465"))
        self.sender = os.environ.get("SMTP_SENDER", "")
        self.auth_code = os.environ.get("SMTP_AUTH_CODE", "")
        self.recipient = os.environ.get("SMTP_RECIPIENT", "")
    
    def send(self, subject: str, body: str, html_body: str = None) -> bool:
        """发送邮件"""
        if not self.sender or not self.auth_code:
            print("邮件配置不完整，跳过发送")
            return False
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = self.recipient
        
        msg.attach(MIMEText(body, "plain", "utf-8"))
        
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        try:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as s:
                s.login(self.sender, self.auth_code)
                s.sendmail(self.sender, self.recipient, msg.as_string())
            return True
        except Exception as e:
            print(f"邮件发送失败: {e}")
            return False
    
    def send_daily_briefing(self, papers: List[Dict], date: str) -> bool:
        """发送每日简报"""
        # 生成文本版本
        body = self._generate_text_briefing(papers, date)
        
        # 生成HTML版本
        html_body = self._generate_html_briefing(papers, date)
        
        subject = f"📚 AI论文日报 {date}（{len(papers)}篇）"
        
        return self.send(subject, body, html_body)
    
    def _generate_text_briefing(self, papers: List[Dict], date: str) -> str:
        """生成文本简报"""
        # 按领域分组
        by_topic = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            by_topic.setdefault(topic, []).append(p)
        
        lines = [
            f"# AI论文日报 {date}",
            f"",
            f"> 共 {len(papers)} 篇论文",
            f"",
            "## 按领域统计",
            ""
        ]
        
        for topic, topic_papers in by_topic.items():
            lines.append(f"- {topic}: {len(topic_papers)}篇")
        
        lines.extend(["", "## Top 10 论文", ""])
        
        for i, p in enumerate(papers[:10], 1):
            title = p.get('title', 'N/A')[:60]
            authors_list = p.get('authors', [])
            if authors_list:
                if isinstance(authors_list[0], dict):
                    authors = ', '.join([a.get('name', 'N/A') for a in authors_list[:2]])
                else:
                    authors = ', '.join(authors_list[:2])
            else:
                authors = 'N/A'
            lines.append(f"{i}. {title}")
            lines.append(f"   ArXiv: {p.get('id', 'N/A')} | 日期: {p.get('published_date', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_briefing(self, papers: List[Dict], date: str) -> str:
        """生成HTML简报"""
        by_topic = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            by_topic.setdefault(topic, []).append(p)
        
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f8f9fa; }}
</style></head><body>
<h1>📚 AI论文日报 {date}</h1>
<p>共 {len(papers)} 篇论文</p>

<h2>按领域统计</h2>
<table>
<tr><th>领域</th><th>论文数</th></tr>
"""
        for topic, topic_papers in by_topic.items():
            html += f"<tr><td>{topic}</td><td>{len(topic_papers)}</td></tr>\n"
        
        html += """</table>

<h2>Top 10 论文</h2>
<table>
<tr><th>#</th><th>标题</th><th>作者</th><th>日期</th></tr>
"""
        for i, p in enumerate(papers[:10], 1):
            title = p.get('title', 'N/A')[:50]
            authors = ', '.join([a.get('name', 'N/A') for a in p.get('authors', [])[:2]])
            html += f"<tr><td>{i}</td><td>{title}</td><td>{authors}</td><td>{p.get('published_date', 'N/A')}</td></tr>\n"
        
        html += """</table>
</body></html>"""
        
        return html
