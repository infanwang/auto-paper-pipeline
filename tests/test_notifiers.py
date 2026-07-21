"""单元测试 - 邮件通知器"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notifiers.email import EmailNotifier


class TestEmailNotifier:
    """邮件通知器测试"""
    
    def test_init(self):
        """测试初始化"""
        notifier = EmailNotifier()
        assert notifier.smtp_host == "smtp.163.com"
        assert notifier.smtp_port == 465
    
    @patch.dict('os.environ', {
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_PORT': '587',
        'SMTP_SENDER': 'test@test.com',
        'SMTP_AUTH_CODE': 'test_auth',
        'SMTP_RECIPIENT': 'recipient@test.com'
    })
    def test_init_with_env(self):
        """测试环境变量初始化"""
        notifier = EmailNotifier()
        assert notifier.smtp_host == "smtp.test.com"
        assert notifier.smtp_port == 587
        assert notifier.sender == "test@test.com"
    
    @patch('notifiers.email.smtplib.SMTP_SSL')
    def test_send_success(self, mock_smtp):
        """测试发送成功"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = lambda s: mock_server
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        
        notifier = EmailNotifier()
        notifier.sender = "test@test.com"
        notifier.auth_code = "test_auth"
        notifier.recipient = "recipient@test.com"
        
        result = notifier.send("Test Subject", "Test Body")
        assert result == True
    
    def test_send_no_config(self):
        """测试无配置发送"""
        notifier = EmailNotifier()
        notifier.sender = ""
        notifier.auth_code = ""
        
        result = notifier.send("Test Subject", "Test Body")
        assert result == False
    
    def test_generate_text_briefing(self, sample_papers):
        """测试文本简报生成"""
        notifier = EmailNotifier()
        briefing = notifier._generate_text_briefing(sample_papers, "2026-07-21")
        
        assert "AI论文日报" in briefing
        assert "2026-07-21" in briefing
        assert "5" in briefing  # 5篇论文
    
    def test_generate_html_briefing(self, sample_papers):
        """测试HTML简报生成"""
        notifier = EmailNotifier()
        briefing = notifier._generate_html_briefing(sample_papers, "2026-07-21")
        
        assert "<html>" in briefing
        assert "AI论文日报" in briefing
        assert "<table>" in briefing
