"""SQLAlchemy 数据模型 - Paper Pipeline V2.0"""

from sqlalchemy import Column, String, DateTime, Integer, Float, Text, Boolean, Index, create_engine
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()


class Paper(Base):
    """论文主表"""
    __tablename__ = "papers"
    
    id = Column(String(50), primary_key=True)  # arXiv ID
    title = Column(String(500), nullable=False)
    abstract = Column(Text, nullable=False)
    categories = Column(JSON)  # ['cs.AI', 'cs.LG']
    authors = Column(JSON)  # [{'name': '...', 'affiliation': '...'}]
    published_date = Column(DateTime)
    updated_date = Column(DateTime)
    
    # 质量指标
    citation_count = Column(Integer, default=0)
    influential_citation = Column(Integer, default=0)
    
    # LLM深度处理
    llm_score = Column(Float)  # 1-10相关性评分
    llm_summary = Column(Text)  # 一句话概括
    llm_insight = Column(Text)  # 核心创新点+局限性
    llm_tags = Column(JSON)  # ['推理优化', 'MoE']
    
    # 状态管理
    is_read = Column(Boolean, default=False)
    is_saved = Column(Boolean, default=False)
    
    # 元数据
    pdf_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_published', 'published_date'),
        Index('idx_score', 'llm_score'),
    )


class PaperAnalysis(Base):
    """论文深度分析表"""
    __tablename__ = "paper_analyses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(50), nullable=False)
    
    # 结构化分析
    problem = Column(Text)  # 解决的问题
    method = Column(Text)  # 核心方法
    architecture = Column(JSON)  # 架构组成
    experiments = Column(JSON)  # 实验设置
    results = Column(JSON)  # 实验结果
    
    # 代码分析
    github_repo = Column(String(500))
    code_quality = Column(Float)  # 代码质量评分
    reproducibility = Column(Float)  # 可复现性评分
    dependencies = Column(JSON)  # 依赖列表
    
    # 改进点
    strengths = Column(JSON)  # 优点列表
    weaknesses = Column(JSON)  # 缺点列表
    improvements = Column(JSON)  # 改进建议
    
    # 复现结果
    reproduction_status = Column(String(50))  # pending/running/completed/failed
    reproduction_metrics = Column(JSON)  # 复现指标
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Experiment(Base):
    """实验记录表"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(50), nullable=False)
    
    # 实验信息
    name = Column(String(200))
    description = Column(Text)
    environment = Column(JSON)  # {gpu, cuda, python_version, etc.}
    
    # 结果
    status = Column(String(50))  # pending/running/completed/failed
    metrics = Column(JSON)  # {accuracy, latency, memory, etc.}
    artifacts = Column(JSON)  # 输出文件列表
    
    # 对比
    paper_results = Column(JSON)  # 论文报告结果
    comparison = Column(JSON)  # 对比分析
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class Topic(Base):
    """研究领域表"""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    keywords = Column(JSON)  # 搜索关键词
    categories = Column(JSON)  # ArXiv分类
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    """通知记录表"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50))  # email/wechat/telegram
    subject = Column(String(500))
    content = Column(Text)
    status = Column(String(50))  # pending/sent/failed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)


# 数据库初始化
def init_db(db_path=None):
    """初始化数据库"""
    if db_path is None:
        db_path = Path("/root/git/mimo/paper-pipeline/data/pipeline.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    session = init_db()
    print("数据库初始化完成")
    print("表:", [table for table in Base.metadata.tables.keys()])
