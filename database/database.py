import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# 创建SQLite连接池
url = 'sqlite:///database/ai-labs.db'
engine = create_engine(url=url,
                       poolclass=QueuePool,
                       pool_size=5,
                       max_overflow=10)

conn = engine.connect()
