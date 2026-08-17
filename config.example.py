"""本地基础设施连接信息模板。

复制为 config.py 后填入自己的主机和账号。
config.py 已被 .gitignore 忽略，不要把真实密码提交到仓库。
"""

postgresql = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "change-me",
    "port": 5432,
}
milvus = {
    "host": "127.0.0.1",
    "token": "root:Milvus",
    "port": 19530,
}
