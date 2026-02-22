# RememberMe

通过聊天记录复活记忆中的人。

导入你和某个人的聊天记录，AI 会学习 TA 的说话风格、口头禅、语气习惯，然后你可以像以前一样和 TA 聊天。

## 工作原理

```
聊天记录导入 → 人格特征提取 → 记忆向量化(RAG) → LLM 模拟对话
```

- **人格分析**：从聊天记录中提取说话风格、口头禅、emoji 习惯、语气词、连发模式、活跃时段等
- **记忆检索**：所有聊天记录存入向量数据库，对话时检索最相关的历史片段作为上下文
- **连发模拟**：真人不会一次回一大段，TA 可能连发 3-5 条短消息，AI 也会这样做
- **对话引擎**：Gemini 3.1 Pro 驱动，结合人格档案 + RAG 记忆生成回复

## 快速开始

### 安装

```bash
git clone https://github.com/ryusaksun/RememberMe.git
cd RememberMe
uv sync
```

### 配置

创建 `.env` 文件：

```bash
GEMINI_API_KEY=你的Gemini_API_Key
```

### 使用

#### 1. 导入聊天记录

**纯文本格式**（`名字: 消息内容`）：

```bash
uv run remember-me import-chat chat.txt --format text --target "小明"
```

**微信导出格式**：

```bash
uv run remember-me import-chat wechat_export.txt --format wechat --target "小明"
```

**JSON 格式**：

```bash
uv run remember-me import-chat chat.json --format json --target "小明"
```

**网易云音乐私信**（在线导入）：

```bash
uv run remember-me import-netease
```

首次使用需要提供网易云音乐的 `MUSIC_U` cookie（[获取方法](#获取网易云-music_u-cookie)）。

#### 2. 开始对话

```bash
uv run remember-me chat 小明
```

#### 3. 查看已有人格

```bash
uv run remember-me list-personas
```

## 支持的聊天记录格式

| 格式 | 命令参数 | 说明 |
|------|---------|------|
| 纯文本 | `--format text` | `名字: 消息内容`，每行一条 |
| JSON | `--format json` | `[{"sender": "名字", "content": "内容", "timestamp": "..."}]` |
| 微信导出 | `--format wechat` | 微信聊天记录导出的文本格式 |
| 网易云音乐 | `import-netease` | 通过 API 在线拉取私信记录，支持图片下载 |

## 人格分析维度

从聊天记录中提取的特征：

- 说话长度、短消息比例
- 口头禅 / 高频短语
- Emoji 使用习惯
- 语气词（哈哈、嘛、吧、啊...）
- 句尾特征
- 连发消息模式（平均连发几条、连发概率分布）
- 活跃时段
- 打招呼 / 告别方式
- 话题偏好关键词
- 真实连发对话样例（用于 few-shot 学习）

## 数据存储

所有数据保存在 `data/` 目录（已 gitignore）：

```
data/
├── history/    # 完整聊天记录 (JSON)
├── profiles/   # 人格档案 (JSON)
├── chroma/     # 向量数据库 (RAG 检索)
└── images/     # 聊天图片
```

## 获取网易云 MUSIC_U Cookie

1. 浏览器打开 [music.163.com](https://music.163.com) 并登录
2. 按 `F12` 打开开发者工具
3. 切换到 **Application** 标签页
4. 左侧 **Cookies** → `https://music.163.com`
5. 找到 `MUSIC_U` 那一行，复制值

可以写入 `.env` 避免每次输入：

```bash
echo 'NETEASE_COOKIE=你的MUSIC_U值' >> .env
```

## 技术栈

- **语言**：Python 3.11+
- **LLM**：Google Gemini 3.1 Pro
- **向量数据库**：ChromaDB（本地，all-MiniLM-L6-v2 embedding）
- **CLI**：Click + Rich
- **网易云 API**：自实现 weapi 加密协议（AES + RSA）

## License

MIT
