# Autogen 智能助手系统 - 项目文档介绍

## 📘 简介

本项目是一个基于 [Autogen](https://github.com/microsoft/autogen) 框架的多智能体协作系统，构建了一个面向 iPhone 手机销售场景的对话代理团队。通过引入模型流式对话、多代理任务分工、任务状态持久化、数据库交互和图结构流程控制，项目实现了一个具有丰富功能的智能问答系统。

## 👤 面向对象

本项目主要适用于以下人群：

- AI 应用开发者
- LLM 多智能体系统研究人员
- 自动化对话系统开发者
- 教学或课程自测评分场景

## 📂 项目功能总览

- ✅ 支持多个 Agent 协同工作（销售、信息查询、售后）
- ✅ 支持工具调用（库存查询、价格查询）
- ✅ 支持多种团队协作模式（轮询、选择器、Swarm、图结构流程）
- ✅ 支持流式响应显示（Streaming）
- ✅ 支持任务状态保存与恢复（包括 JSON 文件与数据库 MySQL）
- ✅ 支持任务信息持久化存储（MySQL）
- ✅ 支持 Agent Memory 能力（记忆用户偏好）

## 🧠 主要组件说明

### 模型客户端配置

使用 `OpenAIChatCompletionClient` 指定自定义 API 接口，如 Qwen 模型：

```python
model_client = OpenAIChatCompletionClient(
    model="Qwen/QwQ-32B",
    api_key="...",
    base_url="https://api.siliconflow.cn/v1",
    ...
)
```

### 智能体定义

- `begin_agent`：前端销售助手，负责新品介绍与兴趣引导，具备handoff能力
- `info_agent`：信息助手，调用工具查询库存与价格
- `after_agent`：售后服务助手，处理用户售后问题

### 工具调用

定义了两个异步工具函数：

```python
async def query_stock(model: str) -> str
async def query_price(model: str) -> str
```

供 `info_agent` 使用以返回查询结果。

### 团队协作模式

项目演示了以下团队协作结构：

- `SelectorGroupChat`：基于发言策略自动选择 Agent
- `Swarm`：所有 Agent 并发处理任务，自我激活
- `GraphFlow`：基于图结构的顺序执行 Agent（例如：begin → info → after）

### 终止条件

组合使用：

```python
termination = SourceMatchTermination([...]) | TextMentionTermination("终止")
```

以控制任务终止。

## 💾 状态管理功能

### JSON 文件存储

支持通过 `team.save_state()` 与 `load_state()` 保存/加载任务状态：

```python
with open("agent_state.json", "w") ...
await team.load_state(...)
```

### 数据库管理（MySQL）

封装类 `DBManager`：

- `create_task()` 创建任务记录
- `update_task_status()` 更新任务状态
- `add_message()` 插入任务对话信息
- `get_messages_for_task()` 查询历史记录

## ✅ 示例任务运行

任务示例：推荐 iPhone 机型，查询库存价格，展示售后流程

```python
await Console(team.run_stream(
    task="我想了解一下iPhone17和iPhone16Pro，哪个更值得买？有现货吗？售后怎么样？"
))
```

## 🧠 Agent Memory 示例

可扩展 `memory_agent`，集成 `ListMemory` 组件，支持用户信息记忆（例如天气单位偏好、饮食偏好等）。

## 🚀 启动方式

```bash
python test_agent_stream.py
```

## 📝 文件结构说明

| 文件                   | 描述                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `test_agent_stream.py` | 主逻辑脚本，包含模型接入、Agent 定义、团队协作结构与状态保存 |
| `agent_state.json`     | 任务状态保存文件（运行中自动生成）                           |
| MySQL 数据库           | 用于持久化存储任务及消息记录                                 |

## 📌 注意事项

- 请替换 `model_client` 中的 API Key 和模型名称为你自己的。
- MySQL 需预先创建 `llm_test` 数据库，自动建表。
- 如部署到服务器，可使用 Streamlit、Gradio 等框架进行封装展示。

---

## 📮 联系与反馈

如需进一步优化或对接 Web 页面、API，请联系开发者或提交 issue。
