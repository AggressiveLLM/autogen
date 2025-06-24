# test_agent_stream.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from autogen_agentchat.conditions import TextMentionTermination, SourceMatchTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
import json
from autogen_agentchat.teams import Swarm
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
import mysql.connector
import datetime
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
# model_client = OpenAIChatCompletionClient(
#     model="gpt-4o",
#     # api_key="YOUR_API_KEY",
# )
model_client = OpenAIChatCompletionClient(
    model="Qwen/QwQ-32B",
    api_key="sk-obamsnsjkryughmonriiqmrdomvnkqoarrajpilucpiojvlm",
    base_url="https://api.siliconflow.cn/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
        "multiple_system_messages": True,
    },
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
# 先定义工具
async def query_stock(model: str) -> str:
    fake_db = {
        "iPhone16Pro": "有货",
        "iPhone17": "紧张，仅剩5台"
    }
    return f"{model} 库存情况：{fake_db.get(model, '无库存信息')}"

async def query_price(model: str) -> str:
    fake_price = {
        "iPhone16Pro": "￥7999",
        "iPhone17": "￥8999"
    }
    return f"{model} 售价：{fake_price.get(model, '未知')}"

# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
# begin_agent = AssistantAgent(
#     name="begin_agent",
#     model_client=model_client,
#     description="You are an iPhone front-sales assistant",
#     system_message="你是iphone手機的前端銷售，向用戶推銷新款的iPhone17手機",
#     reflect_on_tool_use=True,
#     model_client_stream=True,  # Enable streaming tokens from the model client.
# )
begin_agent = AssistantAgent(
    name="begin_agent",
    model_client=model_client,
    description="iPhone前端销售，负责新品推介和用户引导",
    handoffs=["info_agent", "after_agent"],
    system_message=(
        "你是iPhone手机的前端销售，负责向用户介绍iPhone17新款手机的卖点，引导用户兴趣。"
        "当用户需要查询库存或价格时，必须用如下格式把请求交接给info_agent：\n"
        "@info_agent 我要查询iPhone17的库存和价格。\n"
        "注意：必须以@info_agent开头进行handoff交接。"
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)

after_agent = AssistantAgent(
    name="after_agent",
    model_client=model_client,
    description="You are an iPhone after-sales assistant",
    handoffs=["user"],
    system_message=(
        "你是iPhone手机的售后助手，负责解答用户关于售后服务的问题。\n"
        "注意：当你回答结束后，必须要：“如果没有其他问题，那就这样咯，感谢咨询”并结束。\n"
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

# 2. 信息助理，负责查询库存和价格，调用工具，汇总信息
info_agent = AssistantAgent(
    name="info_agent",
    model_client=model_client,
    description="信息助理，负责查询指定型号手机的库存和价格",
    system_message=(
        "你是信息助理，收到型号和请求后，调用query_stock和query_price工具查询库存和价格，"
        "完成后必须用@after_agent 格式把结果交给 after_agent。"
        "例如：@after_agent iPhone17 有货，价格￥8999。"
    ),
    tools=[query_stock, query_price],
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# user_memory = ListMemory()

# async def get_weather(city: str, units: str = "imperial") -> str:
#     if units == "imperial":
#         return f"The weather in {city} is 73 °F and Sunny."
#     elif units == "metric":
#         return f"The weather in {city} is 23 °C and Sunny."
#     else:
#         return f"Sorry, I don't know the weather in {city}."

#     # Add user preferences to memory
#     await user_memory.add(MemoryContent(content="The weather should be in metric units", mime_type=MemoryMimeType.TEXT))

#     await user_memory.add(MemoryContent(content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT))

# memory_agent = AssistantAgent(
#     name="memory_agent",
#     model_client=model_client,
#     description="你需要记住用户所有的输入信息",
#     tools=[get_weather],
#     memory=[user_memory],
# )

# 結束條件 -------------------------------------------------------------
# Test - 1 
# termination1 = SourceMatchTermination(["after_agent"])
# termination2 = SourceMatchTermination(["begin_agent"])
# termination = termination1 or termination2
# Test - 2
termination1 = SourceMatchTermination(["after_agent" , "begin_agent" , "info_agent"])
termination2 = TextMentionTermination("终止")
termination = termination1 or termination2


#-Step-1 
# team = RoundRobinGroupChat([weather_agent, rednote_agent], termination_condition=text_termination)
#-Step-2
team = SelectorGroupChat(
            [begin_agent, after_agent, info_agent],
            model_client=model_client,
            termination_condition=termination,
            # selector_prompt=selector_prompt,
            allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)

#  User-Proxy ------------------------------------------------------------------------
# user_proxy = UserProxyAgent(
#     name="user_proxy",
#     human_input_mode="NEVER",
#     model_client=model_client,
#     team=team,
# )
class DBManager:
    def __init__(self, host="localhost", user="root", password="123456", database="llm_test"):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.create_tables()

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            task_name VARCHAR(255),
            status VARCHAR(50),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        ''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            task_id INT,
            sender VARCHAR(50),
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        )
        ''')
        self.conn.commit()
        cur.close()

    def create_task(self, task_name, status="pending"):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO tasks (task_name, status) VALUES (%s, %s)", (task_name, status))
        self.conn.commit()
        task_id = cur.lastrowid
        cur.close()
        return task_id

    def update_task_status(self, task_id, new_status):
        cur = self.conn.cursor()
        cur.execute("UPDATE tasks SET status=%s, updated_at=%s WHERE id=%s", (new_status, datetime.datetime.now(), task_id))
        self.conn.commit()
        cur.close()

    def get_task(self, task_id):
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM tasks WHERE id=%s", (task_id,))
        result = cur.fetchone()
        cur.close()
        return result

    def add_message(self, task_id, sender, content):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO messages (task_id, sender, content) VALUES (%s, %s, %s)", (task_id, sender, content))
        self.conn.commit()
        message_id = cur.lastrowid
        cur.close()
        return message_id

    def get_messages_for_task(self, task_id):
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM messages WHERE task_id=%s ORDER BY created_at", (task_id,))
        results = cur.fetchall()
        cur.close()
        return results
async def main():
    # ++++++  Get Grade 60 (自测） ----------------------------------------------------------------------
    # 实现一个RobinGroup团队，功能自选。  （实现√）
    # 至少包含3个Agent，需使用工具调用。  （实现√）
    # 团队需展示协作效果。    （实现√）
    await Console(team.run_stream(task="幫我分析一下我买iphone16pro好还是iphone17好，给我一个建议"))  # Stream the messages to the console.
    
    # ++++++  Get Grade 70 （自测） ----------------------------------------------------------------------
    # 实现单个任务状态的保存与加载功能。 （实现√）
    # 需支持任务中断后恢复。  （实现√）
    # sava state ---------------------------------------------------------------------------
    agent_state = await team.save_state()
    with open("agent_state.json", "w", encoding="utf-8") as f:
        json.dump(agent_state, f, ensure_ascii=False, indent=2, default=str)
    print(" Agent 状态已保存到 agent_state.json")
    
    # load state ----------------------------------------------------------------------
    
    with open("agent_state.json", "r", encoding="utf-8") as f:
        team_state = json.load(f)

    new_team = SelectorGroupChat(
        [begin_agent, after_agent],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )

    await new_team.load_state(team_state)

    stream = new_team.run_stream(task="請問上一個回答你說到了什麼？")
    await Console(stream)
    await model_client.close()

    # ++++++  Get Grade 80 （自测）----------------------------------------------------------------------
    # 实现一个Swarm团队，功能自选。（实现√）
    # 至少包含3个Agent，需使用工具调用。->  {begin_agent, info_agent, after_agent} （实现√）
    # 团队需展示分布式协作能力。 （实现√）
    # Swarm-Team ----------------------------------------------------------------------

    # ========== 终止条件 ==========
    termination = SourceMatchTermination(["after_agent"]) | TextMentionTermination("感谢咨询")
    

    # ========== Swarm团队 ==========
    team = Swarm([begin_agent, info_agent, after_agent], termination_condition=termination)
    
    await Console(team.run_stream(
        task="我想了解一下iPhone17和iPhone16Pro，哪个更值得买？有现货吗？售后怎么样？"
    ))

    # ++++++  Get Grade 90 （自测）----------------------------------------------------------------------
    # Q1 实现官方GraphFlow团队接口，完成官方案例即可。  （实现√）
    # GraphFlow -> 按照图结构执行各 Agent，控制数据流与任务顺序
    # 基于本项目，我们一共设立了3节点，begin_agent, info_agent, after_agent ； 链接顺序为 begin_agent -> info_agent -> after_agent
    # 构建GraphFlow图
    builder = DiGraphBuilder()
    builder.add_node(begin_agent).add_node(info_agent).add_node(after_agent)
    builder.add_edge(begin_agent, info_agent)
    builder.add_edge(info_agent, after_agent)
    graph = builder.build()

    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )
    await Console(flow.run_stream(task="我想了解一下iPhone17和iPhone16Pro，哪个更值得买？有现货吗？售后怎么样？"))
    
    # Q2 数据库状态管理
    # 使用数据库管理任务历史状态与消息。  （实现√）
    # 需支持查询与更新。  （实现√）
    db = DBManager()
    # 创建一个任务
    task_id = db.create_task("iPhone 选购咨询", status="in_progress")

    # 开始运行
    result = await Console(team.run_stream(task="我想了解一下iPhone17和iPhone16Pro，哪个更值得买？有现货吗？售后怎么样？"))

    # 保存消息
    db.add_message(task_id, "user", "我想了解一下iPhone17和iPhone16Pro，哪个更值得买？有现货吗？售后怎么样？")
    db.add_message(task_id, "assistant", str(result))
    # 更新状态
    db.update_task_status(task_id, "completed")

    #  Q3 Agent Memory能力
    # 根据官方文档为单Agent添加Memory能力。
    # 需展示记忆功能效果。

    # 预估得分95分 
  



if __name__ == "__main__":
    asyncio.run(main())

# Run the agent and stream the messages to the console.
# async def main() -> None:
#     await Console(agent.run_stream(task="What is the weather in New York?"))
#     # Close the connection to the model client.
#     await model_client.close()


# # NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
# asyncio.run(main()) 

# 运行指令 - python D:\桌面\test_agent_stream.py
