---
title: "LangChain（四万字全网最详细攻略）：彻底搞懂原理并亲手搭建RAG应用"
pubDate: 2026-04-03
description: "这是一篇关于 LangChain 的详细教程。"
---

**🔥个人主页**：[北辰水墨](https://blog.csdn.net/2301_80215560?spm=1010.2135.3001.5343 "北辰水墨")

> 本节内容我们来搞懂检索增强生成技术，并构建基于私有数据的智能问答引擎！
___
### 一、大模型：

#### 1. 模型 的介绍：

通过海量的数据和答案，训练出自己的一套规则。这个规则就是模型。

就如数学中的函数一样，通过输入-> 函数 -> 输出。

模型的关键特征：

（1）只能处理特定的任务

（2）需要大量的数据和答案（标注数据）

（3）参数较少（规则太少--模型能力有限）

#### 2.大模型的介绍：

大模型是基于大规模神经网络实现的。神经网络就是很多的节点，每一个节点只处理一小部分的事情，将结果传递给下一个节点，以此类推！最终将结果返回出来。

这种基于神经网络的大模型，参数众多，可以处理更加复杂的工作。

大模型的特点：

（1）规模巨大，参数众多

（2）通用性强

（3）主要以自监督的方式完成训练

#### 3.大模型的三种接入方式：

（1）API接入：通过HTTP请求调用厂商的模型服务。

（2）本地部署：将模型的文件（权重和配置文件）下载到本地。

（3）SDK接入：厂商会将原生的API接入封装起来，通过Python等进行调用。

#### 4.认识嵌入模型（Embedding Model）：

大模型是通过上文来预测下一个字最有可能是谁。

也就是说大模型需要处理文本，但是计算机天生不适合处理文本，而擅长处理数据。我们需要把文本转换成数据。

我们可以把文本表示出坐标轴上的一个点，通过点与点之间的距离/夹角来表示他们之间的关系。

但是常规的二维空间无法表示那么多的文本。我们需要更加高的维度来表示。

<img
  src="https://i-blog.csdnimg.cn/direct/a5993b29ba1f47bca3d91e7473d9e27f.png"
  referrerPolicy="no-referrer"
  alt=""
/>


话收回来，既然大模型需要用到高维向量，也就是说我们要将文本表示成高维向量。这就要借助嵌入模型。

嵌入模型的接入方式：


<img
  src="https://i-blog.csdnimg.cn/direct/a76ce2e381634c4c8a07d8a9409b241e.png"
  referrerPolicy="no-referrer"
  alt=""
/>
___

### 二、LangChain：

#### 1.我将通过大模型的痛点引出LangChain。

（1）简单的提示词得到的结果经常出现幻觉。

（2）大模型的训练具有截止日期，对于最新消息，大模型无法感知。

（3）大模型集结了全人类的智慧。但是他只会说，而不会自己动手干活。

#### 2\. LangChain（Lang：语言，Chain：链）

LangChain中，将不同的大语言模型，嵌入模型，搜索工具等封装成一个个的组件。都是Runnable对象，之后可以通过管道符将各个组件串联在一起形成链。最后调用链。

-   **统一接口 (`Runnable` Protocol)**：
    
    所有组件都实现了标准的接口方法，使得它们可以以一致的方式被调用：
    
    -   `.invoke(input)`: 同步执行，传入输入，返回结果。
    -   `.stream(input)`: 流式输出，适用于实时生成场景。
    -   `.batch(inputs)`: 批量处理，提高效率。
    -   `.ainvoke()`, `.astream()`, `.abatch()`: 对应的异步方法。
-   **管道操作符 (`|`)**：
    
    这是 LCEL 最优雅的特性。你可以使用管道符 `|` 将多个 `Runnable` 对象串联起来，形成一个复杂的链（Chain）。
    
    -   **数据流向**：前一个组件的输出会自动作为下一个组件的输入。
    -   **声明式编程**：你只需要定义“数据如何流动”，而不需要编写繁琐的中间变量赋值代码。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
 
# 1. 初始化各个组件 (都是 Runnable 对象)
# 模型
model = ChatOpenAI(model="gpt-3.5-turbo")
 
# 提示词模板
template = """基于以下上下文回答问题：
{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)
 
# 嵌入模型与检索器
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["文档内容示例..."], embeddings)
retriever = vectorstore.as_retriever()
 
# 2. 定义链 (使用管道符 | 串联)
# 流程：输入 -> 检索上下文 -> 组装 Prompt -> 调用 LLM -> 解析输出
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
 
# 3. 调用链
response = chain.invoke("你的问题是什么？")
print(response)
 
# 4. 流式调用 (自动支持)
for chunk in chain.stream("你的问题是什么？"):
    print(chunk, end="", flush=True)
```

LangChain框架提供了一系列标准化模块与接口：

•统⼀的模型调⽤：通过抽象化的接⼝⽀持多种⼤语⾔模型和嵌⼊模型，可灵活切换不同模型。

•灵活的提⽰词管理：提供提⽰词模板（PromptTemplates），⽀持动态⽣成输⼊内容，并可管理

少样本⽰例与提⽰选择策略，以提升模型响应质量。

•可组合的任务链（Chains）：允许将多个步骤串联成完整流程，如先检索⽂档再⽣成回复，或组合多次模型调⽤。开发者能够通过⾃定义链实现复杂的任务编排。

•上下⽂记忆机制（Memory）：⽤于存储多轮对话中的状态信息。LangChain曾提供多种记忆管理

⽅案（如对话历史记忆和摘要记忆），以实现连贯的交互体验（注：该功能⽬前已由LangGraph

⽀持，原有实现已过时）。

•检索与向量存储集成：⽀持从外部加载⽂档，经分割和向量化处理后存储⾄向量 数据库 ，在查询时检索相关信息并输⼊⼤语⾔模型，帮助构建检索增强⽣成（RAG）类应⽤。LangChain兼容多种主流向量数据库（如FAISS、Pinecone、Chroma）和⽂档加载⼯具，简化知识库应⽤的开发流程。

___

### 三、LangChain安装包：

LangChain生态系统包含不同的包，用来正确选择要安装的功能。

<img
  src="https://i-blog.csdnimg.cn/direct/b02395b6902042848a6833ac75907753.png"
  referrerPolicy="no-referrer"
  alt=""
/>


#### 主langchain包：

```python
pip install langchain
```

#### langchain-core包：

        除了 langsmith SDK之外，LangChain⽣态系统中的所有包都依赖于 langchain-core ，包 含其它包使⽤的基类和抽象，以及LangChainLCEL（表达式语⾔）。 它由 langchain 包⾃动安装，不需要显式安装该包。

#### Intergrations集成包：

        之后按需选择就行。目前不需要安装。

___

### 四、LangChain的快速上手 和 了解基础概念：

#### 1.申请API key并配置环境变量：

（1）去到官网就可以申请API key，我要教的是配置环境变量。

（2）为什么我们要配置环境变量？而不是直接写到.env文件中，或者直接在调用模型的时候传递API key,是因为配置在环境变量中，隐蔽性更好，不会出现在自己的代码文件中。
<img
  src="https://i-blog.csdnimg.cn/direct/63dc9e414e7c4a34a7b0ac6abb825099.png"
  referrerPolicy="no-referrer"
  alt=""
/>


#### 2.定义大模型：

（1）先安装OpenAI包：

\-U， -upgrade 的缩写，即使包已经安装过，但是非最新版，那么也要更新到最新版

```python
pip install -U langchain-openai
```

（2）定义大模型 --- 定义消息列表 --- 调用大模型

```python
from langchain_core.message import HumanMessage, SystemMessage
 
#定义模型
model = ChatOpenAI(model="gpt-4o-mini")
 
#定义消息队列
messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="hi xiaoming!"),
]
 
 
#调用大模型
result = model.invoke(messages)
print(result)
```

（3）输出说明：

AIMessage：来自 AI 的消息。从聊天模型返回，作为对提示（输入）的响应。

-   content：消息的内容。
-   additional\_kwargs：与消息关联的其他有效负载数据。对于来自 AI 的消息，可能包括模型提供程序编码的工具调用。
-   response\_metadata：响应元数据。例如：响应标头、logprobs、令牌计数、模型名称。
    -   侧重于“响应”本身的信息，比如这次请求的 ID、使用的模型版本、以及服务提供商返回的所有原始元数据。它主要用于调试、日志记录和获取请求的上下文信息。
-   usage\_metadata：消息的使用元数据，例如令牌计数。
    -   侧重于“资源消耗”的量化信息，即这次请求消耗了多少 Token。它主要用于成本计算、监控和预算控制。

（4）如果想要输出聊天模型返回的结果 字符串 ，可以使用StrOutputParser输出解析器组件，将大模型输出结果解析为最可能的字符串。

```python
# 定义str字符串输出解析器
 
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
print(parser.invoke(result))
```

使用解析器，而不是.content的原因：当你使用 LangChain 的表达式语言 (LCEL) 构建复杂链条时，解析器是一个独立的“组件”。这使得链条更加清晰和模块化。

**一句话比喻**：

-   `result.content` 就像是你去菜市场买肉，屠夫给你一块带骨头的生肉（`AIMessage`），你自己回家切（取 content）。
-   `OutputParser` 就像是预制菜工厂，它接收生肉，经过标准化处理（去骨、切块、真空包装），最后交给你的是可以直接下锅的净菜（`str`, `dict`, `list`）。在大规模做饭（生产环境）时，预制菜流程更稳定、不易出错。

#### 3.引出LangChain相关概念：

（1）Runnable接口：

Runnable 接口是使用 LangChain Components（组件）的基础。

**概念说明：**

**Components（组件）**：用来帮助当我们在构建应用程序时，提供了一系列的核心构建块，例如语言模型、输出解析器、检索器、编译的 LangGraph 图等。

Runnable 定义了一个标准接口，允许 Runnable 组件：

-   **Invoked（调用）**：单个输入转换为输出。
-   **Batched（批处理）**：多个输入被有效地转换为输出。
-   **Streamed（流式传输）**：输出在生成时进行流式传输。
-   **Inspected（检查）**：可以访问有关 Runnable 的输入、输出和配置的原理图信息。
-   **Composed（组合）**：可以组合多个 Runnable，以使用 LCEL 协同工作以创建复杂的管道。

（2）LangChain Expression Language（LCEL）：

通过LangChain Expression Language构建出来的Runnable对象，被称作RunnableSequence，表示可运行序列。

多个Runnable实例通过管道符链起来的实例都是RunnableSequence。

___

### 

### 五、聊天模型核心能力：

#### 1.调用工具：

在LangChain中，聊天模型提供了工具调用。

（1）创建工具：

        ① 使用@tool自定义工具：

```python
from langchain_core.tools import tool
 
@tool
def multiply(a:int,b:int)->int:
    """
    Multiply two integers.
    Args:
        a:First inter
        b:Second inter
    """
    return a * b
 
print(multiply.invoke({"a": 2, "b": 3})) # 输出: 6
print(multiply.name) # 输出: multiply
print(multiply.description) # 输出: Multiply two ...省略...b: Second integer
print(multiply.args) # 输出: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
```

          ② 继承BaseTool类：

BaseTool提供了三个核心护城河：数据校验，同步/异步，元数据。

```python
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
 
# --- 1. 定义“安检规则” ---
class SearchInput(BaseModel):
    # Field 不仅仅是注释，它是传给 AI 的“填表指南”
    query: str = Field(description="搜索关键词，例如 '北京天气' 或 '英伟达股价'")
 
# --- 2. 制造“精密仪器” ---
class CustomSearchTool(BaseTool):
    # [身份标识]
    name: str = "custom_search"
    # [说明书] 模型根据这段文字判断要不要拿取这个工具
    description: str = "当你需要查询实时天气或新闻时使用此工具。"
    
    # [关联安检站] 告诉工具，收到数据后先用 SearchInput 检查一遍
    args_schema: Type[BaseModel] = SearchInput
 
    # [内部逻辑：同步版]
    def _run(self, query: str) -> str:
        # 真正干活的地方。query 已经是经过校验的字符串了。
        # 你可以在这里写：requests.get(...) 
        return f"查询结果：{query} 目前气温 25°C"
 
    # [内部逻辑：异步版]
    async def _arun(self, query: str) -> str:
        # 对应异步 IO 场景，例如使用 httpx.AsyncClient()
        # 这保证了你的 AI 应用在处理海量请求时不掉链子
        return await some_async_api_call(query)
```

        ③ 使用StructuredTool （将普通函数变成工具）：

普通函数并没有： ① name名称     ② description描述

```python
from langchain_core.tools import StructuredTool
 
def complex_process(user_id: int, action: str):
    return f"用户 {user_id} 执行了 {action}"
 
custom_tool = StructuredTool.from_function(
    func=complex_process,
    name="UserActionManager",
    description="用于管理和记录特定用户的操作行为"
)
```

（2）绑定工具 并且 调用：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing_extensions import Annotated
 
# 定义大模型
 
model = ChatOpenAI(model="gpt-4o-mini")
 
@tool
def add(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Add two integers."""
    return a + b
 
@tool
def multiply(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Multiply two integers."""
    return a * b
 
# 绑定工具
model_with_tools = model.bind_tools([add, multiply])
 
# 调用工具
result = model_with_tools.invoke("9乘6等于多少？")
print(result)
```

        对于这个result，是一个AIMessage（来自AI的消息），它的

        content=“”，

        additional\_kwargs包含模型提供程序编码的工具调用，

        response\_metadata:响应元数据。例如：响应标头，logprobs，令牌计数，模型名称。

        对于工具来说，还包含tool\_calls属性，包含执行该工具的工具名称，输入参数，工具id。

（3）将工具输出传递给聊天模型：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing_extensions import Annotated
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
# 定义工具
@tool
def add(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Add two integers."""
    return a + b
 
@tool
def multiply(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Multiply two integers."""
    return a * b
 
# 绑定工具
tools = [add, multiply]
model_with_tools = model.bind_tools(tools)
 
# 添加AIMessage到消息中去
messages = [
    HumanMessage("9乘6等于多少？5加3等于多少？")
]
 
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)
 
for tool_call in ai_msg.tool_calls:
    # 根据工具名选择对应工具函数（不区分大小写）
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    
    # 执行工具调用，返回 ToolMessage
    tool_msg = selected_tool.invoke(tool_call)
    
    # 将 ToolMessage 加入消息
    messages.append(tool_msg)
 
print(messages)
 
result = model.invoke(messages)
print(result)
```

-   **注意 `messages` 的顺序**：模型非常依赖对话序列。必须按照 `Human -> AI (Tool Call) -> Tool (Result)` 的顺序喂给它，否则它会迷糊。
    
-   **`tool_call_id`**：如果你手动构建 `ToolMessage` 而不带 ID，模型会报错，因为它不知道这个结果是给哪个任务的。
    

（4）使用现场的工具（如搜索工具）：

TavilySearch搜索：要想使用TavilySearch，需要去官网获得api，配置到环境变量中，并且还需要

```python
pip install -U langchain-tavily
```

代码的实现如下：

 

```python
from langchain_openai import ChatOpenAI
from langchain_core.message import HumanMessage
from langchain_tavily import TavilySearch
 
 
#定义模型 和 工具
model = ChatOpenAI(model="gpt-4o-mini")
tool = TavilySearch(max_results=4) #返回4条搜索结果
 
 
#绑定工具
model_with_tool = model.bind_tools([tool])
 
 
#将AIMessage 和 ToolMessage 追加进 messages 中
messages = [
    HumanMessage("中国西安今天的天气怎么样？")
]
 
ai_msg = model_with_tool.invoke(messages)
 
messages.append(ai_msg)
 
for tool_call in ai_msg.tool_calls:   
    #执行工具调用，返回ToolMessage
    tool_msg = tool.invoke(tool_call)
    messages.append(tool_msg)
 
result = model_with_tools.invoke(messages)
print(result.content)
```

> 想要理清楚里面的逻辑，就需要知道ai\_msg.tool\_calls 和 tool\_call 分别是什么类型：

```python
ai_msg.tool_calls :   #列表
[
    {
        "name": "tavily_search_results_json", 
        "args": {"query": "中国西安今天天气"}, 
        "id": "call_1a2b3c",  # 唯一身份证号
        "type": "tool_call"
    },
    {
        "name": "multiply", 
        "args": {"a": 5, "b": 6}, 
        "id": "call_4d5e6f",  # 另一个唯一身份证号
        "type": "tool_call"
    }
]
 
 
tool_call :   #字典
{
    "name": "tavily_search_results_json", 
    "args": {"query": "中国西安今天天气"}, 
    "id": "call_1a2b3c",  # 唯一身份证号
    "type": "tool_call"
}
```

**为什么 `tool.invoke(tool_call)` 很神奇？**

现在我们把这个 `tool_call` 字典传给 `tool.invoke`：

-   **它会自动“对暗号”**：它会检查这个字典里的 `id`（即 `call_1a2b3c`）。
    
-   **它会自动“拆包裹”**：它发现 `args` 里面有个 `query`，于是它会自动把字符串 `"中国西安今天天气"` 喂给 Tavily 搜索函数。
    
-   **它会自动“贴标签”**：执行完搜索后，它会生成一个 `ToolMessage`。
    

#### 2.结构化输出（Structred Output）：

（1）为什么要有结构化输出？

大模型的输出，只是返回流水似的字符串。

有了结构化输出，AI返回的数据就可以直接存储到数据库或者传递给前端的JSON/对象。

（2）为什么公司不直接使用prompt “请返回JSON格式的数据”？

大模型数据格式不稳定，导致程序奔溃。

类型检查难，数字不是真数字，而是字符串。存储数据库的时候int 与 string 混淆

（3）企业级的标准写法：Pydantic模式

```python
from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
 
# --- 第一步：定义数据结构 (Schema) ---
# 使用 Pydantic 定义你想要的数据样子
class Joke(BaseModel):
    """关于笑话的结构化数据结构""" # 这里的 Docstring 也会发给 AI 帮助它理解
    setup: str = Field(description="笑话的铺垫/开头")
    punchline: str = Field(description="笑话的笑点/包袱")
    rating: Optional[int] = Field(description="笑话的幽默指数 (1-10)", default=None)
 
# --- 第二步：初始化模型 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
# --- 第三步：核心步骤 - 绑定结构化输出 ---
# 这会创建一个“结构化模型”对象
structured_llm = llm.with_structured_output(Joke)
 
# --- 第四步：直接调用并获取对象 ---
result = structured_llm.invoke("给我讲个关于程序员的笑话")
 
# 深度细节：result 不再是字符串，而是一个 Joke 类的实例对象！
print(f"铺垫: {result.setup}")
print(f"笑点: {result.punchline}")
print(f"评分: {result.rating}")
```

进阶的嵌套用法：

 

```python
class Step(BaseModel):
    explanation: str = Field(description="这一步的逻辑解释")
    output: str = Field(description="这一步的计算结果")
 
class MathReasoning(BaseModel):
    steps: List[Step] = Field(description="解题的详细步骤列表")
    final_answer: str = Field(description="最终答案")
 
# 同样的方法绑定
reasoning_llm = llm.with_structured_output(MathReasoning)
ans = reasoning_llm.invoke("为什么 0.1 + 0.2 不等于 0.3？")
 
# 你可以直接循环 steps
for i, step in enumerate(ans.steps):
    print(f"步骤 {i+1}: {step.explanation}")
```

最后大模型输出的结果，ans的格式：

```python
steps=[ 
    Step(explanation='...', output='...') , 
    Step(explanation='...', output='...') ,
    .....
],
final_answer=''
```

（4）返回JSON格式：

```python
from langchain_openai import ChatOpenAI
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
json_schema = {
    "title": "joke",
    "description": "给用户讲一个笑话。",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "这个笑话的开头",
        },
        "punchline": {
            "type": "string",
            "description": "这个笑话的妙语",
        },
        "rating": {
            "type": "integer",
            "description": "从1到10分，给这个笑话评分",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
 
structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke("给我讲一个关于唱歌的笑话")
print(result)
```

打印结果如下：

```python
{'setup': '为什么唱歌的人总是很开心？', 'punchline': '因为他们总是有很多音符可供选择！', 'rating': 7}
```

#### 3.流式传输：

（1）stream:

```python
from langchain_openai import ChatOpenAI
 
model = ChatOpenAI(model="gpt-4o-mini")
 
for chunk in model.stream("讲一个50字的笑话"):
    print(chunk.content,end="",flush=true)
```

流式输出 的返回是一个个的AIMessageChunk消息块。

（2）astream 异步传输：

我们需要了解 asyncio、 协程、 事件循环。

① 多进程，多线程，多协程的对比：

|**维度**|**多进程**|**多线程**|**多协程**|
|---|---|---|---|
|**调度者**|操作系统 (OS)|操作系统 (OS)|用户/程序员 (User)|
|**内存占用**|很大 (独立空间)|中等 (共享空间)|极小 (栈空间很小)|
|**切换成本**|极高 (环境切换)|中等|极低|
|**通信方式**|IPC (管道/消息队列)|直接读写共享内存|消息通道 (Channel) 或内存|
|**稳定性**|最强 (互不影响)|一般 (一崩全崩)|一般 (受线程影响)|

② asyncio:

        Python标准库中的模块，⽤于编写异步I/O操作的代码.

③ 事件循环：

        事件循环是asyncio核心，它的⼯作流程⾮常简单：

                 它维护着⼀个任务列表（⽐如：煮⽔、发短信）。

                 它不断地循环检查每个任务：

                        a. 如果任务处于 已经 “ “ 等待 I/O” 状态（⽐如等⽔开、等⽹络响应），就暂停它，⽴即去执⾏下⼀个 就绪 ” 的任务。

                        b. 如果任务的等待时间到了或者I/O操作完成了，事件循环就恢复执⾏这个任务。

```python
from langchain_openai import ChatOpenAI
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
# 异步调用
async def async_stream():
    print("=== 异步调用 ===")
    async for chunk in model.astream("讲一个50字的笑话"):
        print(chunk.content, end="|", flush=True)
 
import asyncio
asyncio.run(async_stream())
```

（3）输出解析器：（在流式输出中，输出解析器的占用）

        ① langchain中自带一个输出解析器StrOutputParser

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
 
#定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
#定义解析器
parser = StrOutputParser()
 
chain = model | parser
 
for chunk in chain.stream(("写⼀段关于爱情的歌词，需要5句话"):
    print(chunk,end="|",flush=True)
```

```python
|在|星|空|下|许|下|心|愿|，|
|你的|笑|容|如|晨|光|温|暖|，|
|手|握|手|走|过|每|段|光|阴|，|
|无|论|风|雨|依|然|不|离|不|弃|，|
|爱|是|永|恒|，|心|与|心|相|连|。| |||
```

        ② 自己实现一个输出解析器：

        可以将一个字或者两个字的输出，自定义成一句话一句话的输出。

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, List
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
# 定义输出解析器
parser = StrOutputParser()
 
# 定义生成器
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    buffer = ""
    for chunk in input:
        buffer += chunk
        while "。" in buffer:
            # 只要缓冲区中包含句号，就找到第一个句号的位置
            stop_index = buffer.index("。")
            # 将句号之前的内容（去除首尾空格）作为一个句子放入列表中并产出
            yield [buffer[:stop_index].strip()]
            # 更新缓冲区，保留句号之后的内容
            buffer = buffer[stop_index + 1 :]
    yield [buffer.strip()]
 
# 定义链
chain = model | parser | split_into_list
 
for chunk in chain.stream("写一份关于爱情的歌词，需要5句话，每句话用句号分割"):
    print(chunk, end="|", flush=True)
```

```python
['在星空下许下承诺的誓言']|['你的笑容如同晨曦，温暖了我的心']|['无论时光如何流转，我愿与你携手共行']|['爱情是我们心中永恒的旋律']|['每一次相拥，都是一场甜蜜的重逢']|['']|
```

（4）深度理解流式输出：

     流式输出（服务器中的大模型需要不断向 客户端 发送数据），我们可以想到使用 websocket 网络传输协议，但是最终选择的却是SSE网络传输协议。

___

WebSocket网络传输协议：

优点：

        全双工：支持服务器自动向客户端发送数据。

        长连接：一次连接就可以持续不断的进行通信。

缺点：

        需要客户端和服务端都支持WebSocket协议才能用。

        服务器需要管理与每一个客户端建立起来的WebSocket长连接。消耗资源。

___

SSE网络传输协议：

        SSE协议是基于HTTP协议改编而来的。通过 Server-Sent Events （服务器发送事件，简称SSE）技术可实现流式传输。SSE协议可以做到，一次连接：客户端向服务器发送一次请求，服务端向客户端发送多次数据，直到断开连接。

<img
  src="https://i-blog.csdnimg.cn/direct/ebc9dfb2cf744dd1a2df72e5d658572f.png"
  referrerPolicy="no-referrer"
  alt=""
/>


特点：

-   **基于 HTTP 协议：**复用标准 HTTP/HTTPS 协议，无需额外端口或协议，兼容性好且易于部署。
    
-   **单向通信机制**SSE： 仅支持服务器向客户端的单向数据推送，客户端通过普通 HTTP 请求建立连接后，服务器可持续发送数据流，但客户端无法通过同一连接向服务器发送数据。
    
-   **自动重连机制：**支持断线重连，连接中断时，浏览器会自动尝试重新连接（支持 `retry` 字段指定重连间隔）。
    
-   **自定义消息类型：**客户端发起请求后，服务器保持连接开放，响应头设置 `Content-Type: text/event-stream`，标识为事件流格式，持续推送事件流。
    

#### 4.使用LangSmith跟踪 LLM 应用：

使用 LangChain 构建的许多应用程序，可能会包含多个步骤和多次的 LLM 调用。随着这些应用程序变得越来越复杂，作为开发者，我们能够检查链或代理内部到底发生了什么变得至关重要。最好的方法是使用 **LangSmith**。

LangSmith 与框架无关，它可以与 langchain 和 langgraph 一起使用，也可以不使用。LangSmith 是一个用于帮助我们构建生产级 LLM 应用程序的平台，它将密切监控和评估我们的应用。

**LangSmith 平台地址**：[https://smith.langchain.com/](https://smith.langchain.com/ "https://smith.langchain.com/") （新用户需要注册）

要想让 LangSmith 跟踪 LLM 应用，第一步申请 LangSmith API Key，点击 Settings，就会跳转到 “API Keys” 设置页面，若没有跳转，可以在左侧 tab 栏中找到进入。

<img
  src="https://i-blog.csdnimg.cn/direct/44703fdbd6cc4546bca7dee6b4ba82af.png"
  referrerPolicy="no-referrer"
  alt=""
/>

<img
  src="https://i-blog.csdnimg.cn/direct/836c72d0838249c28e7bdb628179054e.png"
  referrerPolicy="no-referrer"
  alt=""
/>


创建完成后，保存好你的APIKey。 接下来配置两个环境变量：

```python
LANGSMITH_TRACING="true"LANGSMITH_API_KEY="你的 LangSmith API Key"
```
<img
  src="https://i-blog.csdnimg.cn/direct/caba7e05131746e8800477d4223c85d8.png"
  referrerPolicy="no-referrer"
  alt=""
/>

配置完成后，我们之后执行的代码，都会被LangSmith跟踪起来，方便我们查询。跟踪会以瀑布流形式展⽰调⽤的完整步骤，以及每个步骤的详细信息和耗时。让我们能够检查内部到 底发⽣了什么！！

___

### 

### 六、核心组件：

#### 1.消息：

##### （1）LLM消息结构：

-   消息角色：

|角色|描述|
|---|---|
|system（系统角色）|用于告诉聊天模型如何行为并提供额外的上下文。并非所有聊天模型提供商都支持。|
|user（用户角色）|表示用户与模型交互的输入，通常以文本或其他交互式输入的形式。|
|assistant（助理角色）|表示来自模型的响应，其中可以包括文本或调用工具的请求。|
|tool（工具角色）|用于在检索外部数据或将工具调用的结果传递回模型的消息。与支持工具调用的聊天模型一起使用。|

-   消息内容：

> 表示多模态数据（例如，图像、音频、视频）的消息文本或字典列表的内容。内容的具体格式可能因底层不同的 LLM 而异。目前，大多数模型都支持文本作为主要内容类型，对多模态数据的支持仍然有限。

-   消息其他元数据：

|元数据|描述|
|---|---|
|ID|消息标识符。|
|Name|名称允许区分具有相同角色的不同实体。并非所有型号都支持此功能！|
|Metadata|有关消息的其他信息，例如时间戳、令牌使用情况等。|
|ToolCalls|模型发出的一个或多个工具的调用请求|

##### （2）LangChain消息结构：

消息格式：

|消息类型|对应角色|描述|
|---|---|---|
|**SystemMessage**|system（系统角色）|用于启动 AI 模型的行为并提供额外的上下文，例如指示模型采用特定角色或设定对话的基调（例如，“你是一个后端开发的专家”）。|
|**HumanMessage**|user（用户角色）|人类消息表示用户与模型交互的输入。大多数聊天模型都希望用户输入采用文本形式。|
|**AIMessage**|assistant（助理角色）|这是来自模型的响应，其中可以包括文本或调用工具的请求。它还可能包括其他媒体类型，如图像、音频或视频 —— 尽管这目前仍然不常见。|
|**AIMessageChunk**|assistant（助理角色，用于流式响应）|通常在生成聊天模型时流式传输响应，因此用户可以实时看到响应，而不是等待生成整个响应后再显示。|
|**ToolMessage**|tool（工具角色）|这表示一条角色为 “tool” 的消息，其中包含调用工具的结果。|

##### （3）缓存历史消息：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
store = {}
# 接受一个 session_id 并返回一个消息历史对象。
# 这个 session_id 用于区分不同的对话，并应作为配置的一部分在调用新链时传入
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        # InMemoryChatMessageHistory() 将消息存储在内存列表中。
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
 
# 包装 model，管理聊天消息历史记录
with_message_history = RunnableWithMessageHistory(model, get_session_history)
 
config = {"configurable": {"session_id": "1"}}
with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
).pretty_print()
 
with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
).pretty_print()
 
"""
当你调用 with_message_history.invoke(input, config) 时，LangChain 内部发生了以下步骤：
第一步：拦截与提取 (Interception)
    RunnableWithMessageHistory 是一个包装器（Wrapper）。它首先不会把 config 发给 LLM，而是截获它。它通过内部逻辑扫描 config["configurable"] 这个字典。
第二步：动态参数映射 (Dynamic Mapping)
    在底层实现中，RunnableWithMessageHistory 维护了一个参数映射表。
    它会寻找一个名为 session_id 的键（这个键名是在你实例化 RunnableWithMessageHistory 时指定的，默认就是 session_id）。
第三步：回调工厂函数 (Callback Execution)
    这是最关键的一步。还记得你定义的这个函数吗？
    def get_session_history(session_id: str) -> BaseChatMessageHistory
第四步：上下文拼接 (Context Merging)
    拿到 store["1"] 里的历史消息。
    将这些消息和当前的 HumanMessage 组合成一个 List[BaseMessage]。
    重新包装输入，最后才把这个完整的列表交给 ChatOpenAI 模型。
"""
```

##### （4）管理历史消息：

**①上下文窗口：用户的输入 + 大模型的输出**

每一个大模型都存在上下文窗口的现实。也就是说，我们并不能在每一次对话的时候，都把之前的消息传递给大模型，让他拥有记忆。

这时候，就需要对我们的输入进行处理：

        消息裁剪，消息过滤，消息合并

**② 消息裁剪 (Trimming)**

裁剪是控制 Token 消耗最有效的手段。`trim_messages` 函数支持基于消息数量或 Token 数量进行操作。

> 代码示例：基于 Token 数量裁剪

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import OpenAIEmbeddings # 或者是其他的 Tiktoken 分词器
 
# 模拟一段对话历史
messages = [
    SystemMessage(content="你是一个专业的助手。"),
    HumanMessage(content="你好，我想了解量子力学。"),
    AIMessage(content="量子力学是物理学的一个分支..."),
    HumanMessage(content="那它和经典物理有什么区别？"),
    AIMessage(content="区别在于微观粒子的波粒二象性..."),
]
 
# 配置裁剪器
# strategy="last" 表示保留最后的对话（最近的）
# token_counter 通常传入模型的 tiktoken 计数器
trimmer = trim_messages(
    max_tokens=45, 
    strategy="last",
    token_counter=len, # 简单起见用字符串长度，实际建议用模型分词器
    include_system=True, # 始终保留系统提示词
    start_on="human",    # 确保裁剪后以人类消息开始（对某些模型很关键）
)
 
selected_messages = trimmer.invoke(messages)
# 结果将只包含 SystemMessage 和最后 1-2 条对话
```

> 基于消息数裁剪 (By Message Count)
> 
> 这是最直接的控制方式：无论消息多长，只保留最近的 N 条。这在对话逻辑简单、Token 预算充足的情况下非常高效。

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
 
messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="第一轮提问"),
    AIMessage(content="第一轮回答"),
    HumanMessage(content="第二轮提问"),
    AIMessage(content="第二轮回答"),
    HumanMessage(content="第三轮提问"),
]
 
# 核心：基于消息数裁剪
# strategy="last"：保留最后的
# token_counter=len：在基于数量裁剪时，我们可以把每条消息看作“1个单位”
# 或者直接不指定复杂的 counter，仅通过控制逻辑实现
trimmer_by_count = trim_messages(
    strategy="last",
    max_tokens=3,        # 这里的“token”定义取决于下面的 counter
    token_counter=lambda msgs: len(msgs), # 强制让计数器返回消息条数
    include_system=True, # 依然保留系统消息
    start_on="human"     # 确保裁剪后第一条是用户发的
)
 
trimmed = trimmer_by_count.invoke(messages)
# 结果通常包含：SystemMessage + 最后 2 条对话
```

**③ 消息过滤 (Filtering)**

> 当你只想提取特定类型的对话（例如只看用户的提问，不看 AI 的中间思考或工具调用）时使
> 
> 用。
> 
> 代码示例：按类型过滤

```python
from langchain_core.messages import filter_messages, ToolMessage
 
# 过滤掉所有的 ToolMessage，只保留人类和 AI 的核心对话
filtered = filter_messages(
    messages, 
    include_types=["human", "ai"], 
    exclude_names=["bad_tool_output"]
)
```

**④ 消息合并 (Merging)**

> 当系统中有多个连续的同角色消息时（例如用户连续输入、或者多个 Tool 连续返回），合并可以净化结构。

```python
from langchain_core.messages import merge_message_runs
 
messages = [
    SystemMessage(content="助手模式"),
    HumanMessage(content="第一部分提问..."),
    HumanMessage(content="第二部分补充..."), # 连续两条 Human 消息
    AIMessage(content="我正在思考..."),
    AIMessage(content="这是我的回答。"),  # 连续两条 AI 消息
]
 
merged = merge_message_runs().invoke(messages)
# 合并后：Human 消息内容会通过换行符拼接在一起
```

___

**⑤ 💡 最佳实践：在 LCEL 链中集成**

在生产环境中，你不需要手动调用这些函数，而是将它们作为链的一部分：

```python
from langchain_openai import ChatOpenAI
 
model = ChatOpenAI(model="gpt-4")
 
# 构建处理链
# 输入消息 -> 合并重复 -> 过滤无关信息 -> 裁剪 Token -> 发送给模型
chain = (
    merge_message_runs() 
    | filter_messages(exclude="tool") 
    | trimmer 
    | model
)
 
# 直接传入完整的 history，链会自动处理好所有裁剪逻辑
response = chain.invoke(messages)
```

**总结**

-   **裁剪 (Trimming)**：解决“装不下”的问题（硬限制）。
    
-   **过滤 (Filtering)**：解决“噪音多”的问题（提高质量）。
    
-   **合并 (Merging)**：解决“格式乱”的问题（提高兼容性）。
    

#### 2.提示词模板：

LangChain Hub 平台，有很多现成的提示词模板。[Hub - LangSmith](https://smith.langchain.com/hub/ "Hub - LangSmith")

##### （1）字符串模板：

LangChain 提供了 PromptTemplace 类，实现了标准的 Runnable 接口。

```python
from langchain_core.prompts import PromptTemplate
 
# 1.定义模板
prompt_template = PromptTemplate.from_template("Translate the following into 
{language}")
 
print(prompt_template.invoke({"language":"Chinese"}))
```

```python
Translate the following into Chinese
```

##### （2）聊天消息模板：

LangChain 专门为聊天消息模型提供了 ChatPromptTemplate 模板。

```python
from langchain_core.prompts import ChatPromptTemplate
 
# 1. 设置模板
prompt_template = ChatPromptTemplate(
    [
        ("system", "Translate the following into {language}."),
        ("user", "{text}")
    ]
)
 
# 说明:
# 在 0.2.24 版本后可以直接使用ChatPromptTemplate()来初始化模板
# 在 0.2.24 版本前，需要使用 ChatPromptTemplate.from_messages()来初始化模板
 
# 2. 实例化模板，获取消息实例
messagesValue = prompt_template.invoke(
    {
        "language": "Chinese",
        "text": "what is your name?"
    }
)
 
messages = messagesValue.to_messages()
print(messages)
```

```python
[
    SystemMessage(content='Translate the following into Chinese.', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='what is your name?', additional_kwargs={}, response_metadata={})
]
```

需要注意的是，为什么ChatPromptTemplate可以将system和user转换成SystemMessage和HumanMessage？

> 核心原因：ChatPromptTemplate**自动识别角色字符串**，然后**new 一个对应消息**：
> 
> "system" → SystemMessage
> 
> "user" → HumanMessage
> 
> "ai" → AIMessage
> 
> "assistant" → AIMessage
> 
> "tool" → ToolMessage

```python
# 快捷写法（我上面写的）
ChatPromptTemplate([
    ("system", "翻译"),
    ("user", "{text}")
])
 
 
# 原始写法（手动创建消息）
ChatPromptTemplate([
    SystemMessage(content="翻译"),
    HumanMessage(content="{text}")
])
```

现在，我们可以将该结果发送给任何⼀个LLM来获取答案。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
# 1. 设置模板
prompt_template = ChatPromptTemplate(
    [
        ("system", "Translate the following into {language}."),
        ("user", "{text}")
    ]
)
 
# 2. 实例化模板，获取消息实例
# messageValue 是一个模板结果对象，不是消息。模型看不懂，需要使用to_message()
messagesValue = prompt_template.invoke(
    {
        "language": "Chinese",
        "text": "what is your name?"
    }
)
messages = messagesValue.to_messages()
print(messages)
 
# 3. 输出解析
parser = StrOutputParser()
chain = model | parser
print(chain.invoke(messages))
```

由于 ChatPromptTemplate 也实现了标准的 Runnable 接口。因此我们也可以通过链来完成调用。

```python
# 定义消息模板
prompt_template = ChatPromptTempalte([
    ("system":"Translate the following into {language}"),
    ("user":"{text}")
])
 
# 定义模型
model = ChatOpenAI(model = "gpt-4o-mini")
 
# 定义解析器
parser = StrOutputParser()
 
chain = prompt_template | model | parser
 
for chunk in chain.stream(
    {
        "language":"English",
        "text":"你好，我叫小明，很高兴认识你！"
    }
):
    print(chunk,end="",flush=true)
```

##### （3）消息占位符：

由上面的聊天消息模板中可以看出，我们可以给字符串进行占位。但是如果我们如果想要给消息进行占位呢？我们就需要使用到MessagesPlaceholder，负责在特定位置添加消息列表。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
 
prompt_template = ChatPromptTemplate([
    ("system", "你是一个聊天助手"),
    MessagesPlaceholder("msgs")  # 消息占位符
])
 
messages_to_pass = [
    HumanMessage(content="中国首都是哪里?"),
    AIMessage(content="中国首都是北京。"),
    HumanMessage(content="那法国呢?")
]
 
formatted_prompt = prompt_template.invoke({"msgs": messages_to_pass})
print(formatted_prompt)
```

```python
messages = [
    SystemMessage(content='你是一个聊天助手', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='中国首都是哪里?', additional_kwargs={}, response_metadata={}),
    AIMessage(content='中国首都是北京。', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='那法国呢?', additional_kwargs={}, response_metadata={})
]
```

在不显式使⽤ MessagesPlaceholder 类也可以完成该能⼒：

原因是因为ChatPromptTemplate中进行映射。将 placeholder 与 MessagesPlaceholder 映射。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
 
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{msgs}")
    ]
)
 
messages_to_pass = [
    HumanMessage(content="中国首都是哪里?"),
    AIMessage(content="中国首都是北京。"),
    HumanMessage(content="那法国呢?")
]
 
formatted_prompt = prompt_template.invoke({"msgs": messages_to_pass})
print(formatted_prompt)
```

#### 3.少样本提示（few-shotting）

**（1）优点：**

-   统一输出格式
-   模拟特定语气
-   处理复杂的逻辑推论
-   解决专业领域知识不足问题（公司私有数据）

**（2）动态少样本**（工作常用）**：**

-   **用户提问：** “如何重置我的专业版订阅？”
    
-   **向量数据库检索：** 自动寻找与“订阅重置”最相似的 3 个历史优质回答。
    
-   **动态组合：** 将这 3 个例子塞进 Prompt，发送给模型。
    
-   **结果：** 模型基于这三个相似案例，生成精准的回复。
    

（3）少样本提示和结构化输出有那么多相似的地方，为什么还需要它？

-   **结构化输出** 解决的是 **“形”**（数据格式对不对，是不是 JSON，字段齐不齐）。
    
-   **少样本提示** 解决的是 **“神”**（内容质量好不好，推理逻辑深不深，语气像不像）。
    
-   最主要就是**动态**少样本
    

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
 
# 1. 准备大量的“业务知识库”例子
examples = [
    {"input": "医疗险：感冒发烧去医院开药", "output": "判定：不理赔。理由：未达起付线。"},
    {"input": "意外险：走路摔跤导致腿部骨折", "output": "判定：理赔。理由：符合意外伤害定义。"},
    {"input": "财产险：家里水管爆裂淹了地板", "output": "判定：理赔。理由：属于突发财产损失。"},
    {"input": "重疾险：体检发现早期原位癌", "output": "判定：部分理赔。理由：按轻症比例赔付。"},
    # 假设这里有 1000 条例子...
]
 
# 2. 初始化向量数据库，存放这些例子
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(), # 使用 Embedding 模型计算相似度
    Chroma,             # 使用 Chroma 作为本地向量库
    k=2                 # 每次只取最相似的 2 个例子
)
 
# 3. 定义 Few-Shot 模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])
 
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector, # 关键：传入选择器而不是固定的 examples
)
 
# 4. 最终组合
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个保险专家。请参考以下相似案例进行分析："),
    few_shot_prompt,
    ("human", "{input}"),
])
 
# 测试：当我们问关于“猫抓伤”时，系统会自动去搜“意外险/伤害”相关的例子
chain = final_prompt | model
print(chain.invoke({"input": "我被家里的猫抓伤了，打了狂犬疫苗。"}).content)
```

> **这个就是RAG（检索增强生成）的变式。我们不是检索文档，而是检索例子。**

采用动态少样本生成，在公司中就是降维打击：

-   **极高的针对性：** 如果用户问的是“车险”，Prompt 里出现的全是“车险”的例子，不会出现“医疗险”来干扰模型，准确率暴增。
    
-   **节省 Token：** 你不需要把整个公司手册喂给模型，每次只喂最相关的 2-3 个例子（几百个 Token），既省钱又快速。
    
-   **数据实时更新：** 如果业务逻辑变了，你只需要往向量数据库里新增/修改几条例子，**无需重新开发代码或微调模型**，系统立刻就能学会新规则。
    

写到这里的时候，我突然想到！如果少样本中混入垃圾数据，那么生成的效果将会大打折扣！！！

我们需要对向量数据库中的数据进行过滤：

> **1\. 黄金集过滤（Gold Dataset Curation）**
> 
> 不是所有的历史对话都能当例子。公司通常会建立一套**审核机制**：
> 
> -   **人工清洗：** 业务专家（SME）对向量库中的 `input-output` 对进行打分，只有达到 90 分以上的才能被标注为 `is_gold=True`。
>     
> -   **多样性过滤：** 避免库里全是相似的简单案例。我们会使用聚类算法，确保库中涵盖了“简单、中等、极难”以及各种业务子类（如车险、寿险、意外险）的典型代表。
>     
> 
> ___
> 
> **2\. 负样本（Negative Examples）的妙用**
> 
> 有时候，告诉模型“什么是错的”比“什么是对的”更管用。
> 
> -   **策略：** 在 Prompt 中加入 1-2 个 **Counter-examples（反例）**。
>     
> -   **结构：** > **错误示范：** \[输入内容\] -> \[错误逻辑\] -> \[警告：请勿这样处理，因为违反了合规项 A\]
>     
> -   **用途：** 这种方式在金融、医疗等高度合规的行业非常有效，能显著降低模型违规回复的概率。
>     

#### 4.输出解析器：

在LangChain的企业级开发中，**输出解析器**是连接**非结构化的AI文本**与**结构化的程序代码**的桥梁。

在langChain中，支持十几种的输出解析器。我就讲常见的两种：PydanticOutputParser 和  JsonOutputParser。

##### （1）PydanticOutputParser:强类型校验王者

它是最严谨的解析器。它不仅定义了JSON的结构，还利用Pydantic的校验能力（类型检查，长度限制，数值区间） 来确保 AI 返回的数据完全符合业务逻辑。

**核心特点**

-   **强类型：** 确保 `age` 必须是 `int`，`email` 必须符合格式。
-   **自动生成指令：** 它能自动生成一段 Prompt 注入到系统消息中，告诉模型：“请按以下 JSON 架构输出...”。
-   **报错精准：** 如果模型漏掉字段，它会抛出详细的 Pydantic 错误。

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
 
# 1. 定义期望的结构
class UserAction(BaseModel):
    action: str = Field(description="用户的操作意图")
    confidence: float = Field(description="置信度 0-1")
    
    # 甚至可以添加校验逻辑
    @validator("confidence")
    def check_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError("置信度必须在 0 到 1 之间")
        return v
 
# 2. 初始化解析器
parser = PydanticOutputParser(pydantic_object=UserAction)
 
# 3. 获取解析器生成的格式指令，并注入 Prompt
prompt = ChatPromptTemplate.from_template(
    "回答用户问题。\n{format_instructions}\n{query}"
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
 
# 4. 链路组合
chain = prompt | model | parser
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
# 定义输出结构: Pydantic 类
class Joke(BaseModel):
    """给用户讲一个笑话。"""
 
    setup: str = Field(description="这个笑话的开头")
    punchline: str = Field(description="这个笑话的妙语")
    rating: Optional[int] = Field(
        default=None, description="从1到10分，给这个笑话评分"
    )
 
# 设置解析器
parser = PydanticOutputParser(pydantic_object=Joke)
 
# 提示词模板
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    # partial_variables: 提示模板携带的部分变量的字典，无需在每次调用提示时都传入它们。
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
 
chain = prompt | model | parser
for chunk in chain.stream({"query": "给我讲一个关于唱歌的笑话"}):
    print(chunk, end="|")
```

##### （2） JsonOutputParser：灵活与流式的先锋

相比 Pydantic 的严苛，`JsonOutputParser` 更加轻量。它不强制要求你定义一个 Pydantic 类（虽然也可以配合使用），它更擅长从模型的原始回复中提取 JSON 块。

核 心特点

-   **支持流式（Streaming）：** 这是它最大的优势。当模型正在一个字一个字蹦出 JSON 时，`JsonOutputParser` 可以**实时**解析出部分已完成的 JSON 对象，这对于提升用户体验（如实时看板）至关重要。
-   **容错性：** 它在处理 Markdown 里的 \`\`\`json 代码块时非常稳健。

```python
from langchain_core.output_parsers import JsonOutputParser
 
# 可以不定义类，直接让它返回字典
parser = JsonOutputParser()
 
# 如果想指定结构，也可以传入 Pydantic 类，但它比 PydanticOutputParser 更能抗住“流式”压力
# parser = JsonOutputParser(pydantic_object=MyModel)
 
chain = prompt | model | parser
 
# 支持流式获取结果
for chunk in chain.stream({"query": "告诉我三条天气信息"}):
    print(chunk) # 你会看到 JSON 对象是一点点长出来的
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
 
# 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")
 
# 设置解析器
parser = JsonOutputParser()
 
# 提示词模板
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    # 提前固定format_instructions这个变量
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
 
chain = prompt | model | parser
print(chain.invoke({"query": "给我讲一个关于唱歌的笑话"}))
```

除了上面讲的文本、对象、JSON 解析器，其实 LangChain 官方还提供了更多类型的解析器，如:

-   XML 解析器：`XMLOutputParser`
-   Yaml 解析器：`YamlOutputParser`
-   CSV 解析器：`CommaSeparatedListOutputParser`
-   枚举解析器：`EnumOutputParser`
-   日期解析器：`DatetimeOutputParser` 等等。

除此之外，LangChain 还支持我们**自定义输出解析器**，以将模型输出结构化自定义格式，详细情况参考[这里](https://docs.langchain.com/oss/python/langchain/overview "这里")。

#### 5.文档加载器：

文档加载器是实现RAG的重要一步。

##### （1）加载 PDF 文档：

```python
from langchain_community.document_loaders import PyPDFLoader
#文档加载器（PDF） -- 图片并没有加载出来，只加载了文本
loader = PyPDFLoader(file_path="../Docs/PDF/C++ - 仿RabbitMQ实现消息队列.pdf")
 
#生成文档列表
docs = loader.load()
 
print(f"PDF文档总页数：{len(docs)}页")
print(f"第一页文本内容是：{docs[0].page_content[:200]}")
print(f"第一页的元数据字典是：{docs[0].metadata}")
 
print(f"第二页文本内容是：{docs[1].page_content[:200]}")
print(f"第二页的元数据字典是：{docs[1].metadata}")
```

这个PyPDFLoader函数，默认按照页来分割。

##### （2）加载Markdown文档：

```python
#使用UnstructuredMarkdownLoader 加载Markdown文件
from langchain_community.document_loaders import UnstructuredMarkdownLoader
loader = UnstructuredMarkdownLoader(file_path="../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md",mode="elements")
docs = loader.load()
print(f"Markdown文档总页数：{len(docs)}页")
print(f"第一文档的内容是：{docs[0].page_content[:200]}")
print(f"第一文档的元数据字典是：{docs[0].metadata}")
```

UnstructuredMarkdownLoader函数，默认 model="single" 将一整个Markwodn文档当作一个文档。docs只有一个元素。

而我设置成了model = “elements” 将按照文档结构元素分割。

> 它会把整个 Markdown 按 **这些元素类型** 来切割：
> 
> -   标题（# H1、## H2、### H3...）
> -   段落（普通文本）
> -   代码块（`...`）
> -   列表（有序 / 无序列表）
> -   表格
> -   引用块
> -   图片
> -   加粗 / 斜体等**行内元素**（会合并到父元素）

##### （3）上面两种不同的格式的文档进行划分，都存在问题。

PDF的有可能一个问题被划分成两个部分。导致问题不完整。

Markdown中 model="single",文本超长，无法分段，无法按章节检索，大模型上下文窗口token长度限制。

Markdown中 model="elements",划分太细，一个问题被拆成多个文本。

这些问题，都需要文本分割器来解决。

#### 6.文本分割器：

我们已经知道可以通过文档加载器完成各种数据源的加载，将其转换为文档对象 `Document`。那么接下来要做的就是**文档拆分**。

文档拆分通常是将大文本分解为更小的、易于管理的块。这对于索引数据并将其传递到模型中都很有用。因为，**大块更难搜索并且不适合模型的有限上下文窗口**。拆分可以提高搜索结果的粒度，从而可以更精确地将查询与相关文档部分进行匹配。

LangChain 的**文本分割器**便能将大型文档分解为更小的块。
<img
  src="https://i-blog.csdnimg.cn/direct/5712358278f14978a77d92464c8e8d31.png"
  referrerPolicy="no-referrer"
  alt=""
/>

##### （1）基于文档长度 + 文档语义拆分：

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
 
markdown_path = "../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md"
# single 模式加载后，默认只有一个 Document 对象
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
 
# 文本分割器
text_splitter = CharacterTextSplitter(
    separator="\n\n",         # 选择分隔符: 它有一个默认的分隔符优先级列表，通常是: ["\n\n", "\n", " ", ""]。它会按顺序尝试这些分隔符
    chunk_size=100,           # 设定目标: 目标块大小
    chunk_overlap=20,         # 设定目标: 块之间的重叠大小
    length_function=len,      # 使用测量长度的函数
    is_separator_regex=False, # 分隔符是正则表达式吗
)
 
# 分割文档，返回被分割的文档列表
texts = text_splitter.split_documents(data)
# 打印前10个被分割出来的文档
for document in texts[:10]:
    print("*" * 30)
    print(f"{document}\n")
```

##### （2）基于token + 文档语义拆分：

```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(  
    eencoding_name="cl100k_base", # cl100k_base是tiktoken分词器中的一种编码方式  
    chunk_size=300,         # 块大小(参考标准，为了保证段落/句子完整，会超出设定的大小)  
    chunk_overlap=50,       # 块重叠长度  
)  
```

##### （3）多种分割的规则：

        需要用到递归分割的函数(RecursiveCharacterTextSplitter)：

```python
text_splitter = RecursiveCharacterTextSplitter(  
    separator=["\n\n", "\n", " "],       # 分隔符。一般来说，有一个默认的分割符优先级列表： ["\n\n", "\n", " "]    
    chunk_size=300,         # 块大小(参考标准，为了保证段落/句子完整，会超出设定的大小)  
    chunk_overlap=50,       # 块重叠长度  
    length_function=len,    # 测量字符长度的函数  
    is_separator_regex=False, # 是否正则表达式描述分隔符？  
)  
```

#### 7.文本向量：

##### （1）向量：

首先我们要知道，嵌入的结果就是一个向量，它本质上是一个数字列表（一维数组）。例如：`[0.023, 0.487, -0.129, ..., 0.325]`。对于向量来说，有两个关键概念需要了解：

**向量维度**

嵌入结果得到的列表长度是固定的，称为向量的 “维度”。例如，OpenAI 的 `text-embedding-ada-002` 模型会生成一个 1536 维的向量，`text-embedding-3-large` 模型会生成一个 3072 维的向量。

维度越高，通常能捕捉更细微的语义信息，但也需要更多的计算和存储资源。

**向量空间**

想象一个无限延伸的、拥有无数个维度的宇宙，这个宇宙就是一个向量空间。这有点抽象，可以想象一下：

-   在三维世界里，一个点可以用 `(x, y, z)` 坐标表示，例如 `(2, 5, -1)`。
-   在机器学习的高维向量空间中，一个点可能是 `(0.1, 0.7, -0.2, 0.4, ..., 0.02)`，一个有几百或几千个数字的坐标。

在这个空间里，每个点（即每个向量）都能代表一个概念。例如在嵌入模型中，一个点可以代表一个单词、一句话、一张图片、一个用户、一部电影等。

> 如何捕获语义上的相似性？
> 
> 欧式距离：两点之间的直线距离。（直线距离：相似度）
> 
> 余弦相似度：两个向量在方向上的差异。（方向：相似度      长度：文本长度）

常用：余弦相似度。因为还可以判断两个文本的长度。比如一个“开心”，和一篇开心的1000字文章。不应该相似！

##### （2）嵌入模型：

```python
from langchain_openai import OpenAIEmbeddings
 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 是 OpenAI 2024年发布的最新嵌入模型，生成3072维的高质量向量
)
```

在 LangChain 框架中**基础 Embeddings 类**（`OpenAIEmbeddings` 继承了它）设计了两个核心方法来处理文本嵌入。

-   `.embed_documents()`：用于处理文档 `Documents`。它的输入是多个文本。例如要将一个知识库里的所有段落都转换成向量后存入数据库，就会使用这个方法。
    
    -   它返回一个【二维列表】`List[List[float]]`。外层列表的每个元素对应一个输入文档，内层列表则是该文档的向量表示。
-   `.embed_query()`：用于处理查询 `Query`。它的输入是单个文本（一个字符串，`str`）。例如，当用户提出一个问题时，需要将这个问题转换成向量，以便在数据库中搜索相似的文档段落，就会使用这个方法。
    
    -   它返回一个【一维列表】，里面是浮点数（`List[float]`），代表单个查询文本的向量。

其实分别对应文档与查询的向量生成。

**① `.embed_documents()`**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
 
markdown_path = "../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md"
# single 模式加载后，默认只有一个 Document 对象
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
 
# 生成分割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=50
)
# 分割文档
documents = text_splitter.split_documents(data)
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 嵌入文档列表，生成向量列表
# 注意这里需要提取文档内容为字符串列表，才能传递给嵌入模型
texts = [doc.page_content for doc in documents]
documents_vector = embeddings.embed_documents(texts)
print(f"文档数量为: {len(documents)}，生成了{len(documents_vector)}个向量的列表")
print(f"第一个文档向量维度: {len(documents_vector[0])}")
print(f"第二个文档向量维度: {len(documents_vector[1])}")
```

**② `.embed_query()`**

```python
from langchain_openai import OpenAIEmbeddings
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 嵌入单个查询
query_vector = embeddings.embed_query("项目中遇到了哪些挑战？如何解决？")
print(f"向量维度: {len(query_vector)}")
print(f"向量前五个数值为: {query_vector[:5]}")
```

##### （3）向量存储：

        向量是被管理在专⻔的向量存储介质中，如向量数据库。向量存储的核⼼任务是解决⼀个传 统数据库（如MySQL）不擅⻓的问题：基于**内容的相似性搜索**（SimilaritySearch），⽽不是基于精确匹配的查询。

**① 常⻅的向量数据库核⼼机制如下：**

-   **专门的索引**
    
    -   是向量数据库的核心，不采用暴力搜索，会预先为所有向量构建特殊索引结构
    -   常用近似最近邻（ANN）搜索，牺牲少量精度换取极致速度，高概率找到高度相似向量
    -   通过聚类、分层、压缩等算法，将搜索范围从全库缩小至候选集
    -   类比：图书馆按分类找书，而非遍历所有书架
-   **向量相似度计算优化**
    
    -   底层使用高度优化的库执行向量运算
    -   典型代表：FAISS（Facebook AI 研究院开发）
    -   支持高效相似性搜索、聚类，快速处理大规模高维向量数据
-   **数据管理功能**
    
    -   CRUD 操作：支持增删改查，可动态更新向量数据
    -   元数据过滤：基于文档元数据（创建时间、作者、类别等）先筛选，再做向量搜索，提升准确性与效率
    -   可扩展性与持久化：支持分布式部署，处理海量数据，数据持久化不丢失
    -   集成方便：提供 gRPC、RESTful 等友好 API，可无缝对接 LangChain 等框架

**② 内存存储：**

        LangChain提供了 InMemoryVectorStore 来实现向量的内存存储。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 内存存储初始化
vector_store = InMemoryVectorStore(embedding=embeddings)
 
 
# 生成分割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=50
)
# 加载文档
data = UnstructuredMarkdownLoader("../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md").load()
# 分割文档
documents = text_splitter.split_documents(data)
# 添加文档
ids = vector_store.add_documents(documents=documents)
print(f"共编排了{len(ids)}个文档索引")
print(f"前3个文档的索引是: {ids[:3]}")
 
# 获取文档：
doc_3 = vector_store.get_by_ids(ids[:3])
print(f"{[doc.page_content for doc in doc_3]}")
 
# 删除文档：
vector_store.delete(ids=ids[:3])
 
# 相似性搜索：
search_docs = vector_store.similarity_search(query="数据库表怎么设计的？", k=2)
for doc in search_docs:
    print("*" * 30)
    print(doc.page_content)
 
```

**③  Redis向量数据库：**

        目前Redis提供了诸如搜索和查询功能，允许用户在Redis内创建二级索引结构，这使得Redis能够以缓存的速度充当向量数据库。

        RediSearch 是Redis官⽅提供的⼀款⾼性能【搜索】与【全⽂索引】引擎模块。它基于Redis构建， 使⽤⼾能够直接在Redis数据库中执⾏复杂的【搜索】和【分词查询】，⽆需额外引⼊外部搜索引 擎。RediSearch特别适⽤于轻量级、响应速度要求较⾼的分词搜索场景。
<img
  src="https://i-blog.csdnimg.cn/direct/edaf93eec0fc4998a96de393bfd22aac.png"
  referrerPolicy="no-referrer"
  alt=""
/>

Redis环境设置：

使用 Redis 来存储向量，首先需要将相关环境配置好。

第一步：启动 Redis 服务端

使用 Docker 启动 Redis 实例。

-   对于 **Redis 版本 >= 8.0**，使用：
    
    ```cobol
    docker run -d -p 6379:6379 -it redis:latest
    ```
    
-   对于 **Redis 版本 < 8.0**，使用：
    
    ```cobol
    docker run -d -p 6379:6379 redis/redis-stack:latest
    ```
    

使用 `docker ps` 查看是否启动成功。

___

第二步：安装 Redis 客户端包

由于使用 Python 开发，选择 `redis-py` 库完成客户端定义：

```undefined
pip install redis
```

___

第三步：安装 LangChain Redis 集成包

在 LangChain 中使用 Redis 向量库，需要安装 `langchain-redis`：

```undefined
pip install -qU langchain-redis
```

___

第四步：定义 Redis 连接 URL

Redis 连接 URL 基本结构：

```markdown
[protocol]://[auth]@[host]:[port]/[database]
```

```csharp
"redis://localhost:6379"
```

___

第五步：测试连接（Ping）

```python
import redis
 
redis_url = "redis://localhost:6379"
# 定义Redis客户端
redis_client = redis.from_url(redis_url)
# Ping
print(redis_client.ping())
```

若输出 `True`，则表示连接测试成功。

___

```python
"""
Redis 存储初始化
"""
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 配置 Redis 客户端
redis_url = "redis://192.168.100.238:6379"
config = RedisConfig(
    index_name="qa",
    redis_url=redis_url,
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)
 
# Redis 存储初始化
vector_store = RedisVectorStore(embeddings, config=config)
 
 
"""
向Redis数据库中添加数据
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
 
# 生成分割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=50
)
# 加载文档
data = UnstructuredMarkdownLoader("../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md", category="QA").load()
# 分割文档
documents = text_splitter.split_documents(data)
# 为文档添加元数据
for i, doc in enumerate(documents, start=1):
    doc.metadata["category"] = "QA"
    doc.metadata["num"] = i
 
ids = vector_store.add_documents(documents=documents)
print(f"共编排了{len(ids)}个文档索引")
print(f"前3个文档的索引是: {ids[:3]}")
 
 
 
"""
向Redis数据库中获取指定索引的数据
"""
ids = [":01K4Q0A3DSQVZBRFKJD5MS25HJ", ":01K4Q0A3DSQVZBRFKJD5MS25HKK", ":01K4Q0A3DSQVZBRFKJD5MS25HM"]
doc_3 = vector_store.get_by_ids(ids)
print(f"{[doc.page_content for doc in doc_3]}")
 
 
 
"""
删除指定索引的数据
"""
vector_store.delete([":01K4Q0A3DSQVZBRFKJD5MS25HJ"])
 
 
 
"""
相似性搜索
"""
search_docs = vector_store.similarity_search(query="数据库表怎么设计的？", k=2)
for doc in search_docs:
    print("*" * 30)
    print(doc.page_content)
 
```

④Pinecone 向量存储

        Pinecone 是为机器学习应⽤量⾝打造的⽣产级向量数据库服务，适⽤于⾼维向量数据的⾼效存储、索 引与查询。

        它屏蔽了基础设施管理，提供⽆缝扩展、实时数据写⼊和强⼤安全保障，让开发者和数据 科学家能够以极低运维成本，快速构建⾼效的相似度搜索、推荐系统和AI应⽤。

        Pinecone 是⼀个全托管的向量数据库平台，即负责所有后端维护、扩展、更新和监控，让⽤⼾专注于 应⽤开发，⽆需担⼼数据库管理。

        Pinecone 地址：https://www.pinecone.io/（魔法上⽹）

___

环境设置

-   ⾸次使⽤需注册新⽤⼾，选择个⼈免费版。
<img
  src="https://i-blog.csdnimg.cn/direct/470a8399b1f54b52837276c0fb54f891.png"
  referrerPolicy="no-referrer"
  alt=""
/>

-   继续创建账⼾相关信息，或者直接右上⻆skip跳过。这⾥我们直接跳过。
<img
  src="ttps://i-blog.csdnimg.cn/direct/c5fe2893c3b8442b98bc00a22dfc8149.png"
  referrerPolicy="no-referrer"
  alt=""
/>

-   注册成功会⽣成⼀个默认的APIKey。注意保存好你的key。
<img
  src="https://i-blog.csdnimg.cn/direct/29072ddb9e3a4ca4ac51fce66120f813.png"
  referrerPolicy="no-referrer"
  alt=""
/>
-   也可以创建新的API key。
<img
  src="https://i-blog.csdnimg.cn/direct/a31280d6d02245ec9e499d1f399493cf.png"
  referrerPolicy="no-referrer"
  alt=""
/>
-   设置 PINECONE\_API\_KEY ，将Key添加进环境变量。
<img
  src="https://i-blog.csdnimg.cn/direct/314542afd7804b07b3b1fa3110b8d5a6.png"
  referrerPolicy="no-referrer"
  alt=""
/>

更新包。

```python
pip install -qU pinecone langchain-pinecone
```

初始化：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
 
# 建立索引
pc = Pinecone()
index_name = "qa"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,         # 索引名称
        dimension=3072,           # 尺寸，表示向量维度，需要和嵌入模型维度一致
        metric="cosine",          # 度量方式，cosine 表示余弦相似度
        spec=ServerlessSpec(
            cloud="aws",         # 亚马逊云
            region="us-east-1"   # 区域
        ),
    )
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 获取索引
index = pc.Index(index_name)
# 定义 Pinecone 向量存储
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

添加文档：

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
 
# 生成分割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=50
)
# 加载文档
data = UnstructuredMarkdownLoader("../Docs/markdown/12.LangChain 模型流式输出（Streaming）笔记.md", category="QA").load()
# 分割文档
documents = text_splitter.split_documents(data)
# 为文档添加元数据
for i, doc in enumerate(documents, start=1):
    doc.metadata["category"] = "QA"
    doc.metadata["num"] = i
 
ids = vector_store.add_documents(documents=documents)
print(f"共编排了{len(ids)}个文档索引")
print(f"前3个文档的索引是: {ids[:3]}")
```

删除文档：

```python
# 全量删除
vector_store.delete(delete_all=True)
 
# 删除指定id的文档列表
delete_ids = []
vector_store.delete(ids=delete_ids)
```

向量搜索：

```python
search_docs = vector_store.similarity_search(
    query="数据库表怎么设计的？",
    k=2,
    filter={"category": "QA"},
)
for doc in search_docs:
    print("*" * 30)
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

___

#### 8.检索器：

接收来自用户接口的查询，检索出包含查询关键词的候选文档集合。

##### （1）向量数据库作为检索器：

向量存储是索引和检索⾮结构化数据的⼀种强⼤⽽有效的⽅法。可以通过调⽤向量数据库的 as\_retriever ⽅法，将向量存储⽤作检索器。在这⾥我们使⽤Redis向量存储。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
 
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 配置 Redis 客户端
redis_url = "redis://192.168.100.238:6379"
config = RedisConfig(
    index_name="qa",
    redis_url=redis_url,
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ]
)
 
# Redis 存储初始化
vector_store = RedisVectorStore(embeddings, config=config)
 
retriever = vector_store.as_retriever()
docs = retriever.invoke("数据库表怎么设计的？")
for doc in docs:
    print("*" * 30)
    print(doc.page_content[:30])
```

LangChain 检索器是⼀个Runnable的对象，它是LangChain组件的标准接⼝。这意味着它有⼀些常 ⽤⽅法，包括 invoke ，⽤于与其交互。默认情况下，向量存储检索器使⽤相似性搜索。

要注意的是， Retrievers 检索器虽然是Runnable对象，但其不提供任何流式处理，因为它本⾝ 通常是同步的、阻塞的操作。也就是说，你采用流式输出，他也是完成所有的查询之后一次性给你返回。

#####  （2）使用 `@chain` 创建“检索器”

除了使用 `as_retriever` 方法，我们还可以自行创建一个“检索器”。回想一下检索器的特点：

-   LangChain 检索器是一个 Runnable 的对象
-   LangChain 检索器输入为查询字符串，输出为文档列表（标准化的 LangChain 文档对象 Document）

综上所述，我们可以：

```python
from langchain_core.runnables import chain
from typing import List
from langchain_core.documents import Document
 
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=2)
 
docs = retriever.invoke("数据库表怎么设计的？")
 
for doc in docs:
    print("*" * 30)
    print(doc.page_content[:30])
```

上面定义了一个函数，使用 `@chain` 修饰，该修饰可以使其成为 Runnable 函数，且满足检索器输入输出的要求。在函数中，我们依旧使用向量数据库的相似性搜索方法，这样灵活性也更高，想要进行元数据筛选也更方便。

注意，这并不是真正的检索器，检索器是一个 Runnable 对象，而我们定义的只是一个函数，具备其特点罢了。

### 七、（总结）具体的RAG实现的实例

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
 
# 定义聊天模型
model = ChatOpenAI(model="gpt-4o-mini")
# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
 
# 配置 Redis 客户端
redis_url = "redis://192.168.100.238:6379"
config = RedisConfig(
    index_name="qa",
    redis_url=redis_url,
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)
# 定义 Redis 向量存储
vector_store = RedisVectorStore(embeddings, config=config)
# 生成检索器
retriever = vector_store.as_retriever()
 
# 定义提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """你是一个知识库问答助手，你需要根据提供的上下文回答问题。如果你不知道答案，请直接返回“不知道”。最多回复三句话的结果，回答要简洁。
            Question:{question}
            Context:{context}
            Answer:"""
        )
    ]
)
 
# 将文档转换为字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
# 定义链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
 
# 循环输入问题
while True:
    # 获取用户输入
    question = input("\n请输入您的问题（输入'退出'或'quit'结束程序）：").strip()
 
    # 检查是否退出
    if question.lower() in ["退出", "quit"]:
        print("程序已结束，再见！")
        break
 
    # 检查输入是否为空
    if not question:
        print("问题不能为空，请重新输入。")
        continue
 
    # 执行链，流式输出
    print("回答：", end="", flush=True)
    chunks = []
    for chunk in rag_chain.stream(question):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()  # 换行
```

上述代码中，RunnablePassthrough 我们之前还没有见过。

简单来说，RunnablePassthrough 是一个“伪”Runnable，它的主要作用是在链（Chain）中透明地传递输入数据，而不做任何修改。

当我们需要将原始输入和另一个处理过程的输出一起传递给下一个步骤时，就需要 RunnablePassthrough。

就例如代码中，我们需要将【Query】与【通过检索出来的文档转换的字符串】同时发送给提示词模板。

___

        **看到这里的各位大佬支持一下，点赞关注加收藏，你的支持就是我创作的动力！**