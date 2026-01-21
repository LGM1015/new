import streamlit as st
import os
import tempfile
import datetime

# 1. 环境变量配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["TAVILY_API_KEY"] = "你的Key" # 确保 Key 存在

from dotenv import load_dotenv
# RAG 相关
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# Agent 相关
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
# 👇 新增：文件管理工具箱
from langchain_community.agent_toolkits import FileManagementToolkit

# 基础组件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==========================================
# ⚙️ 全局配置：定义工作区路径
# ==========================================
# Agent 只能在这个文件夹里读写文件，保证安全
WORKSPACE_DIR = "./agent_workspace"
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(page_title="超级 AI 员工", page_icon="💼", layout="wide")
st.title("💼 超级 AI 员工 (RAG + 联网 + 写文件)")


# ==========================================
# 核心功能函数
# ==========================================

def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7
    )


# 1. 构建 RAG 链 (保持不变)
@st.cache_resource
def create_rag_chain(file_paths):
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "你是一个专业的文档助手。请根据以下【上下文】回答问题。"
        "如果【上下文】里没有答案，请诚实地说不知道。"
        "\n\n【上下文】: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        temperature=0
    )
    question_answer_chain = create_stuff_documents_chain(rag_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


# 2. 构建全能 Agent (联网 + 文件系统)
def create_general_agent():
    llm = get_llm()

    # A. 搜索工具
    search_tool = TavilySearchResults(max_results=3)

    # B. 文件系统工具 (限制在 WORKSPACE_DIR 目录下)
    # 包含：write_file, read_file, list_directory 等工具
    file_toolkit = FileManagementToolkit(root_dir=WORKSPACE_DIR)
    file_tools = file_toolkit.get_tools()

    # 合并工具集
    tools = [search_tool] + file_tools

    current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"你是一个全能型 AI 助手。当前时间：{current_time}。\n"
            "你有两个核心能力：\n"
            "1. **联网搜索**：使用 search 工具获取实时信息。\n"
            "2. **文件管理**：使用 write_file 工具在本地工作区创建文件，用 list_directory 查看文件。\n"
            "⚠️ 只有当用户明确要求'生成报告'、'保存文件'或'写代码'时，才使用文件工具。\n"
            "⚠️ 默认将文件保存为 Markdown (.md) 或 Python (.py) 格式。"
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# 3. 普通闲聊
def create_chat_chain():
    current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个友好的 AI 助手。当前时间：{current_time}。"),
        ("human", "{input}"),
    ])
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain


# ==========================================
# 侧边栏：控制中心 & 文件浏览器
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制台")

    # --- RAG 部分 ---
    st.subheader("📚 知识库")
    uploaded_files = st.file_uploader("上传 PDF", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("🔄 构建知识库"):
            st.session_state.is_processing = True

    # --- Agent 开关 ---
    st.subheader("🤖 智能体能力")
    # 如果没传文件，可以让用户选择开启 Agent 模式
    enable_agent = st.toggle("开启全能 Agent (联网+写文件)", value=False)

    st.divider()

    # --- 📂 文件浏览器 (新功能) ---
    st.subheader("📂 本地工作区")
    st.caption(f"路径: {WORKSPACE_DIR}")

    # 刷新文件列表按钮
    if st.button("🔄 刷新文件列表"):
        st.rerun()

    # 列出工作区的所有文件
    try:
        files = os.listdir(WORKSPACE_DIR)
        if not files:
            st.info("暂无文件")
        else:
            for f in files:
                file_path = os.path.join(WORKSPACE_DIR, f)
                # 简单的下载/查看逻辑
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ 下载 {f}",
                        data=file,
                        file_name=f,
                        mime="text/plain"
                    )
    except Exception as e:
        st.error(f"无法读取工作区: {e}")

    st.divider()
    if st.button("🗑️ 清空历史"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()

# ==========================================
# 主逻辑
# ==========================================

# 初始化 Session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# RAG 处理
if getattr(st.session_state, 'is_processing', False):
    with st.spinner("正在构建知识库..."):
        temp_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)
        st.session_state.rag_chain = create_rag_chain(temp_paths)
        st.success("✅ 知识库就绪！")
        st.session_state.is_processing = False

# 模式判断
current_mode = "chat"
if st.session_state.rag_chain:
    current_mode = "rag"
elif enable_agent:
    current_mode = "agent"  # Agent 模式 = 搜索 + 文件操作

# 状态显示
if current_mode == "rag":
    st.info("🟢 模式：**知识库问答** (RAG)")
elif current_mode == "agent":
    st.success("🌍 模式：**全能 Agent** (联网搜索 + 文件读写)")
else:
    st.caption("🔵 模式：**自由闲聊** (Chat)")

# 历史回显
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📖 来源"):
                for s in msg["sources"]:
                    st.markdown(f"**P{s['page']}**: {s['content']}...")

# 输入处理
if prompt := st.chat_input("指令..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在执行任务..."):

            # 1. RAG 模式
            if current_mode == "rag":
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                sources = [{"page": d.metadata.get("page", 0) + 1, "content": d.page_content[:50]} for d in
                           response["context"]]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

            # 2. Agent 模式 (搜索 + 写文件)
            elif current_mode == "agent":
                agent = create_general_agent()
                # Agent 的输出通常包含执行过程，我们取 output
                response = agent.invoke({"input": prompt})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # 🎉 如果生成了文件，自动提示刷新侧边栏
                if "write_file" in str(response):  # 简单判断日志里有没有调用写文件
                    st.toast("✅ 文件已生成！请在左侧侧边栏查看。", icon="📂")

            # 3. 闲聊模式
            else:
                chat = create_chat_chain()
                response = chat.invoke({"input": prompt})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})