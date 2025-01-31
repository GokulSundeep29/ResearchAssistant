# 🔍 Multi-Agent Research Assistant 🤖📚

![Workflow Diagram](https://via.placeholder.com/800x400.png?text=Research+Workflow+Diagram+%7C+Add+Your+Architecture+Image+Here)

An AI-powered research assistant combining multiple data sources and validation agents for reliable information analysis. 🌐🔬✅

## 🚀 Features
- **📥 Multi-Source Retrieval**: Wikipedia 📚 + Arxiv 📑 + Web Search 🌐 (via Tavily)
- **🧠 AI Processing**: GPT-4 powered summarization ✍️ + fact-checking ✅
- **🔍 Validation Pipeline**: Error detection 🚨 + auto-recovery ♻️
- **❓ Follow-up Q&A**: Context-aware RAG model 💬
- **🗃️ Knowledge Storage**: Chroma vector DB integration 💾
- **🖥️ Web Interface**: Streamlit-powered UI 🎨

## 🛠️ Technologies
- ![LangChain](https://img.shields.io/badge/LangChain-FF6F00?style=flat&logo=langchain&logoColor=white) Agent Orchestration
- ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white) GPT Models
- ![Chroma](https://img.shields.io/badge/Chroma-FF6B6B?style=flat) Vector DB
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) Web UI
- ![Tavily](https://img.shields.io/badge/Tavily-00C7B7?style=flat) Web Search

## ⚙️ Installation

```bash
# 1️⃣ Clone repo
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Configure environment
cp .env.example .env
# 🖋️ Edit .env with your API keys


# 🚀 Launch the app
streamlit run research_assistant.py

# 🗺️ Architecture
%%{init: {'theme': 'forest'}}%%
graph TD
    A[User Question] --> B[Wikipedia Loader]
    A --> C[Arxiv Loader]
    A --> D[Query Translator]
    D --> E[Tavily Web Search]
    B --> F[Retriever]
    C --> F
    E --> F
    F --> G[Summarizer]
    G --> H[Fact Checker]
    H --> I[Error Detector]
    I -->|Errors Found| A
    I -->|Clean Output| J[📄 Results]