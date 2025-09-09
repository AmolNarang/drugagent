# 💊 Drug Analysing Agent  

This project is an **AI-powered agent** that collects patient discussions from **Reddit**, extracts drug-related side effects, conditions, and alternative medications, and generates a **comprehensive summary report**. It also enables **natural language querying** of patient experiences using vector search and LLM reasoning.  

---

## 🚀 Features  
- 🔎 **Reddit Search** – Scrape posts & comments related to a given drug.  
- 🧠 **Entity Extraction (LLM)** – Identify **side effects**, **conditions**, and **other drugs** from posts.  
- 📂 **CSV Export** – Save processed data in a structured CSV format.  
- 📊 **Summary Report** – Auto-generate a markdown report highlighting key findings.  
- 📚 **Vector Search (ChromaDB)** – Index posts for semantic search.  
- 💬 **Natural Language Querying** – Ask drug-related questions and get answers with cited patient sources.  
- 🌐 **Streamlit UI** – Interactive interface to run analysis and explore results.  

---

## 🛠️ Tech Stack  
- **LangGraph** – Workflow orchestration  
- **LangChain + Google Gemini** – LLM-powered entity extraction & summarization  
- **PRAW** – Reddit API wrapper  
- **SentenceTransformers** – Embedding generation (`all-MiniLM-L6-v2`)  
- **ChromaDB** – Vector database for querying  
- **Streamlit** – Frontend interface  
- **Pandas** – Data processing & CSV export  

---

## 📂 Project Structure  
├── drugworkflow.py # Core workflow logic & tools
├── app.py # Streamlit interface
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Setup  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/your-username/drug-analysing-agent.git
cd drug-analysing-agent

2️⃣ Install dependencies

👉 First, install PyTorch (CPU version):

pip install torch --index-url https://download.pytorch.org/whl/cpu

👉 Then install the rest of the dependencies:

pip install -r requirements.txt

3️⃣ Configure environment variables

Create a .env file in the project root:

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password

GEMINI_API_KEY=your_google_gemini_api_key

4️⃣ Run the Streamlit app
streamlit run app.py
