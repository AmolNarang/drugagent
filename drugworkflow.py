# drug_agentic_workflow.py
import dotenv
dotenv.load_dotenv()
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from datetime import datetime
import time
import random
import praw
import json
import re
import os
from collections import Counter
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# Configuration (should use environment variables in production)
REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT"),
    "username": os.getenv("REDDIT_USERNAME"),
    "password": os.getenv("REDDIT_PASSWORD")
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
reddit = praw.Reddit(**REDDIT_CONFIG)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Define the state
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    drug_name: str
    posts_data: List[dict]
    processed_data: List[dict]
    summary: Dict[str, Any]
    output_file: str
    query_mode: bool
    collection: Any  # Store the Chroma collection for querying
    exit_query_mode: bool  # ✅ NEW FIELD



# Define tools
@tool
def search_reddit_for_drug(drug_name: str, limit_per_subreddit: int = 5) -> List[dict]:
    """
    Search Reddit for posts about a specific drug and its side effects.

    Args:
        drug_name: The name of the drug to search for
        limit_per_subreddit: Number of posts to retrieve from each subreddit

    Returns:
        List of dictionaries containing post data
    """
    all_posts_data = []
    subreddits_to_search = ["askdrugs", "Drugs", "ChronicIllness", "Anxiety", "depression", "Pharmacy", "Humira"]
    search_query = f"title:{drug_name} side effects"

    print(f"Searching Reddit for '{search_query}'...")

    for subreddit_name in subreddits_to_search:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(query=search_query, limit=limit_per_subreddit, sort='relevance'):
                print(f"Processing post from r/{subreddit_name}: {submission.title}")

                # Get the main post content
                post_text = f"Title: {submission.title}\n\n{submission.selftext}"
                post_url = f"https://www.reddit.com{submission.permalink}"
                post_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                post_author = str(submission.author) if submission.author else "[deleted]"

                # Get comments
                submission.comments.replace_more(limit=0)
                comments_text = ""
                for comment in submission.comments.list():
                    if comment.body != "[deleted]":
                        comments_text += f"\n--- Comment by {comment.author} ---\n{comment.body}\n"

                full_text = f"{post_text}\n\n=== COMMENTS ==={comments_text}"

                post_data = {
                    'drug_name': drug_name,
                    'source': 'reddit',
                    'subreddit': subreddit_name,
                    'post_author': post_author,
                    'post_date': post_date,
                    'post_title': submission.title,
                    'post_url': post_url,
                    'full_text': full_text
                }
                all_posts_data.append(post_data)

                # Be polite and avoid hitting rate limits
                time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"Error searching in r/{subreddit_name}: {e}")
            continue

    return all_posts_data


@tool
def extract_entities_with_llm(text: str, drug_name: str) -> dict:
    """
    Use LLM to extract side effects, other drugs, and medical conditions from text.

    Args:
        text: The text to analyze
        drug_name: The name of the drug being discussed

    Returns:
        Dictionary with extracted entities
    """
    prompt = f"""
    Analyze the following text from a patient discussion about {drug_name}.
    Extract detailed information about side effects, other medications, and medical conditions.

    TEXT:
    {text[:2000]}

    Extract the following information as a JSON object with EXACTLY this format:
    {{
      "other_drugs": ["drug1", "drug2", "drug3"],
      "side_effects": ["specific side effect 1", "detailed description of side effect", "another side effect with context"], 
      "conditions": ["condition1", "condition2", "condition3"]
    }}

    CRITICAL INSTRUCTIONS:
    1. For side_effects: Extract PHRASES not just single words. Capture the exact descriptions patients use.
    2. Preserve the patient's original wording when possible to maintain context.
    3. Include both medical terms and layperson descriptions.
    4. Return ONLY the JSON object, nothing else. No additional text, no explanations.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Clean and extract JSON from the response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try to find JSON within the text using regex
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        else:
            # If no JSON found, try to parse the whole response
            result = json.loads(response_text)

        return result

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from response. Raw response: '{response_text}'")
        print(f"JSON decode error: {e}")
        return {"other_drugs": [], "side_effects": [], "conditions": []}
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"other_drugs": [], "side_effects": [], "conditions": []}


@tool
def save_processed_data(processed_data: List[dict], drug_name: str) -> str:
    """
    Save processed data to a CSV file.

    Args:
        processed_data: List of dictionaries with processed post data
        drug_name: Name of the drug for filename

    Returns:
        Path to the saved file
    """
    if not processed_data:
        return "No data to save"

    output_file = f"reddit_{drug_name.replace(' ', '_')}_posts_processed.csv"
    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    return output_file


@tool
def generate_summary_report(processed_data: List[dict], drug_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report of side effects, conditions, and alternative drugs.

    Args:
        processed_data: List of processed post data
        drug_name: The name of the drug being analyzed

    Returns:
        Dictionary containing the summary report
    """
    if not processed_data:
        return {"error": "No data available for summary"}

    # Extract all entities for analysis
    all_side_effects = []
    all_conditions = []
    all_other_drugs = []

    for post in processed_data:
        if post.get('side_effects'):
            all_side_effects.extend([se.strip() for se in post['side_effects'].split('|') if se.strip()])
        if post.get('conditions'):
            all_conditions.extend([c.strip() for c in post['conditions'].split('|') if c.strip()])
        if post.get('other_drugs'):
            all_other_drugs.extend([d.strip() for d in post['other_drugs'].split('|') if d.strip()])

    # Count frequencies
    side_effect_counts = Counter(all_side_effects)
    condition_counts = Counter(all_conditions)
    other_drug_counts = Counter(all_other_drugs)

    # Prepare data for LLM analysis
    summary_data = {
        "total_posts_analyzed": len(processed_data),
        "side_effects": dict(side_effect_counts.most_common(10)),
        "conditions": dict(condition_counts.most_common(10)),
        "other_drugs": dict(other_drug_counts.most_common(10)),
        "most_common_side_effects": [se for se, count in side_effect_counts.most_common(5)],
        "most_common_conditions": [c for c, count in condition_counts.most_common(5)],
        "most_common_alternative_drugs": [d for d, count in other_drug_counts.most_common(5)]
    }

    # Use LLM to generate a comprehensive summary
    prompt = f"""
    Analyze the following data collected from patient discussions about {drug_name} and generate a comprehensive summary report.

    DATA:
    {json.dumps(summary_data, indent=2)}

    Please provide a comprehensive summary report with the following sections:

    1. EXECUTIVE SUMMARY: Brief overview of key findings
    2. MOST COMMON SIDE EFFECTS: List and describe the most frequently reported side effects
    3. PATIENT CONDITIONS: Summarize the medical conditions patients are treating with this drug
    4. ALTERNATIVE/COMPLEMENTARY DRUGS: Identify other drugs patients are taking alongside or instead of {drug_name}
    5. PATIENT EXPERIENCE INSIGHTS: Key themes and patterns in patient experiences
    6. RECOMMENDATIONS: Any insights that might be useful for patients or healthcare providers

    Format your response as a well-structured markdown document.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_report = response.content

        # Save the summary to a file
        summary_file = f"{drug_name.replace(' ', '_')}_summary_report.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)

        return {
            "summary_report": summary_report,
            "summary_file": summary_file,
            "statistics": summary_data
        }

    except Exception as e:
        print(f"Error generating summary: {e}")
        return {"error": f"Failed to generate summary: {str(e)}"}


@tool
def index_data_for_querying(processed_data: List[dict], drug_name: str) -> Dict[str, Any]:
    """
    Index the processed data for querying using vector embeddings.

    Args:
        processed_data: List of processed post data
        drug_name: Name of the drug being analyzed

    Returns:
        Dictionary with collection info
    """
    if not processed_data:
        return {"error": "No data to index"}

    print("Indexing data for querying...")

    # Initialize ChromaDB client with persistence
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get collection
    collection_name = f"drug_posts_{drug_name.lower().replace(' ', '_')}"
    try:
        collection = chroma_client.get_collection(collection_name)
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(collection_name)

    # Prepare documents for indexing
    documents = []
    metadatas = []
    ids = []

    for i, post in enumerate(processed_data):
        # Create a comprehensive text for embedding
        text_content = f"""
        Drug: {post['drug_name']}
        Title: {post['post_title']}
        Side Effects: {post.get('side_effects', '')}
        Conditions: {post.get('conditions', '')}
        Other Drugs: {post.get('other_drugs', '')}
        Author: {post['post_author']}
        Date: {post['post_date']}
        Subreddit: {post['subreddit']}
        """

        documents.append(text_content)
        metadatas.append({
            "drug_name": post['drug_name'],
            "title": post['post_title'],
            "side_effects": post.get('side_effects', ''),
            "conditions": post.get('conditions', ''),
            "other_drugs": post.get('other_drugs', ''),
            "author": post['post_author'],
            "date": post['post_date'],
            "subreddit": post['subreddit'],
            "url": post['post_url']
        })
        ids.append(str(i))

    # Generate embeddings
    embeddings = embedding_model.encode(documents).tolist()

    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    return {
        "status": "success",
        "message": f"Indexed {len(processed_data)} posts for querying",
        "collection_name": collection_name,
        "collection": collection
    }


@tool
def query_drug_data(question: str, drug_name: str, collection: Any, max_results: int = 5) -> Dict[str, Any]:
    """
    Query the indexed drug data using natural language questions.

    Args:
        question: The natural language question to ask
        drug_name: Name of the drug being queried
        collection: ChromaDB collection object
        max_results: Maximum number of results to return

    Returns:
        Dictionary with query results and answer
    """
    print(f"Processing query: {question}")

    # Generate embedding for the question
    query_embedding = embedding_model.encode([question]).tolist()[0]

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results,
        include=["documents", "metadatas", "distances"]
    )

    # Prepare context from results
    context = ""
    source_details = []

    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context += f"SOURCE {i + 1}:\n{doc}\n\n"
        source_details.append({
            "title": metadata.get('title', ''),
            "author": metadata.get('author', ''),
            "date": metadata.get('date', ''),
            "subreddit": metadata.get('subreddit', ''),
            "url": metadata.get('url', ''),
            "side_effects": metadata.get('side_effects', ''),
            "conditions": metadata.get('conditions', ''),
            "other_drugs": metadata.get('other_drugs', '')
        })

    # Use LLM to generate an answer based on the context
    prompt = f"""
    You are a helpful assistant that answers questions about drug side effects based on patient discussions.
    Use the following information from patient posts to answer the question. Always cite your sources.

    QUESTION: {question}

    CONTEXT FROM PATIENT DISCUSSIONS:
    {context}

    Please provide a comprehensive answer to the question based on the patient discussions above.
    Include specific examples and cite your sources using the source numbers.
    If the context doesn't contain relevant information, say so clearly.

    Format your response with:
    1. A direct answer to the question
    2. Supporting evidence from the patient discussions with source citations
    3. A summary of key findings
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        return {
            "question": question,
            "answer": answer,
            "sources": source_details,
            "total_sources_found": len(source_details)
        }

    except Exception as e:
        print(f"Error generating answer: {e}")
        return {"error": f"Failed to generate answer: {str(e)}"}


# Bind tools to LLM
llm_with_tools = llm.bind_tools([
    search_reddit_for_drug,
    extract_entities_with_llm,
    save_processed_data,
    generate_summary_report,
    index_data_for_querying,
    query_drug_data
])


# Define the nodes
import sys
# ... other imports ...

def search_node(state: AgentState):
    """Agent that searches for drug-related posts"""
    print(f"Searching for posts about {state['drug_name']}...")

    # Use the tool to search Reddit
    posts_data = search_reddit_for_drug.invoke({
        "drug_name": state["drug_name"],
        "limit_per_subreddit": 5
    })

    # If no posts found -> exit immediately
    if not posts_data:
        msg = f"No posts found for '{state['drug_name']}'. Exiting and returning to shell."
        st.write(msg)
        sys.exit(0)   # <- immediate stop

    return {"posts_data": posts_data}



def process_node(state: AgentState):
    """Agent that processes posts with LLM"""
    print(f"Processing {len(state['posts_data'])} posts with LLM...")

    processed_data = []

    for i, post in enumerate(state["posts_data"]):
        print(f"Processing post {i + 1}/{len(state['posts_data'])}...")

        # Use the tool to extract entities
        extracted_data = extract_entities_with_llm.invoke({
            "text": post['full_text'],
            "drug_name": post['drug_name']
        })

        result_row = {
            'drug_name': post['drug_name'],
            'source': post['source'],
            'subreddit': post['subreddit'],
            'post_author': post['post_author'],
            'post_date': post['post_date'],
            'post_title': post['post_title'],
            'post_url': post['post_url'],
            'other_drugs': " | ".join(extracted_data.get('other_drugs', [])),
            'side_effects': " | ".join(extracted_data.get('side_effects', [])),
            'conditions': " | ".join(extracted_data.get('conditions', []))
        }
        processed_data.append(result_row)

        # Rate limiting
        if i < len(state["posts_data"]) - 1:
            time.sleep(3)

    return {"processed_data": processed_data}


def save_node(state: AgentState):
    """Agent that saves processed data"""
    print("Saving processed data...")

    # Use the tool to save data
    output_file = save_processed_data.invoke({
        "processed_data": state["processed_data"],
        "drug_name": state["drug_name"]
    })

    return {"output_file": output_file}


def summary_node(state: AgentState):
    """Agent that generates a comprehensive summary"""
    print("Generating comprehensive summary report...")

    # Use the tool to generate summary
    summary = generate_summary_report.invoke({
        "processed_data": state["processed_data"],
        "drug_name": state["drug_name"]
    })

    return {"summary": summary}


def index_node(state: AgentState):
    """Agent that indexes data for querying"""
    print("Indexing data for querying...")

    # Use the tool to index data
    index_result = index_data_for_querying.invoke({
        "processed_data": state["processed_data"],
        "drug_name": state["drug_name"]
    })

    if "error" in index_result:
        print(f"Error indexing data: {index_result['error']}")
        return {}

    print(index_result["message"])
    return {"query_mode": True, "collection": index_result["collection"]}


def query_node(state: AgentState):
    """Placeholder for query mode — handled by Streamlit frontend"""
    print("Query mode activated. Ready for questions in Streamlit UI.")
    return {"exit_query_mode": True}



def supervisor_node(state: AgentState):
    print("Supervising the drug analysis workflow...")

    if state.get("exit_query_mode", False):
        print("User exited query mode. Ending workflow.")
        return {"next": "end"}

    # Determine next step based on current state
    if not state.get("posts_data"):
        return {"messages": [HumanMessage(content="Please search for drug posts first.")]}
    elif not state.get("processed_data"):
        return {"messages": [HumanMessage(content="Please process the found posts.")]}
    elif not state.get("output_file"):
        return {"messages": [HumanMessage(content="Please save the processed data.")]}
    elif not state.get("summary"):
        return {"messages": [HumanMessage(content="Please generate a summary report.")]}
    elif not state.get("query_mode"):
        return {"messages": [HumanMessage(content="Please index data for querying.")]}
    else:
        return {"messages": [HumanMessage(content="Ready for query mode.")]}


# Create the graph
def create_drug_agent_workflow():
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search", search_node)
    workflow.add_node("process", process_node)
    workflow.add_node("save", save_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("index", index_node)
    workflow.add_node("query", query_node)

    # Define the flow
    workflow.set_entry_point("supervisor")

    # Add conditional edges based on state
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: "end" if state.get("exit_query_mode") else
        "search" if not state.get("posts_data") else
        "process" if not state.get("processed_data") else
        "save" if not state.get("output_file") else
        "summary" if not state.get("summary") else
        "index" if not state.get("query_mode") else
        "query",
        {
            "search": "search",
            "process": "process",
            "save": "save",
            "summary": "summary",
            "index": "index",
            "query": "query",
            "end": END  # valid end node
        }
    )

    # Add edges
    workflow.add_edge("search", "supervisor")
    workflow.add_edge("process", "supervisor")
    workflow.add_edge("save", "supervisor")
    workflow.add_edge("summary", "supervisor")
    workflow.add_edge("index", "supervisor")
    workflow.add_edge("query", "supervisor")

    # Compile the graph
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Get drug name from user
    drug_name = input("Enter the drug name to search for: ").strip()
    if not drug_name:
        print("No drug name provided. Exiting.")
        exit()

    # Create and run the workflow
    app = create_drug_agent_workflow()

    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=f"Analyze side effects for {drug_name}")],
        "drug_name": drug_name,
        "posts_data": [],
        "processed_data": [],
        "summary": {},
        "output_file": "",
        "query_mode": False,
        "collection": None
    }

    # Execute the workflow
    print("Starting drug analysis workflow...")
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 100}
    )

    # Display results
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"Processed {len(final_state.get('processed_data', []))} posts")
    print(f"Data saved to: {final_state.get('output_file', 'Unknown')}")

    # Display summary information
    if final_state.get('summary'):
        summary = final_state['summary']
        if 'summary_file' in summary:
            print(f"Summary report saved to: {summary['summary_file']}")

        if 'statistics' in summary:
            stats = summary['statistics']
            print(f"\nKey Statistics:")
            print(f"- Total posts analyzed: {stats.get('total_posts_analyzed', 0)}")
            print(f"- Most common side effects: {', '.join(stats.get('most_common_side_effects', []))}")
            print(f"- Most common conditions: {', '.join(stats.get('most_common_conditions', []))}")
            print(f"- Most common alternative drugs: {', '.join(stats.get('most_common_alternative_drugs', []))}")

    # Display sample data
    if final_state.get('processed_data'):
        sample = final_state['processed_data'][0]
        print(f"\nSample extracted data:")
        print(f"Post: {sample.get('post_title')}")
        print(f"URL: {sample.get('post_url')}")
        print(f"Side effects: {sample.get('side_effects')}")
        print(f"Other drugs: {sample.get('other_drugs')}")
        print(f"Conditions: {sample.get('conditions')}")