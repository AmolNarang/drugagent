# app.py
import streamlit as st
from drugworkflow import create_drug_agent_workflow, query_drug_data
from langchain_core.messages import HumanMessage

# App title
st.set_page_config(page_title="Drug Agent", layout="wide")
st.title("💊 Drug Analysing Agent")

# Get user input for drug name
drug_name = st.text_input("Enter the drug name to analyze", "")

# Run button
if st.button("🚀 Run Analysis") and drug_name:
    with st.spinner("Running the workflow... Please wait."):
        # Build LangGraph workflow
        app = create_drug_agent_workflow()

        # Initial state for the agent
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze side effects for {drug_name}")],
            "drug_name": drug_name,
            "posts_data": [],
            "processed_data": [],
            "summary": {},
            "output_file": "",
            "query_mode": False,
            "collection": None,
            "exit_query_mode": False
        }

        # Run the workflow
        final_state = app.invoke(initial_state)

    st.success("✅ Workflow completed successfully!")

    # Save state for querying
    st.session_state["collection"] = final_state.get("collection")
    st.session_state["drug_name"] = drug_name

    # Show summary
    if final_state.get("summary"):
        summary = final_state["summary"]
        st.subheader("📋 Summary Report")
        st.markdown(summary.get("summary_report", "No summary available."))

    # Show sample extracted data
    if final_state.get("processed_data"):
        st.subheader("🔬 Sample Extracted Data")
        sample = final_state["processed_data"][0]
        st.markdown(f"**Post Title:** {sample['post_title']}")
        st.markdown(f"[🔗 Reddit URL]({sample['post_url']})")
        st.markdown(f"**Side Effects:** {sample['side_effects']}")
        st.markdown(f"**Other Drugs:** {sample['other_drugs']}")
        st.markdown(f"**Conditions:** {sample['conditions']}")

    # Show CSV file path
    if final_state.get("output_file"):
        st.info(f"📁 CSV saved to: `{final_state['output_file']}`")


# Query interface (if collection exists)
if "collection" in st.session_state and st.session_state["collection"]:
    st.markdown("---")
    st.subheader("🔍 Ask a Question About This Drug")

    query = st.text_input("Your question about the drug", key="query_input")

    if st.button("💬 Get Answer") and query:
        with st.spinner("Thinking..."):
            result = query_drug_data.invoke({
                "question": query,
                "drug_name": st.session_state["drug_name"],
                "collection": st.session_state["collection"],
                "max_results": 5
            })

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.markdown("### ✅ Answer")
            st.markdown(result["answer"])

            st.markdown("### 📚 Sources")
            for i, src in enumerate(result["sources"]):
                with st.expander(f"Source {i + 1}: {src['title']}"):
                    st.markdown(f"**Author**: {src['author']}")
                    st.markdown(f"**Date**: {src['date']}")
                    st.markdown(f"**Subreddit**: {src['subreddit']}")
                    st.markdown(f"**Side Effects**: {src['side_effects']}")
                    st.markdown(f"**Conditions**: {src['conditions']}")
                    st.markdown(f"**Other Drugs**: {src['other_drugs']}")
                    st.markdown(f"[🔗 Reddit Post URL]({src['url']})")

