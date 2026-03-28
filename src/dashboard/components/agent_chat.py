"""
Agent Chat Dashboard Component.

Provides an interactive chat interface for the RAG-powered agent.
"""

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def render_agent_page():
    """Render the AI Agent chat interface page."""
    st.markdown(
        '<p class="section-header">🤖 AI Agent</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            <p class="info-box-title">ℹ️ NVIDIA Stock Analysis Agent</p>
            <p>Ask questions about NVIDIA stock data, model predictions,
            market trends, and technical analysis. Powered by RAG + ReAct agent.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about NVIDIA stock, predictions, or market analysis...")

    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _get_agent_response(user_input)
                st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🤖 Agent Controls")

        if st.button("🗑️ Clear Chat", key="btn_clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

        # Example queries
        st.markdown("#### 💡 Example Queries")
        examples = [
            "What is NVIDIA's current stock trend?",
            "Summarize the latest model predictions",
            "Is there any data drift detected?",
            "What are the model's key metrics?",
        ]
        for example in examples:
            if st.button(f"📝 {example}", key=f"ex_{hash(example)}"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": example}
                )
                st.rerun()


def _get_agent_response(query: str) -> str:
    """Get response from the RAG agent.

    Args:
        query: User's question.

    Returns:
        Agent's response text.
    """
    try:
        from src.agent.react_agent import create_agent

        agent = create_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "I couldn't generate a response. Please try again.")
    except ImportError:
        return (
            "⚠️ Agent dependencies not available. Please install LangChain "
            "and configure your LLM API key to use this feature.\n\n"
            "```bash\npip install langchain langchain-openai\n```"
        )
    except Exception as e:
        logger.error("Agent error: %s", str(e))
        return (
            f"⚠️ Agent encountered an error: {str(e)}\n\n"
            "Make sure your LLM API key is configured in `.env`."
        )
