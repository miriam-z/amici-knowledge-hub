import chainlit as cl
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from dotenv import load_dotenv
import yaml
import os
import logging

# Load environment variables
load_dotenv(".env")

# Global variable for selected mode
selected_mode = "default"

# Load source mappings from YAML
with open("sources.yaml", "r") as f:
    source_configs = yaml.safe_load(f)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def get_source_display(source_url):
    for config in source_configs:
        domain = config["url"].split("/")[2]
        if domain in source_url:
            return config.get("display_name", "Other Source")
    return "Other Source"


# MongoDB setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client["test_qa"]
collection = db["stable_diffusion_qna"]

# Vector store setup
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection, embedding=embeddings, index_name="default"
)

retriever = vectorStore.as_retriever()

# LLM setup
llm = ChatOpenAI(
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    streaming=True,
)

# Prompts
history_aware_prompt = PromptTemplate.from_template(
    """
Given the chat history and a new question, rewrite the new question to be a standalone question.

Chat history:
{chat_history}

Follow-up question:
{input}

Standalone question:
"""
)

qa_prompt = PromptTemplate.from_template(
    """
You are an expert assistant for the AIMICI Companion app, designed to help film & TV professionals responsibly use AI.

Given the user's role and project details, provide tailored advice and training on creative AI tools (Stable Diffusion, MidJourney) and AI safety guidelines.

- Use the user's role and project context to customize your response.
- Focus on providing clear, actionable advice.
- Maintain a professional tone, suitable for an educational setting.
- Keep responses concise, directly addressing the user's question without unnecessary detail.
- Ensure the response aligns with the educational goals of the app, emphasizing AI safety and responsible use.

Context:
{context}

Question:
{input}

Answer:
"""
)

# Create chains
retriever_chain = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=history_aware_prompt
)

document_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

qa_chain = create_retrieval_chain(
    retriever=retriever_chain, combine_docs_chain=document_chain
)

# Memory for conversation context
memory = ConversationBufferMemory(return_messages=True)


# Sidebar menu function
async def sidebar():
    logging.debug("Displaying sidebar with actions.")
    await cl.ChatSettings(
        actions=[
            cl.Action(
                name="foundational_ai",
                payload={"mode": "foundational"},
                label="Foundational AI Knowledge",
            ),
            cl.Action(
                name="ai_tools",
                payload={"mode": "tools"},
                label="Finding AI Tools",
            ),
            cl.Action(
                name="responsible_ai",
                payload={"mode": "responsible"},
                label="Responsible AI Use",
            ),
            cl.Action(
                name="reset", payload={"command": "reset"}, label="Reset conversation"
            ),
        ]
    ).send()


# On chat start
@cl.on_chat_start
async def start_chat():
    global selected_mode
    selected_mode = "default"

    logging.debug("Chat started. Displaying welcome message and sidebar.")

    # Initial welcome message with buttons
    await cl.Message(
        content=(
            "### Welcome to the **AI Tools Knowledge Hub**!\n\n"
            "Ask me anything about creative AI tools, AI safety, or best practices."
        ),
        elements=[
            cl.Action(
                name="foundational_ai",
                payload={"mode": "foundational"},
                label="Foundational AI Knowledge",
            ),
            cl.Action(
                name="ai_tools", payload={"mode": "tools"}, label="Finding AI Tools"
            ),
            cl.Action(
                name="responsible_ai",
                payload={"mode": "responsible"},
                label="Responsible AI Use",
            ),
        ],
    ).send()


async def handle_mode_selection(mode):
    # Define mode-specific prompts
    prompts = {
        "foundational": "Welcome to Foundational AI Knowledge. Here you will learn the basics of AI.",
        "tools": "Welcome to Finding AI Tools. Discover various AI tools available.",
        "responsible": "Welcome to Responsible AI Use. Learn about ethical AI practices.",
    }

    # Update system prompt based on selected mode
    system_prompt = prompts.get(mode, "")
    await cl.Message(content=system_prompt).send()


# Handle actions (sidebar button clicks)
@cl.action_callback("foundational_ai")
@cl.action_callback("ai_tools")
@cl.action_callback("responsible_ai")
async def handle_action(action):
    global selected_mode

    payload = action.payload

    if payload.get("mode"):
        selected_mode = payload["mode"]
        await handle_mode_selection(selected_mode)

    if payload.get("command") == "reset":
        memory.clear()
        await cl.Message(
            content="Conversation history cleared. Let's start fresh!"
        ).send()

    # Show sidebar
    await sidebar()


# Main message handler
@cl.on_message
async def handle_message(message: cl.Message):
    global selected_mode

    user_input = message.content.strip().lower()

    # Check for affirmative responses and handle them
    if (
        "achievable approach" in user_input
        or "sounds good" in user_input
        or "thank you" in user_input
    ):
        # Acknowledge the user's positive feedback
        await cl.Message(
            content="I'm glad to hear that! Is there anything else you'd like to know or discuss regarding AI tools or safety guidelines?"
        ).send()
    else:
        # Handle other types of messages normally
        history = memory.load_memory_variables({}).get("chat_history", [])
        inputs = {"input": user_input, "chat_history": history}

        # Run chain
        result = qa_chain.invoke(inputs)

        answer = result["answer"]
        source_docs = result.get(
            "context", []
        )  # Assuming context contains source information

        # Append unique sources as hyperlinks to the answer
        if source_docs:
            unique_sources = {}
            for doc in source_docs:
                metadata = doc.metadata
                source = metadata.get("source", "Unknown")
                title = metadata.get("title", "Unknown Source")
                unique_sources[source] = title
            sources = [
                f"[{title}]({source})" for source, title in unique_sources.items()
            ]
            answer += "\n\nSources: " + ", ".join(sources)

        # Send the answer with sources
        await cl.Message(content=answer).send()

    # Refresh sidebar after each message
    await sidebar()
