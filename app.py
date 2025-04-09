import logging

import chainlit as cl
from chainlit.input_widget import NumberInput, Slider, Switch, TextInput
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts import CREATIVE_SYSTEM_PROMPT, FACTFUL_SYSTEM_PROMPT, NORMAL_SYSTEM_PROMPT


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Normal",
            markdown_description="Answer should be based on common knowledge",
        ),
        cl.ChatProfile(
            name="Factful",
            markdown_description="Answer should be based on facts",
        ),
        cl.ChatProfile(
            name="Creative",
            markdown_description="Answer should be creative and imaginative",
        ),
    ]


@cl.on_message
async def main(message: cl.Message):
    system_prompt = cl.user_session.get("system-prompt")
    memory = cl.user_session.get("memory")
    messages: list = memory.load_memory_variables({})["history"]
    messages.insert(0, SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=message.content))

    streaming = cl.user_session.get("streaming")

    settings = {
        "model": cl.user_session.get("model"),
        "temperature": cl.user_session.get("temperature"),
        "streaming": streaming,
        "api_key": cl.user_session.get("api_key"),
    }

    chat_llm = ChatOpenAI(**settings)

    bot_response = cl.Message(content="", author="ChatBot")

    if streaming:
        async for chunk in chat_llm.astream(messages):
            if chunk.content:
                # Stream the output of the step
                await bot_response.stream_token(chunk.content)

        await bot_response.send()
    else:
        response = await chat_llm.ainvoke(messages)
        bot_response.content = response.content
        await bot_response.send()

    memory.save_context({"input": message.content}, {"output": bot_response.content})
    cl.user_session.set("memory", memory)


@cl.on_chat_start
async def start_chat():
    chat_profile = cl.user_session.get("chat_profile")
    logging.info(f"Starting chat with profile: {chat_profile}")
    if chat_profile == "Factful":
        cl.user_session.set("system-prompt", FACTFUL_SYSTEM_PROMPT)
    elif chat_profile == "Creative":
        cl.user_session.set("system-prompt", CREATIVE_SYSTEM_PROMPT)
    else:
        cl.user_session.set("system-prompt", NORMAL_SYSTEM_PROMPT)

    await cl.ChatSettings(
        [
            TextInput(
                id="Model",
                label="Model",
                initial="gpt-3.5-turbo-0125",
            ),
            Switch(id="Streaming", label="Stream Messages", initial=True),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            NumberInput(
                id="Chat Window Size",
                label="Chat Window Size",
                initial=5,
                description="Chat window size to use for completion",
            ),
        ]
    ).send()

    api_key = cl.user_session.get("env")["OPENAI_API_KEY"]
    cl.user_session.set("api_key", api_key)
    cl.user_session.set("model", "gpt-3.5-turbo-0125")
    cl.user_session.set("streaming", True)
    cl.user_session.set("temperature", 1)
    cl.user_session.set("chat_count_to_use", 5)
    cl.user_session.set("memory", ConversationBufferWindowMemory(k=5, return_messages=True))


@cl.on_settings_update
async def update_chat_settings(settings):
    cl.user_session.set("model", settings["Model"])
    cl.user_session.set("streaming", settings["Streaming"])
    cl.user_session.set("temperature", settings["Temperature"])
    cl.user_session.set("chat_count_to_use", settings["Chat Window Size"])
    cl.user_session.get("memory").k = settings["Chat Window Size"]
