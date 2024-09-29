"""
- Data Masters -
Creazione e gestione di un chatbot interattivo basato su modelli di linguaggio avanzati.

Utilizzando librerie come LangChain e Gradio, viene configurato un modello di chat (ChatOpenAI) con specifiche
personalizzate; viene definito un template di prompt per guidare le risposte del chatbot; è implementata una memoria
per la conversazione in modo da poter mantenere il contesto ed è sfruttata una catena di esecuzione (LangChain) per
gestire l'input e le risposte; durante l'interazione con l'utente vengono loggati alcuni elementi della conversazione.
Infine, il codice utilizza Gradio per creare un'interfaccia utente che permette una gradevole interazione in tempo
reale con il chatbot.
"""

# Iniziamo importando le librerie necessarie i moduli specifici di LangChain per la
# gestione dei modelli di chat e delle memorie e l'interfaccia utente Gradio

from operator import itemgetter
import os
from langchain_openai import ChatOpenAI  # pip install langchain-openai
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # pip install langchain
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema import SystemMessage, HumanMessage

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool, tool
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI  # pip install langchain-openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import tensorflow as tf
import gradio as gr


def predict_wine_quality(query: str) -> int:
    a = query.split()
    a = [float(x) for x in a]
    model = tf.keras.models.load_model("model/model.h5")
    preds = model.predict([a])
    return preds[0][0]


wine_tool = StructuredTool.from_function(
    func=predict_wine_quality,
    name="wine-tool",
    description="Use this tool to predict wine quality",
)

model = ChatOpenAI(
    openai_api_key="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",
    temperature=0,
    max_tokens=1024,
    request_timeout=30,
    model="gpt-4o",
)
tools = load_tools([], llm=model)
tools.append(wine_tool)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant, but don't know current events and reply in italian. You have access to the following tools:\n\n{tools}\n\nThe way you use the tools is by specifying a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the "action" field are: {tool_names}\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding.""",
        ),
        ("user", "{input}"),
    ]
)

prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

chat_model_with_stop = model.bind(stop=["\nFinal Answer"])

agent = (
    {
        "input": lambda x: x["input"],
    }
    | prompt
    | chat_model_with_stop
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


def stream_response(prompt):
    """
    Gestione delle risposte del chatbot utilizzando la catena di esecuzione
    definita precedentemente per ottenere le risposte generate dal chatbot
    durante la generazione dell'output.
    :param input: messaggio di input
    :param history: conversazioni passate
    """
    if prompt:
        resp = agent.invoke({"input": prompt})
        return resp.to_json()["kwargs"]["return_values"]["output"]
    return "Type a prompt!"


# Creazione dell'interfaccia utente Gradio utilizzando "gr.ChatInterface",
# che utilizza la funzione "stream_response" per gestire le interazioni
# con il chatbot in modalità streaming in tempo reale
iface = gr.Interface(
    fn=stream_response,
    inputs=gr.Textbox(lines=5, placeholder="Type your prompt here..."),
    outputs=gr.Textbox(),
    title="Wine Langchain agent 🦜",
    description="Talk to me!",
)

# Launch the interface
iface.launch()
