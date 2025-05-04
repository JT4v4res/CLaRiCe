from langchain_ollama.llms import OllamaLLM
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from requests import Session
import numpy as np
import json


inference_endpoint = "http://clarice-backend:8000/predict/tf/"

@dataclass
class Message:
    actor: str
    payload: str


@st.cache_resource
def get_llm() -> OllamaLLM:
    return OllamaLLM(model="gemma3:4B", host="http://ollama:11434")

def get_classifier_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Você é um classificador de textos que identifica se um determinado texto é uma redação ou não."
        ),
        SystemMessagePromptTemplate.from_template(
            "Tarefa: Você deve verificar se o texto fornecido é uma redação ou não e retornar um JSON contendo apenas o field "
            "'text_type' possuindo um dos dois valores: 'essay' ou 'not essay'. Retorne SOMENTE o JSON requisitado na sua resposta."
        ),
        HumanMessagePromptTemplate.from_template("{msg}")
    ])

    chain = LLMChain(
        llm=get_llm(),
        prompt=prompt,
        verbose=True,
    )

    return chain


def get_conversational_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Persona: Você é Clarice, uma auxiliar de correção de redações amigável e atenciosa."
        ),
        SystemMessagePromptTemplate.from_template(
            "Tarefa: Você deve conversar com os usuários acerca de redações e dicas de escrita no geral, "
            "sempre reconhecendo seus esforços e tentando auxiliá‑los com suas dúvidas."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{msg}"),
    ])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="msg",
        return_messages=True,
    )

    chain = LLMChain(
        llm=get_llm(),
        prompt=prompt,
        memory=memory,
        verbose=True,
    )

    return chain


def get_llm_chain_with_notes_instruction():
    # 1) Consolida todas as instruções num único system message
    full_system = (
        "Persona: Você é Clarice, uma auxiliar de correção de redações amigável e atenciosa.\n"
        "Tarefa: Você deve fornecer um feedback conciso acerca da redação do usuário, "
        "proporcionando uma boa revisão acerca de quais pontos o usuário deve melhorar "
        "para obter melhores notas em sua redação.\n"
        "Critérios de avaliação: As notas da redação foram avaliadas em cinco competências, "
        "cada uma valendo 200 pontos. As competências são: "
        "1. Demonstrar domínio da norma padrão da língua escrita; "
        "2. Compreender a proposta da redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema; "
        "3. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista; "
        "4. Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação; "
        "5. Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.\n"
        "Ao final do feedback, informe ao usuário quais foram as notas dele em cada uma das competências "
        "bem como sua nota total."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(full_system),
        HumanMessagePromptTemplate.from_template(
            "Minhas notas em cada competência foram: "
            "Nota da competência 1: {c1}/200, "
            "Nota da competência 2: {c2}/200, "
            "Nota da competência 3: {c3}/200, "
            "Nota da competência 4: {c4}/200, "
            "Nota da competência 5: {c5}/200, "
            "Nota total da redação: {total}\n\n"
            "E esta é minha redação:\n\n{essay}"
        ),
    ])

    conversation = LLMChain(
        llm=get_llm(),
        prompt=prompt,
        verbose=True,
    )
    return conversation

USER = "user"
ASSISTANT = "assistant"
MESSAGES = "messages"

def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Olá, eu sou a Clarice, vamos começar a nossa conversa?")]

    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain_with_notes_instruction()

    if "llm_conversation" not in st.session_state:
        st.session_state["llm_conversation"] = get_conversational_chain()

def get_llm_chain_from_session(type: str = "llm_chain") -> LLMChain:
    return st.session_state[type]


initialize_session_state()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Vamos conversar!")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Pensando..."):
        classifier = get_classifier_chain()
        output = classifier({"msg": prompt})

        output = json.loads(output["text"].replace("```", '').replace("json", '').strip())

        if output.get("text_type") == "essay":
            inference_req = {"text": prompt}

            if prompt:
                with Session() as session:
                    grades = session.post(inference_endpoint, json=inference_req)

                try:
                    c1, c2, c3, c4, c5 = grades.json().get('predict_grades')
                    c1, c2, c3, c4, c5 = int(c1), int(c2), int(c3), int(c4), int(c5)
                except KeyError as err:
                    print(err)

            llm_chain = get_llm_chain_from_session()
            response: str = llm_chain({"essay": prompt, "c1": c1, "c2": c2, "c3": c3,
                                       "c4": c4, "c5": c5, "total": np.sum([c1, c2, c3, c4, c5])})["text"]
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)

        else:
            llm_chain = get_llm_chain_from_session("llm_conversation")
            response: str = llm_chain({"msg": prompt})["text"]
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
