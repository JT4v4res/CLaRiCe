import streamlit as st
import numpy as np
import asyncio
from ollama import Client
from requests import Session
import json
from dataclasses import dataclass
import os

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "assistant"
MESSAGES = "messages"
ALREADY_PRESENTED = "already_presented"


def get_ollama_client():
    return Client(host=os.environ.get("OLLAMA_HOST", "http://ollama:11434"))

inference_endpoint = "http://clarice-backend:8000/predict/tf/"

def chat(essay:str, c1:int, c2:int, c3:int, c4:int, c5:int, ollama_client):
    """
    Stream a chat from Llama using AsyncClient
    :return:
    """
    global current_message

    current_message = ""

    essay = "imagino que, em algum momento de sua vida, você já teve que defender seu ponto de vista com alguém que pensava de uma forma diferente da sua, certo!? o que é a coisa mais normal do mundo, afinal pessoas diferentes pensam de forma diferentes e todas vivem numa perfeita harmonia, ou pelo menos é assim que deveria ser... acontece que, em diversas situações, estamos tão concentrados em emitir nosso ponto de vista que a opinião do outro acaba por ser ignorada pelos nossos ouvidos e cérebro, já que, para muita gente , o importante é sair de uma discussão com seu argumento vitorioso, ao invés de considerarmos vitória a descoberta de uma nova forma de raciocinar , que somente uma discussão construtiva nos oferece. discussões nas quais ninguém sai aprendendo nada são bastantes frequentes em redes sociais. quantas vezes, por exemplo, você já viu na internet um comentário totalmente distorcido em relação ao assunto da matéria , porque o usuário, simplesmente, apenas leu o título do “post” e já foi imediatamente recriminando a “suposta” e inexistente opinião do autor, sem, ao menos, tentar entender do que se trata determinado assunto . e a partir daí a bola de neve só faz crescer, pois outras pessoas fazem a mesma coisa gerando um mal-entendido gigantesco que talvez nunca se chegue ao esclarecimento de fato. é essencial que antes de criticarmos uma ideia que, a princípio, vai de encontro aos nossos ideais, nos coloquemos no lugar do outro e tentemos entender qual o principal conhecimento que ele/ela está tentando nos transmitir. fazer isso é educado, simples e essencial à compreensão do tema abordado. devemos parar com a ideia de que somente o que “eu” penso faz mais sentido do que outro pensa. será que uma pessoa que muda de ideia deve ser considerada alguém sem personalidade? creio que por mais absurda que possa parecer a ideia da pessoa com quem você esteja discutindo, se esse for um debate saudável em que ambos almejam entender o outro, o pior que pode acontecer é você sair com algum conhecimento a mais do que tinha quando entrou. atenciosamente,"

    for part in ollama_client.chat(
        model="gemma3:4B",
            messages=[
                {
                    "role": "system",
                    "content": "Persona: Você é Clarice, uma auxiliar de correção de redações amigável e atenciosa."
                },
                {
                    "role": "system",
                    "content": "Tarefa: Você deve fornecer um feedback conciso acerca da redação do usuário, proporcionando uma boa " +
                    "revisão acerca de quais pontos o usuário deve melhorar para obter melhores notas em sua redação."
                },
                {
                    "role": "system",
                    "content": "Critérios de avaliação: As notas da redação foram avaliadas em cinco competências, cada uma valendo 200 pontos. " +
                               "As competências são: 1. Demonstrar domínio da norma padrão da língua escrita; " +
                               "2. Compreender a proposta da redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema; " +
                               "3. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista; " +
                               "4. Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação; " +
                               "5. Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos."
                },
                {
                    "role": "system",
                    "content": "Feedback: Você deve fornecer um feedback conciso acerca da redação do usuário, proporcionando uma boa " +
                               "revisão acerca de quais pontos o usuário deve melhorar para obter melhores notas em sua redação. " +
                               "Você não deve informar qual nota você daria para cada competência, apenas dar ao usuário feedbacks que " +
                               "o ajudem a melhorar sua redação."
                },
                {
                    "role": "system",
                    "content": "Instrução adicional: Ao final do feedback informe ao usuário quais foram as notas dele em cada uma das competências bem como sua nota total. " +
                               "Lembre-se de informar qual a nota dele em cada uma das competências."
                },
                {
                    "role": "user",
                    "content": f"Minhas notas em cada competência foram: " +
                               f"Nota da competência 1: {c1}/200, " +
                               f"Nota da competência 2: {c2}/200, " +
                               f"Nota da competência 3: {c3}/200, " +
                               f"Nota da competência 4: {c4}/200, " +
                               f"Nota da competência 5: {c5}/200, " +
                               f"Nota total da redação: {c1 + c2 + c3 + c4 + c5} " +
                               "e está é minha redação:\n\n" + essay
                }
            ],
            stream=True
    ):
        current_message += part.get("message").get("content")
        yield part.get("message").get("content")


def verify_message_type(msg: str, ollama_client) -> any:
    response = ollama_client.chat(
        model="gemma3:4B",
        messages=[
            {
                "role":"system",
                "content": "Você é um classificador de textos que identifica se um determinado texto é uma redação ou não."
            },
            {
                "role":"system",
                "content": "Tarefa: Você deve verificar se o texto fornecido é uma redação ou não e retornar um JSON contendo apenas o field 'text_type' possuindo um dos dois valores: 'essay' ou 'not essay'. Retorne SOMENTE o JSON requisitado na sua resposta,"
            },
            {
                "role": "user",
                "content": msg
            }
        ]
    )

    json_msg = json.loads(response.get("message").get("content").replace('```', '').replace('json', '').strip())

    if json_msg.get("text_type") == "essay":
        # st.write("É essay")
        return True

    # st.write("Não é essay")
    return False


def no_essay_chat(msg: str, ollama_client) -> None:
    global current_message

    current_message = ""

    for model_response in ollama_client.chat(
        model="gemma3:4B",
        messages=[
                    {
                        "role":"system",
                        "content": "Persona: Você é Clarice, uma auxiliar de correção de redações amigável e atenciosa."
                    },
                    {
                        "role": "system",
                        "content": "Tarefa: Você deve conversar com os usuários acerca de redações e dicas de escrita no geral, sempre reconhecendo seus esforços e tentando auxilia-los com suas dúvidas."
                    },
                    {
                        "role": "user",
                        "content": msg
                    },
                ],
        stream=True
    ):
        current_message += model_response.get("message").get("content")
        yield model_response.get("message").get("content")

def presentation(ollama_client):
    global current_message

    current_message = ""

    for model_response in ollama_client.chat(
        model="gemma3:4B",
        messages=[
                    {
                        "role": "system",
                        "content": "Persona: Você é Clarice, uma auxiliar de correção de redações amigável e atenciosa."
                    },
                    {
                        "role": "user",
                        "content": "Por favor, forneça uma breve apresentação sobre quem você é. Não mencione nada que faça alusão a algum tipo de prestação de serviço ou atendimento ao cliente."
                    }
                ],
        stream=True
    ):
        current_message += model_response.get("message").get("content")
        yield model_response.get("message").get("content")


def initialize_session_state(ollama_client):
    global current_message

    current_message = ""

    if ALREADY_PRESENTED not in st.session_state:
        st.session_state[ALREADY_PRESENTED] = None

    if MESSAGES not in st.session_state:
        with st.chat_message(ASSISTANT):
            st.write_stream(presentation(ollama_client))

        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload=current_message)]


def main():
    global current_message

    ollama_client = get_ollama_client()

    current_message = ""

    initialize_session_state(ollama_client)

    essay: str = st.chat_input(placeholder="Fale com Clarice")

    if essay:
        # st.chat_message(USER).write(essay)
        st.session_state[MESSAGES].append(Message(actor=USER, payload=essay))

    if len(st.session_state[MESSAGES]) > 1:
        msg: Message
        for msg in st.session_state[MESSAGES]:
            st.chat_message(msg.actor).write(msg.payload)

    c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0

    if essay and verify_message_type(essay, ollama_client):

        inference_req = {"text": essay}

        if essay:
            with Session() as session:
                grades = session.post(inference_endpoint, json=inference_req)

            try:
                c1, c2, c3, c4, c5 = grades.json().get('predict_grades')
            except KeyError as err:
                print(err)

        if c1 != 0 and c2 != 0 and c3 != 0 and c4 != 0 and c5 != 0:
            with st.chat_message(ASSISTANT):

                st.write_stream(chat(essay, int(c1), int(c2), int(c3), int(c4), int(c5), ollama_client))

                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=current_message))

    else:
        if essay:
            with st.chat_message(ASSISTANT):
                st.write_stream(no_essay_chat(essay, ollama_client))
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=current_message))


if __name__ == "__main__":
    main()
