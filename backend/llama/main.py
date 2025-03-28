import asyncio
from ollama import AsyncClient

c1 = 200
c2 = 200
c3 = 200
c4 = 200
c5 = 200

essay_theme = "A necessidade de estar sempre certo"

essay = "imagino que, em algum momento de sua vida, você já teve que defender seu ponto de vista com alguém que pensava de uma forma diferente da sua, certo!? o que é a coisa mais normal do mundo, afinal pessoas diferentes pensam de forma diferentes e todas vivem numa perfeita harmonia, ou pelo menos é assim que deveria ser... acontece que, em diversas situações, estamos tão concentrados em emitir nosso ponto de vista que a opinião do outro acaba por ser ignorada pelos nossos ouvidos e cérebro, já que, para muita gente , o importante é sair de uma discussão com seu argumento vitorioso, ao invés de considerarmos vitória a descoberta de uma nova forma de raciocinar , que somente uma discussão construtiva nos oferece. discussões nas quais ninguém sai aprendendo nada são bastantes frequentes em redes sociais. quantas vezes, por exemplo, você já viu na internet um comentário totalmente distorcido em relação ao assunto da matéria , porque o usuário, simplesmente, apenas leu o título do “post” e já foi imediatamente recriminando a “suposta” e inexistente opinião do autor, sem, ao menos, tentar entender do que se trata determinado assunto . e a partir daí a bola de neve só faz crescer, pois outras pessoas fazem a mesma coisa gerando um mal-entendido gigantesco que talvez nunca se chegue ao esclarecimento de fato. é essencial que antes de criticarmos uma ideia que, a princípio, vai de encontro aos nossos ideais, nos coloquemos no lugar do outro e tentemos entender qual o principal conhecimento que ele/ela está tentando nos transmitir. fazer isso é educado, simples e essencial à compreensão do tema abordado. devemos parar com a ideia de que somente o que “eu” penso faz mais sentido do que outro pensa. será que uma pessoa que muda de ideia deve ser considerada alguém sem personalidade? creio que por mais absurda que possa parecer a ideia da pessoa com quem você esteja discutindo, se esse for um debate saudável em que ambos almejam entender o outro, o pior que pode acontecer é você sair com algum conhecimento a mais do que tinha quando entrou. atenciosamente,"

async def chat():
    """
    Stream a chat from Llama using AsyncClient
    :return:
    """
    async for part in await AsyncClient().chat(
            model="llama3:8B",
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
                    "content": f"O tema da redação é: {essay_theme}"
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
        print(part["message"]["content"], end="", flush=True)

asyncio.run(chat())
