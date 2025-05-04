# CLaRiCe
Este repositório hospeda os arquivos e código fonte da aplicação desenvolvida como parte
do artigo estendido **Feed-CLaRiCe: Uma abordagem neural para correção e feedbacks automáticos de redações**, submetido para publicação na Revista Brasileira de Informática na Educação (RBIE) 2025.

## Feed-CLaRiCe

A aplicação Feed-CLaRiCe consiste numa interface conversacional web, desenvolvida com o uso do **Streamlit**,
que permite ao usuário submeter a sua redação/texto dissertativo-argumentativo no formado do Exame Nacional do Ensino Médio (ENEM)
para a obtenção de sua nota, gerada pelo modelo de regressão desenvolvido em **CLaRiCe: Uma abordagem neural para a correção automática de redações**, seguido pelo posterior
feedback gerado e fornecido em linguagem natural por um **Large Language Model**.

Ressaltado ainda que o objetivo não foi o de construir um chatbot, mas avaliar as capacidades integradas dos modelos para o fornecimento automático
de feedbacks no tipo de texto acima citado, portanto, algumas capacidades conversacionais podem não funcionar como o esperado.

## Usando

Este repositório possui o código fonte e também possui uma estrutura de containers Docker configurada pronta para ser usada,
bastando-se apenas utilizar o seguinte comando docker para executar a aplicação por meio do docker-compose:

```shell
docker compose up -d
```

Um outro ponto importante a se prestar atenção é que o container do ollama não realiza automaticamente o download do modelo de LLM,
ficando assim a cargo do utilizador a definição de qual LLM deseja-se utilizar.

Para realizar o download do LLM a ser utilizado no container do ollama, pode-se utilizar o seguinte comando Docker:

```shell
docker exec -it ollama ollama pull <nome_do_modelo>
```

Basta apenas procurar o nome do modelo nos repositórios do ollama e substituir no comando acima, após isso aguardar o download
do modelo e utilizar livremente.

O modelo utilizado por padrão no desenvolvimento do artigo bem como da aplicação foi o Gemma3-4B.
