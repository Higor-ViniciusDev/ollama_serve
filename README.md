# ollama_serve

Simples repositório montando um server llm com ollama e utilizando um container python para fazer peguntas e resposta

utilizando das tecnincas RAG utilizo de blocos de texto contendo algumas informações "falsas ou aleatorias".

com um pequeno script python leio esses documentos utilizo da tecnica de embeddings que é disponibilizado pela api do ollama retornando vetores salvo em um banco de vetor.

A ideia e pergunta para a llama3 buscar contexto no banco de vetorh e responde a pergunta utilizando desses contexto
