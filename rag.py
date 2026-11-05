
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# 1. Nossa base de conhecimento
documentos = [
    "Honda Civic é um sedã da Honda.",
    "Toyota Corolla é um sedã da Toyota.",
    "Haval H6 é um SUV híbrido da GWM.",
    "Honda CB 500 é uma motocicleta.",
    "Toyota Hilux é uma picape."
]

# 2. Modelo de embeddings
modelo = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Banco vetorial persistente
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="veiculos",
    metadata={"hnsw:space": "cosine"}
)

# 4. Indexação: texto -> vetor
vetores = modelo.encode(
    documentos,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=[f"veiculo_{i}" for i in range(len(documentos))],
    documents=documentos,
    embeddings=vetores
)

# 5. Retriever
def recuperar(pergunta, top_k=3):
    vetor_pergunta = modelo.encode(
        pergunta,
        normalize_embeddings=True
    ).tolist()

    resultados = collection.query(
        query_embeddings=[vetor_pergunta],
        n_results=top_k,
        include=["documents", "distances"]
    )

    for doc, distancia in zip(
        resultados["documents"][0],
        resultados["distances"][0]
    ):
        print(f"Distância: {distancia:.4f} | {doc}")

    return resultados["documents"][0]

# 6. Geração com LLM local
def gerar_resposta(pergunta, contexto):
    prompt = f"""
Responda apenas com base no contexto abaixo.
Se não houver informação suficiente, diga que não sabe.

Contexto:
{chr(10).join(contexto)}

Pergunta: {pergunta}
Resposta:
"""

    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    resposta.raise_for_status()

    return resposta.json()["response"]

pergunta = "Qual veículo é um SUV híbrido?"

contexto = recuperar(pergunta, top_k=5)
resposta = gerar_resposta(pergunta, contexto)

print("\nResposta do RAG:")
print(resposta)
