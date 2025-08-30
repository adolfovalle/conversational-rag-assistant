import os
import glob
import sys
from dotenv import load_dotenv

# --- Imports de LangChain ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever # <-- AÑADIDO
from langchain_core.messages import HumanMessage, AIMessage

# 1. Cargar las variables de entorno
load_dotenv()

# --- Construcción del Pipeline RAG ---

# 2. Cargar y procesar documentos
print("Cargando documentos pre-procesados...")
loader = DirectoryLoader(
    path="processed_data/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
docs = loader.load()
print(f"Se cargaron {len(docs)} documentos limpios.")

print("Dividiendo documentos en trozos...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
splits = text_splitter.split_documents(docs)
print(f"Se crearon {len(splits)} trozos de texto.")

print("Creando o cargando base de datos vectorial...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
print("Base de datos vectorial lista.")

# 3. Definir el LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# --- CADENAS CONVERSACIONALES, DE RESUMEN Y DE RESPUESTA ---

# 4. Cadena de Re-escritura de Preguntas para la Memoria
contextualize_q_system_prompt = """Dada una conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta autónoma.
**La pregunta autónoma debe ser lo suficientemente detallada como para ser entendida sin el historial del chat.**
Si la pregunta de seguimiento no está directamente relacionada con el historial del chat, **ignora el historial** y devuelve la pregunta de seguimiento tal cual.

Ejemplo 1:
Historial: Humano: "¿Cómo implemento Drupal?", IA: "Debes usar el módulo JSON:API."
Pregunta de Seguimiento: "dame más detalles sobre ese módulo"
Pregunta Autónoma: "dame más detalles sobre el módulo JSON:API para Drupal"

Ejemplo 2:
Historial: Humano: "¿Cómo implemento Drupal?", IA: "Debes usar el módulo JSON:API."
Pregunta de Seguimiento: "¿Qué es Azure AI Search?"
Pregunta Autónoma: "¿Qué es Azure AI Search?"
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 5. Cadena de Resumen de Contexto
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Tu tarea es tomar el siguiente contexto extraído de documentos y la pregunta del usuario, y sintetizar un resumen conciso que contenga la información necesaria para responder la pregunta.
    Extrae solo los hechos y datos más relevantes.

    Contexto:
    {context}
    """),
    ("human", "Pregunta del usuario: {input}")
])
context_summarizer_chain = summarizer_prompt | llm | StrOutputParser()

# 6. Cadena de Generación de Respuesta Final
qa_system_prompt = """
Eres un asistente de IA útil. Responde la pregunta del usuario basándote en el resumen del contexto que se te proporciona.
Si el resumen no contiene la respuesta, di que no tienes suficiente información en los documentos.
Sé claro y responde en español.

Resumen del Contexto:
{context_summary}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}"), 
])
question_answer_chain = qa_prompt | llm | StrOutputParser()

# 7. Cadena Principal que une todo el flujo
main_rag_chain = (
    RunnablePassthrough.assign(
        # 1. Recupera los documentos usando el historial para re-escribir la pregunta
        context=history_aware_retriever
    ).assign(
        # 2. Usando los documentos recuperados y la pregunta original, crea un resumen
        context_summary=context_summarizer_chain
    )
    # 3. Finalmente, usa el resumen y la pregunta original para generar la respuesta.
    | question_answer_chain
)

print("\n--- Asistente de IA con Memoria y Resumen de Contexto listo. Escribe 'salir' para terminar. ---")

# 8. Bucle Interactivo con Memoria
chat_history = []
while True:
    user_input = input("\nTu pregunta: ")
    if user_input.lower() == 'salir':
        break
    
    # Invocamos la cadena principal, que espera 'input' y 'chat_history'.
    response = main_rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
    print("\nRespuesta del Asistente:")
    print(response)

    # Actualizamos el historial del chat
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

print("\nSaliendo del programa...")
sys.exit(0)