from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context and your own knowledge:

{context}

---

Answer the question based on the above context:
"""

Q_SYSTEM = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def generateFromEmbeddings(query_text="", model="gpt-3.5-turbo", chat_history=[]) -> tuple[str, list[str]]:
    print(chat_history)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", Q_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    agent = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model=model)

    history_aware_retriever = create_history_aware_retriever(
        agent, db.as_retriever(), contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    question_answer_chain = create_stuff_documents_chain(agent, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query_text, "chat_history": chat_history})
    
    sources = [doc.metadata["source"] for doc in response["context"]]
    response_text = response["answer"]
    return (response_text, sources)

