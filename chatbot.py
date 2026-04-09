from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Global variables so state persists between calls
rag_chain = None
conversation_history = []  # stores tuples (q, a)


def init_bot():
    global rag_chain
    
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000
    )

    embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    with open('data.txt', 'r', encoding='utf-8') as file:
        text_data = file.read()

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    header_split_data = markdown_splitter.split_text(text_data)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for doc in header_split_data:
        if len(doc.page_content) > 800:
            for chunk in recursive_splitter.split_text(doc.page_content):
                docs.append(Document(page_content=chunk, metadata=doc.metadata))
        else:
            docs.append(Document(page_content=doc.page_content, metadata=doc.metadata))

    vector_store = Chroma(
        persist_directory="Database",
        embedding_function=embedding
    )

    prompt_template = """You are an expert Indian Income Tax consultant. Answer using ONLY the provided context.

{history}

Context:
{context}

Question: {question}

Instructions:
- Provide clear, readable answers (8-12 lines)
- Use line breaks between different points for better readability
- Add relevant emojis (💰📊✅❌)
- Mention key deductions with section numbers
- End with the final result clearly stated
- Use simple language, avoid jargon
- If asked anything outside of itr related stuff just say "I don't know about this sorry" don't say anything else only a single line not more then that

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "history"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )


def ask_bot(question: str) -> str:
    """
    Call this to ask a question to the RAG bot.
    Maintains last 10 interactions.
    """
    global conversation_history, rag_chain

    if rag_chain is None:
        init_bot()

    # Build limited history string
    history_str = ""
    if conversation_history:
        hist = conversation_history[-10:]  # last 10
        numbered = [
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(hist)
        ]
        history_str = "\n\nPrevious Conversation:\n" + "\n\n".join(numbered)

    result = rag_chain.invoke({
        "question": question,
        "history": history_str
    })

    answer = result["result"].strip()

    # update memory
    conversation_history.append((question, answer))
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return answer
