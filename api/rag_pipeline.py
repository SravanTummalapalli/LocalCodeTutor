import os

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

VECTOR_STORE_PATH = "vector_store"

def load_rag_pipeline():
    """Loads FAISS DB + builds the RAG chain."""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 1})

    prompt = ChatPromptTemplate.from_template("""
You are a senior Python developer helping a candidate prepare for technical interviews.

RULES:
- Base your answer ONLY on the CONTEXT provided below
- If information is missing, clearly state: "This topic isn't covered in the materials I have."
- Be precise and accurate - interviews require correct information

CONTEXT:
{context}

QUESTION:
{question}

ANSWER STRUCTURE:

📚 DEFINITION
What is this concept? Explain like you're talking to someone learning it for the first time.

🎯 WHY IT EXISTS
- What problem does it solve?
- Why would a developer choose to use this?
- What's the real-world value?

⏰ WHEN TO USE
- Perfect scenarios for using this
- Common use cases in production code
- Situations where you should avoid it

🔧 HOW IT WORKS
Break down the mechanics step by step:
→ Step 1: [First thing that happens]
→ Step 2: [Next thing]
→ Step 3: [And so on...]
Think of it like: [simple analogy if possible]

💻 CODE DEMONSTRATION

[Clean, formatted code here]
# Comments explaining the important parts
# Show what each line accomplishes

Expected Output:
[Show what running this code produces]

⚡ INTERVIEW TIPS
- Key points to mention in an interview
- Common mistakes candidates make
- Follow-up questions you might get
- Complexity analysis (if applicable)

🔗 RELATED CONCEPTS
[Briefly mention connected topics worth knowing]

Keep your answer focused, clear, and interview-ready!
""")

    llm = ChatOllama(model="phi3", temperature=0.2)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


rag_pipeline = load_rag_pipeline()
