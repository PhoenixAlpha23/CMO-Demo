from langchain.prompts import PromptTemplate

scheme_query_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant helping citizens understand healthcare schemes.

Context:
{context}

Question:
{question}

Answer in a simple, helpful, and concise manner. If the answer is not found, say "Sorry, this scheme detail is not available."
"""
)
