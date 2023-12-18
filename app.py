from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import chainlit as cl

database_path = 'vectorstore/db'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_chain(llm, prompt, db):
    chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = LlamaCpp(
    model_path="mellogpt.Q4_K_M.gguf",
    temperature=0.75,
    #max_tokens=2048,
    top_p=1, 
    verbose=True,
    streaming=True,
    n_ctx=4096)
    return llm

def questionanswer_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(database_path, embeddings)
    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrieval_chain(llm, qa_prompt, db)
    return qa

#output function
def final_result(query):
    qa_result = questionanswer_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = questionanswer_bot()
    msg = cl.Message(content="Starting the Chat.....")
    await msg.send()
    msg.content = "Hi, Welcome to Mental Health Chatbot,Ask about your Mental Health?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()