from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import Document

# Setup LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=""
)

# Setup Embedding + FAISS
embedding_model = OpenAIEmbeddings(openai_api_key="")
docs = [Document(page_content="This is a starter memory document.")]
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# Vector Memory Setup
retriever_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory_key="history"
)

# Chain
conversation = ConversationChain(
    llm=llm,
    memory=retriever_memory,
    verbose=True
)

# Demo Flow
def run_semantic_agent():
    print(" Semantic Memory Agent")

    input_1 = "I need a function to convert Celsius to Fahrenheit."
    print(f"\n User: {input_1}")
    print(" Assistant:\n" + conversation.predict(input=input_1))

    input_2 = "Now add Kelvin support to the function we discussed."
    print(f"\n User: {input_2}")
    print(" Assistant:\n" + conversation.predict(input=input_2))

    print("\n Retrieved Memory:\n", retriever_memory.load_memory_variables({"input": "temperature"})["history"])

# Run with token tracking
if __name__ == "__main__":
    with get_openai_callback() as cb:
        run_semantic_agent()
        print("\n📊 Token Usage Summary:")
        print(cb)
