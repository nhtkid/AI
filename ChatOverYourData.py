# You might need to reinstall Pillow library if you receive PIL error
# pip uninstall Pillow
# pip install --upgrade Pillow
# import PIL
# print(PIL.__version__)

# Requirement
pip install openai -q
pip install langchain -q
pip install chromadb -q
pip install tiktoken -q
pip install pypdf -q
pip install unstructured[local-inference] -q
pip install gradio -q

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

import os
os.environ["OPENAI_API_KEY"] = "Your OpenAI Key Here"

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name="gpt-4")

# Load your data into Reports folder
from langchain.document_loaders import DirectoryLoader

pdf_loader = DirectoryLoader('./Reports/', glob="**/*.pdf")
txt_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")

loaders = [pdf_loader, txt_loader, word_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

print(f"Total number of documents: {len(documents)}")

# Chunk the data, turn into Embeddings and save to VectorStore
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Calling the Langchain's QA chain with Chat History
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)

# Gradio Front End Chat UI
import gradio as gr
# Define chat interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []

    def user(query, chat_history):
        print("User query:", query)
        print("Chat history:", chat_history)

        # Convert chat history to list of tuples
        chat_history_tuples = []
        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))

        # Get result from QA chain
        result = qa({"question": query, "chat_history": chat_history_tuples})

        # Append user message and response to chat history
        chat_history.append((query, result["answer"]))
        print("Updated chat history:", chat_history)

        return gr.update(value=""), chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
