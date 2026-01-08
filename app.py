import streamlit as st
from dotenv import load_dotenv
import os
from htmlTemplate import css, bot_template, user_template
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import threading
from langchain.schema import Document


load_dotenv()

# llmtemplate = """[INST]
# As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
# - Answer the question based on the provided documents.
# - Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
# - Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
# - If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
# - Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
# - Do not fabricate information or include questions in your responses.
# - do not prompt to select answers. do not ask me questions
# {question}
# [/INST]
# """



def display_timer():
    # Initialize start_time if not already present
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()

    # Calculate the elapsed time
    elapsed_time = time.time() - st.session_state.start_time

    # Display the elapsed time
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
# def display_timer():
#     elapsed_time = time.time() - st.session_state.start_time
#     minutes, seconds = divmod(int(elapsed_time), 60)
#     st.session_state.timer_placeholder.write(f"Session Time: {minutes:02}:{seconds:02}")

def update_timer():
    while True:
        display_timer()
        time.sleep(1)
        st.experimental_rerun()

def prepare_docs(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
        print(pdf.name)
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf.name + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    return content, metadata


def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs


def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db


# def get_conversation_chain(vectordb):

#     # llm = Llama.from_pretrained(
#     # 	repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
#     # 	filename="llama-2-7b-chat.Q4_K_M.gguf",
#     #     temperature=0.75,
#     #     max_tokens=200,
#     #     top_p=1,
#     #     n_ctx=3000)


#     model_path = hf_hub_download(
#         repo_id="TheBloke/Llama-2-7B-Chat-GGUF", 
#         filename="llama-2-7b-chat.Q4_K_M.gguf"
#     )


    
#     llama_llm = LlamaCpp(
#     model_path="llama-2-7b-chat.Q4_K_M.gguf",
#     temperature=0.75,
#     max_tokens=200,
#     top_p=1,
#     n_ctx=3000)


    
#     retriever = vectordb.as_retriever()
#     CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True, output_key='answer')

#     conversation_chain = (ConversationalRetrievalChain.from_llm
#                           (llm=llama_llm,
#                            retriever=retriever,
#                            #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#                            memory=memory,
#                            return_source_documents=True))
#     print("Conversational Chain created for the LLM using the vector store")
#     return conversation_chain


template = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""

#template = """Given the document and the current conversation between a user and an agent, your task is as follows: Answer any user query by using information from the document. The response should be detailed."""
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
def get_conversation_chain(vectordb):


    import os
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())

    # !pip install gguf #https://github.com/ggerganov/llama.cpp/tree/master/gguf-py
    # !git clone https://github.com/ggerganov/llama.cpp
    os.system("git clone https://github.com/ggerganov/llama.cpp")
    model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_basename = "llama-2-7b-chat.Q4_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    llama_llm = LlamaCpp(
    # model_path="llama-2-7b-chat.Q2_K.gguf",
    model_path=model_path,
    temperature=0.75,
    max_tokens=200,
    n_gpu_layers=8,
    top_p=1,
    callback_manager=callback_manager,
    n_ctx=3000,
    verbose=True)

    retriever = vectordb.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llama_llm,
                           retriever=retriever,
                           #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain

def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_threshold = 0.5  
    source_texts = [doc.page_content for doc in source_documents]

    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)

    cosine_scores_list = cosine_scores[0].tolist()
    
        # Check if any cosine score exceeds the similarity threshold
    is_valid = any(score > similarity_threshold for score in cosine_scores_list)

    return is_valid, cosine_scores_list
    
import time

    
# Timer function to display elapsed time
def display_timer():
    elapsed_time = time.time() - st.session_state.start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    st.session_state.timer_placeholder.write(f"Session Time: {minutes:02}:{seconds:02}")

def handle_userinput(user_question):

    
#     col1, col2 = st.columns(2)
#     display_timer()
#     s = [Document(
#     metadata={'title': '/kaggle/input/maternal-data/mat_sample.pdf page 5'}, 
#     page_content="""and with continuity of care throughout the antenatal 
# period. Systematic review of evidence shows that the rou-
# tine involvement of obstetricians in the care of women 
# with an uncomplicated pregnancy at scheduled times 
# does not appear to improve perinatal outcomes compared 
# with involving obstetricians when complications arise.1 
# A system of clear referral paths should be established so 
# that, if additional care is required, women can be seen and 
# treated by the appropriate specialist teams. Obstetricians 
# and specialist teams should be involved if pre-existing 
# medical conditions are present, or when maternal or fetal 
# complications arise.\n
# National guidance recommends that antenatal 
# appointments should take place in a location that women 
# can easily access and that maternity records should be 
# standardised, containing an agreed minimum data-set and 
# held by the woman.1\n
# Schedule for appointments\n
# The NICE recommendation for schedule of antenatal 
# appointments can be found within the guidance and 
# website link: http://pathways.nice.org.uk/pathways/""")
# ]

# # Display the document content
#     with col2:
#         for doc in s:
#             st.write(f"**Title**: {doc.metadata['title']}")
#             # Use st.write() for better handling of line breaks and word wrapping
#             st.write(doc.page_content)

    
    
    response = st.session_state.conversation({'question': user_question})
    
    
    st.session_state.chat_history = response['chat_history']
    
    col1, col2 = st.columns(2)

# Display chat history in the first column (col1)
    with col1:
        for i, message in enumerate(st.session_state.chat_history):
            print(i)
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                print(message.content)
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

        


    if 'source_documents' in response:
        with col2:
            response_answer = response['answer']
            # checck the above line??
            source_docs = response['source_documents']
            is_valid, cosine_scores=validate_answer_against_sources(response_answer, source_docs)
            st.write("Similarity score:",cosine_scores)

            st.write("**Similarity Scores:**")
            for i, score in enumerate(cosine_scores):
                icon = "✅" if score > 0.5 else "❌"  # Example icons based on the similarity score
                st.markdown(f"- **Document {i + 1}:** {icon} **Score:** {score:.2f}", unsafe_allow_html=True)
            
            st.write("**Source Documents:**")

            
            for doc in response['source_documents']:
                st.write(f"**Title**: {doc.metadata['title']}")
                # Use st.write() for better handling of line breaks and word wrapping
                st.write(doc.page_content)

    
        
    


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    # Initialize session state variables for timer if not already present
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    if 'timer_placeholder' not in st.session_state:
        st.session_state.timer_placeholder = st.empty()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Start timer update in a separate thread
    if 'timer_thread' not in st.session_state:
        timer_thread = threading.Thread(target=update_timer, daemon=True)
        timer_thread.start()
        st.session_state.timer_thread = timer_thread

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                content, metadata = prepare_docs(pdf_docs)
                st.write("📄 PDF texts parsing done!")
                

                # get the text chunks
                split_docs = get_text_chunks(content, metadata)
                st.write("📑 Chunking done!")

                # create vector store
                vectorstore = ingest_into_vectordb(split_docs)
                st.write("📊 Vector store creation done!") 

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.success("✅ Processing complete!")


if __name__ == '__main__':
    main()