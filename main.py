import streamlit as st
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from typing import Any
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.image_processing import generate_img_summaries
from utils.retriever import create_multi_vector_retriever
from utils.rag_chain import multi_modal_rag_chain, plt_img_base64
from utils.rag_evaluation import LLM_Metric
from io import BytesIO
import base64
from PIL import Image
import io

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

# Streamlit app setup
st.set_page_config(page_title='Multi-Modal RAG Application', page_icon='random', layout='wide', initial_sidebar_state='auto')

def process_document(uploaded_file):
    # Process PDF
    with st.spinner('Processing PDF...'):
        st.sidebar.info('Extracting elements from PDF...')
        pdf_bytes = uploaded_file.read()
        elements = partition_pdf(
            file=BytesIO(pdf_bytes),
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir="docs/saved_images",
        )
        st.sidebar.success('PDF elements extracted successfully!')

    # Create chunks by title
    with st.spinner('Chunking content...'):
        st.sidebar.info('Creating chunks by title...')
        chunks = chunk_by_title(elements)
        st.sidebar.success('Chunking completed successfully!')

    # Categorize Elements
    class Element(BaseModel):
        type: str
        text: Any

    categorized_elements = []
    for element in chunks:
        if "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
        elif "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))

    text_elements = [e for e in categorized_elements if e.type == "text"]
    table_elements = [e for e in categorized_elements if e.type == "table"]

    # Prompt
    prompt_text = """You are an expert Research Assistant tasked with summarizing tables and texts from research articles. \
    Give a concise summary of the text. text chunk: {element} """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    texts = [i.text for i in text_elements]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    # Image summaries
    fpath = "docs/saved_images"
    img_base64_list, image_summaries = generate_img_summaries(fpath)

    # Vectorstore
    vectorstore = Chroma(
        collection_name="mm_tagiv_paper", embedding_function=OpenAIEmbeddings()
    )

    # Create retriever
    st.session_state.retriever = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    # Create RAG chain
    st.session_state.chain = multi_modal_rag_chain(retriever=st.session_state.retriever)
    st.session_state.processed = True

with st.sidebar:
    # File upload
    st.subheader('Add your PDF')
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if st.button('Submit'):
        if uploaded_file is not None:
            process_document(uploaded_file)
            st.success('Document processed successfully!')
        else:
            st.error('Please upload a PDF file first.')

# Main page for query response and evaluation
st.subheader("RAG Assistant")
query = st.text_input("Enter your query:")

if query and st.session_state.processed:
    # Execution
    retrieval_context = st.session_state.retriever.invoke(query, limit=1)
    actual_output = st.session_state.chain(query)

    # Evaluation
    llm_metric = LLM_Metric(query, retrieval_context, actual_output)
    faith_score, faith_reason = llm_metric.get_faithfulness_metric()
    relevancy_score, relevancy_reason = llm_metric.get_contextual_relevancy_metric()
    answer_relevancy_score, answer_relevancy_reason = llm_metric.get_answer_relevancy_metric()
    hallucination_score, hallucination_reason = llm_metric.get_hallucination_metric()

    # Display results
    st.subheader("Query Response")
    st.write(actual_output)

    st.subheader("Evaluation Metrics")
    st.write(f"Faithfulness Score: {faith_score}, Reason: {faith_reason}")
    st.write(f"Contextual Relevancy Score: {relevancy_score}, Reason: {relevancy_reason}")
    st.write(f"Answer Relevancy Score: {answer_relevancy_score}, Reason: {answer_relevancy_reason}")
    st.write(f"Hallucination Score: {hallucination_score}, Reason: {hallucination_reason}")

    # st.subheader("Retrieved Documents")
    # for item in retrieval_context:
    #     if isinstance(item, dict):
    #         content = str(item)
    #     else:
    #         content = item

    #     # Check if the content is a base64 encoded image
    #     try:
    #         # Try to decode the base64 string
    #         # img_data = base64.b64decode(content)
    #         # Try to open it as an image
    #         # img = Image.open(io.BytesIO(img_data))
    #         img = plt_img_base64(content)
    #         # If successful, display the image
    #         st.image(img, caption="Retrieved Image")
    #     except:
    #         # If it's not a valid base64 image, display a message
    #         st.write("No image can be displayed. Text content:")
    #         st.write(content)

    #     st.write("---")  # Add a separator between items
elif query and not st.session_state.processed:
    st.warning("Please upload and process a document first.")