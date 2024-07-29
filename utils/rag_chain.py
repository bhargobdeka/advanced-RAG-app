import io
import re
import base64

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from PIL import Image
from langchain.memory import ConversationBufferMemory


def plt_img_base64(img_base64):
    """Display base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{image}"},
            }
            messages.append(image_message)
            
    # Format chat history
    chat_history = data_dict.get("chat_history", [])
    formatted_chat_history = "\n".join([f"{m.type}: {m.content}" for m in chat_history])

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a Research Assistant tasked with answering questions on research articles.\n"
            "You will be given a mixed of text, tables, and image(s) usually of tables, charts or graphs.\n"
            "Use this information to provide accurate information related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
            "Chat History:\n"
            f"{formatted_chat_history}\n\n"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]



def multi_modal_rag_chain(retriever, memory=None):
    """
    Multi-modal RAG chain
    """
    if memory is None:
        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)

    # RAG pipeline
    chain = (
        RunnableParallel(
            {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        })
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    def run_chain(query):
        result = chain.invoke(query)
        memory.save_context({"input": query}, {"output": result})
        return result

    return run_chain

