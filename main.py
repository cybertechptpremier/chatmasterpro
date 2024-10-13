from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict
import tiktoken
import base64
from io import BytesIO
from PIL import Image
from genfromdoc import generateFromEmbeddings
from genfromgoogle import getGoogleAgent
from streamlit_paste_button import paste_image_button
from langchain_core.messages import HumanMessage
import pickle
from pathlib import Path
import logging
import streamlit_authenticator as stauth
import sys

# __import__("pysqlite3")
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

logging.basicConfig(level=logging.INFO)

load_dotenv()


st.set_page_config(page_title="ChatMaster Pro", page_icon="🤖", layout="wide")
client = OpenAI()

names = ["ptpremier"]
usernames = ["ptpremier"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords, "ptpremier", "ptpremier", cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
if authentication_status is None:
    st.warning("Please enter your username and password")


def handleFinetuned(client, prompt, spinner_placeholder):
    logging.info("ENTERED INTO FINE TUNED")
    if len(st.session_state.uploaded_images) != 0:
        spinner_placeholder.text("Uploading Images...")
        base_prompt = [
            {
                "role": "system",
                "content": "You are an Image Reader. I will provide you with images containing text. Your task is to read the text in the images and return the exact texts, formatted as it appears, combine them in one response intelligently if there are multiple. without any modifications or interpretations.",
            },
        ]
        image_count = 0
        images_to_remove = []
        for msg in st.session_state.messages:
            if isinstance(msg["content"], list):
                base_prompt.extend(
                    [
                        {
                            "role": msg["role"],
                            "content": f"Image No. {image_count}",
                        },
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        },
                    ]
                )
                image_count += 1
                images_to_remove.append(msg)
        for image in images_to_remove:
            st.session_state.messages.remove(image)
        res = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=base_prompt,
        )
        content = res.choices[0].message.content
        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "content": "Here are some images, return the text",
                },
                {"role": "assistant", "content": "Text in Image:\n" + content},
            ]
        )
    spinner_placeholder.text("Generating using fine tuned model...")
    PROMPT = "As a logical reasoning assistant, your role is to help users analyze passages and answer complex questions about arguments and deductions. Carefully read the passage to understand its claims, then guide users to the most sound conclusion. Identify question types—whether they aim to weaken, strengthen, infer, or identify assumptions. Analyze each answer choice by comparing it to the passage, evaluating its effect on the argument, Eliminate them first as you see fit, look back on the passage for sanity checks of your statments. Finally, explain the reasoning clearly to support users in understanding the process and reaching defensible conclusions. Only answer this question, be precise, and use the passage as context. NEVER START GENERATING NEW QUESTION."
    messages = [
        {
            "role": "assistant",
            "content": PROMPT,
        },
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.extend(
        [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if not isinstance(msg["content"], list)
        ]
    )

    logging.info(messages)
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
    )
    response = response.choices[0].message.content
    logging.info(response)
    st.write(response)
    return response


def handleGoogle(prompt, spinner_placeholder):
    spinner_placeholder.text("Generating response using Google Search...")
    agent = getGoogleAgent(model=st.session_state["openai_model"])
    chat_history = []
    messages = st.session_state.messages
    # Loop in steps of 2 to pair (quest, resp)
    for i in range(0, len(messages) - 1, 2):
        if not isinstance(messages[i]["content"], list):
            user_message = messages[i]["content"]
            assistant_response = messages[i + 1]["content"]
            chat_history.append(
                f"User: {user_message}\nAssistant: {assistant_response}"
            )

            # Join the tuples into a single string
    formatted_chat_history = "\n".join(chat_history)

    response = agent.invoke({"input": prompt, "chat_history": formatted_chat_history})[
        "output"
    ]
    spinner_placeholder.text("Response generated using Google Search.")
    st.write(response)
    return response


def handleNormal(client,prompt, spinner_placeholder):
    spinner_placeholder.text("Generating response using normal method...")
    PROMPT = "As a logical reasoning assistant, your role is to assist users in analyzing passages, evaluating arguments, and answering complex questions based on given images or text. You will carefully read the image or text, analyze the question in relation to the context, and critically evaluate each answer choice. By methodically assessing the validity and relevance of each option, you will guide the user toward the most accurate conclusion or deduction for the problem at hand."
    messages = [
        {
            "role": "assistant",
            "content": PROMPT,
        },
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.extend(
        [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
    )
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
    )
    response = response.choices[0].message.content
    logging.info(response)
    st.write(response)
    return response

if authentication_status:
    st.info(f"Welcome *{name}*")
    # App branding and title
    st.title("Welcome to ChatMaster Pro")
    with st.expander("Instructions"):
        st.markdown(
            """
    ### ChatMaster Pro: Your AI Chat Companion

    **ChatMaster Pro** is a powerful AI-powered chat application that allows you to engage in natural and informative conversations with advanced language models. 

    **Here's how to use it:**

    1. **Choose a Model:** Select the desired language model from the sidebar.
    2. **Start Chatting:** Type your questions or requests in the chat box.
    3. **Upload Images:**  You can upload images to enhance your conversations.
    4. **Clear History:** Use the "Clear Chat History" button to start a fresh conversation.

    **Enjoy a seamless and engaging chat experience with ChatMaster Pro!**
    """
        )

    # Sidebar for model selection and clear chat history button
    with st.sidebar:
        st.header("Settings")
        col1, col2 = st.columns(2)
        with col1:
            authenticator.logout("↩", "main")
        with col2:
            if st.button("Rerun"):
                st.rerun()
        selected_model = st.selectbox(
            "Choose a Model",
            options=[
                "gpt-4o-mini",
                # "ft:gpt-4o-mini-2024-07-18:primetime-premier:rc-lr:AHCNvENj",s
                "gpt-4o-2024-08-06",
            ],
            index=0,
        )
        st.session_state["openai_model"] = selected_model

        # Button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.uploaded_images = {}  # Clear uploaded images as well
            st.success("Chat history cleared!")

        # Image upload widget
        uploaded_files = st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )

        # Move the Generate using documents to the most bottom
        st.markdown("---")

        # Checkbox for using documents
        # use_documents = st.checkbox("Generate using documents")  # Moved to sidebar
        use_google = st.checkbox("Google Search")  # Moved to sidebar
        col1, col2 = st.columns(2)
        with col1:
            paste_result = paste_image_button(
                label="Paste Image from Clipboard",
                background_color="#FF0000",
                hover_background_color="#380909",
                errors="raise",
            )
        with col2:
            if st.button("Clear Images"):
                # Check if buffer image is the same as the current image
                if st.session_state.buffer_image != st.session_state.current_image:
                    st.session_state.buffer_image = st.session_state.current_image
                    st.session_state.current_image = None
                    st.session_state.uploaded_images = (
                        {}
                    )  # Clear uploaded images as well
                    st.success("Images cleared!")
            if st.button("Reset Image History"):
                st.session_state.buffer_image = None

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = {}
    if "buffer_image" not in st.session_state:
        st.session_state.buffer_image = None

    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    # Function to convert image to base64
    def image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    # Function to process messages and ensure token limit, ignoring image messages
    def process_messages(
        messages: List[Dict[str, str]], model: str
    ) -> List[Dict[str, str]]:
        encoding = tiktoken.encoding_for_model(model)
        total_tokens = 0

        # Calculate total tokens and remove excess messages if needed
        for message in messages:
            # Ignore messages with image content
            if isinstance(message["content"], list):
                # Check if the message contains image content; if so, skip it
                if any(item["type"] == "image_url" for item in message["content"]):
                    continue

            # Only process text content for token count
            content_to_encode = (
                message["content"]
                if isinstance(message["content"], str)
                else str(message["content"])
            )
            encoded_length = len(encoding.encode(content_to_encode))
            total_tokens += encoded_length

        # Remove oldest non-system messages if token count exceeds the limit
        token_limit = 8000 if model == "gpt-4" else 4000
        while total_tokens > token_limit:
            index_to_remove = next(
                (
                    i
                    for i, msg in enumerate(messages)
                    if msg["role"] != "system" and not isinstance(msg["content"], list)
                ),
                None,
            )
            if index_to_remove is not None:
                removed_content = messages.pop(index_to_remove)["content"]
                total_tokens -= len(
                    encoding.encode(
                        removed_content
                        if isinstance(removed_content, str)
                        else str(removed_content)
                    )
                )
            else:
                break

        return messages

    # Handle image uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            img_base64 = image_to_base64(img)
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_base64},
                    },
                ],
            }
            # Add image to message list but don't submit yet
            st.session_state.uploaded_images[uploaded_file.name] = image_message

    if paste_result.image_data is not None:
        # Store the new image in session state
        st.session_state.current_image = paste_result.image_data

    if st.session_state.uploaded_images != {}:
        with st.sidebar:
            with st.spinner("Uploading images..."):
                for (
                    image_name,
                    image_message,
                ) in st.session_state.uploaded_images.items():
                    st.image(image_message["content"][0]["image_url"]["url"])
    if (
        st.session_state.current_image
        and st.session_state.current_image != st.session_state.buffer_image
    ):
        try:
            # Since paste_result.image_data is already a PIL image, use it directly
            img = st.session_state.current_image
            # Convert to base64 string for display
            img_base64 = image_to_base64(img)
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_base64},
                    },
                ],
            }

            # Add image to message list but don't submit yet
            st.session_state.uploaded_images[img_base64[:100]] = image_message
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Display chat messages
    for message in st.session_state.messages:
        if not isinstance(message["content"], list) and (
            message["content"] == "Here are some images, return the text"
            or message["content"].startswith("Text in Image:")
        ):
            continue
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"], width=200)
            else:
                # Display message
                st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask anything..."):
        # Append uploaded images to messages list
        with st.chat_message("user"):
            for image_message in st.session_state.uploaded_images.values():
                st.session_state.messages.append(image_message)
                st.image(image_message["content"][0]["image_url"]["url"], width=200)
            if prompt == True:
                st.session_state.messages.append(
                    {"role": "user", "content": "Here is an image."}
                )
            else:
                st.markdown(prompt)

        # Process messages for token limits before sending to the model
        if (
            st.session_state["openai_model"]
            != "ft-gpt-4o-mini-2024-07-18:primetime-premier:rc-lr:AHCNvENj"
        ):
            st.session_state.messages = process_messages(
                st.session_state.messages, st.session_state["openai_model"]
            )

        with st.chat_message("assistant"):
            try:
                # Create a placeholder to dynamically update the spinner text
                spinner_placeholder = st.empty()
                # Initial spinner text
                spinner_placeholder.text("Starting response generation...")
                if use_google:
                    response = handleGoogle(prompt, spinner_placeholder)
                else:
                    if (
                        st.session_state["openai_model"]
                        == "ft:gpt-4o-mini-2024-07-18:primetime-premier:rc-lr:AHCNvENj"
                    ):
                        response = handleFinetuned(client, prompt, spinner_placeholder)
                    else:
                        response = handleNormal(client, prompt, spinner_placeholder)
                # Append the assistant response to the message list
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Clear the placeholder after generating the response
                spinner_placeholder.empty()
                paste_result.image_data = None

                if st.session_state.buffer_image != st.session_state.current_image:
                    st.session_state.buffer_image = st.session_state.current_image
                    st.session_state.current_image = None
                    st.session_state.uploaded_images = (
                        {}
                    )  # Clear uploaded images as well
            except Exception as e:
                st.error(f"Error: {str(e)}")
