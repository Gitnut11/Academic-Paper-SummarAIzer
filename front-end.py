import streamlit as st
import requests
import base64

API_URL = "http://localhost:8000/"


def preset_states():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "file_list" not in st.session_state:
        st.session_state.file_list = []
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "login_form_response" not in st.session_state:
        st.session_state.login_form_response = ["", False]


def set_response(response: str, status: bool):
    st.session_state.login_form_response[0] = response
    st.session_state.login_form_response[1] = status
    st.rerun()


def login_page():
    st.title("AI Summarizer", anchor=False)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_container, register_container = st.columns(2)
    with login_container:
        if st.button("Login", use_container_width=True, type="primary"):
            data = {"username": username, "password": password}
            try:
                response = requests.post(API_URL + "login", data=data)
                if response.status_code == 200:
                    user_data = response.json()
                    # get user id and list of files (file id, file name)
                    st.session_state.user_id = user_data['user_id']
                    st.session_state.file_list = user_data['files']
                    st.rerun()
                else:
                    set_response("Invalid username or password", False)
            except requests.exceptions.ConnectionError:
                set_response("Server unavailable", False)

    with register_container:
        if st.button("Register", use_container_width=True, type="primary"):
            data = {"username": username, "password": password}
            try:
                response = requests.post(API_URL + "register", data=data)
                if response.status_code == 200:
                    res = response.json()
                    st.success()
                    set_response(res['message'], True)
                else:
                    set_response("Username already exists", False)
            except requests.exceptions.ConnectionError:
                set_response("Server unavailable", False)

    # render message
    if st.session_state.login_form_response[0] != "":
        if st.session_state.login_form_response[1]:
            st.success(st.session_state.login_form_response[0])
        else:
            st.error(st.session_state.login_form_response[0])

# ------------------------------------------- after logged


def entry_page():
    # write some instruction
    st.title("Paper Summarizer", anchor=False)
    st.markdown(
        """
        Blah blah

        **👈 Uplaoad a pdf paper to start summarizing!
        """
    )

# ------------------------------------------- side bar


def sidebar():
    # pop up when click upload button
    @st.dialog("Upload your file")
    def upload_dialog():
        uploaded_file = st.file_uploader(
            type="pdf", accept_multiple_files=False, label_visibility="collapsed")
        if uploaded_file:
            with st.spinner("Uploading to server..."):
                data = {"user_id": st.session_state.user_id}
                files = {"file": (uploaded_file.name,
                                  uploaded_file, uploaded_file.type)}
                response = requests.post(
                    API_URL + "upload", data=data, files=files)
                st.session_state.file_list = response.json()
                st.rerun()
    # upload button
    if st.sidebar.button("Upload", use_container_width=True, type="primary"):
        upload_dialog()

    # style uploaded files on sidebar
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #f1f2f4;
            color: #202123;
            border: none;
            border-radius: 0%;
            text-align: left;
            transition: all 0.2s ease-in-out;
            font-size: 0.9rem;
            text-align: left;
            justify-content: flex-start;
            display: flex;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #e0e0e0;
            color: #000;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------------------ choose another file
    # file list, each is a button to navigate to the corresponding file
    st.sidebar.subheader("Your file list")
    if len(st.session_state.file_list) != 0:
        for idx, file in enumerate(st.session_state.file_list):
            if st.sidebar.button("📁 " + file['name'], use_container_width=True, key=f"{idx}"):
                # delete last file
                if st.session_state.selected_file is not None and "binary" in st.session_state.selected_file:
                    del st.session_state.selected_file["binary"]
                # select new file
                st.session_state.selected_file = file
                # get chat history
                with st.spinner("Retrieving chat history..."):
                    response = requests.get(API_URL + f"chat/{file["id"]}")
                    if response.status_code == 200:
                        history = response.json()
                        st.session_state.message = history["history"]
                    else:
                        st.error("Something went wrong D:")
                st.rerun()
    else:
        st.sidebar.caption("empty...")


@st.cache_data
def displayPDF(file):
    base64_pdf = base64.b64encode(file).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    return pdf_display

# ------------------------------------------- chat bot


def chat_page():
    @st.dialog(st.session_state.selected_file["name"], width="large")
    def view_file():
        if "binary" not in st.session_state.selected_file:
            with st.spinner("Requesting file..."):
                response = requests.get(
                    API_URL + f"pdf/{str(st.session_state.selected_file["id"])}")
                if response.status_code == 200:
                    st.session_state.selected_file["binary"] = response.content
                else:
                    st.error("error")
        if "binary" in st.session_state.selected_file:
            pdf_display = displayPDF(st.session_state.selected_file["binary"])
            st.markdown(pdf_display, unsafe_allow_html=True)

    # button to open file
    if st.button("📝 Review file", use_container_width=True):
        view_file()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user").markdown(prompt)
        
        # get response from server
        st.session_state.messages.append({"role": "user", "content": prompt})
        data = {
            "prompt": prompt,
            "file_id": st.session_state.selected_file["id"]
        }
        with st.spinner("Wait a minute..."):
            try:
                response = requests.post(API_URL + "chatbot", data=data)
                if response.status_code == 200:
                    reply = response.json()
                    with st.chat_message("assistant"):
                        st.markdown(reply["response"])
                    st.session_state.messages.append({"role": "assistant", "content": reply["response"]})
                else:
                    st.error("Something went wrong :(")
            except Exception as e:
                st.error(e)



if __name__ == "__main__":
    preset_states()
    if st.session_state.user_id is None:
        login_page()
    else:
        st.set_page_config(layout="wide")
        # remove white space on the top
        st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
        sidebar()
        if st.session_state.selected_file is None:
            entry_page()
        else:
            chat_page()
