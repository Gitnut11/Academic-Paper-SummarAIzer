import streamlit as st
import requests
import base64
from streamlit_extras.stylable_container import stylable_container
from tos import TOS

API_URL = "http://localhost:8000"


def preset_states():
    """Initialize session state variables for managing user session, UI state, and cached data."""
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("file_list", [])
    st.session_state.setdefault("selected_file", None)
    st.session_state.setdefault("login_form_response", ["", False])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("page_num", 1)


def set_response(response: str, status: bool):
    """Update the login form response state and re-render the page."""
    st.session_state.login_form_response = [response, status]
    st.rerun()


def login_page():
    """Render login and registration UI, and manage related API requests."""

    @st.dialog("Terms of Service")
    def tos(username, password):
        with st.container(height=700):
            st.markdown(TOS)
        if st.button("Accept & Register", use_container_width=True, type="primary"):
            data = {"username": username, "password": password}
            try:
                response = requests.post(API_URL + "/register", data=data)
                if response.status_code == 200:
                    set_response(response.json()["message"], True)
                else:
                    set_response("Username already exists", False)
            except requests.exceptions.ConnectionError:
                set_response("Server unavailable", False)

    st.title("Scientific Paper SummarAIzer", anchor=False)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_col, register_col = st.columns(2)

    # Login button
    with login_col:
        if st.button("Login", use_container_width=True, type="primary"):
            data = {"username": username, "password": password}
            try:
                response = requests.post(API_URL + "/login", data=data)
                if response.status_code == 200:
                    user_data = response.json()
                    st.session_state.user_id = user_data['user_id']
                    st.session_state.file_list = user_data['files']
                    st.rerun()
                else:
                    set_response("Invalid username or password", False)
            except requests.exceptions.ConnectionError:
                set_response("Server unavailable", False)

    # Register button
    with register_col:
        if st.button("Register", use_container_width=True, type="primary"):
            tos(username, password)

    # Display login/register result
    msg, success = st.session_state.login_form_response
    if msg:
        st.success(msg) if success else st.error(msg)


def entry_page():
    """Render landing page after login with instructions."""
    st.title("Paper Summarizer", anchor=False)
    st.markdown("Welcome!\n\n**👈 Upload a PDF paper to start summarizing!**")


def sidebar():
    """Render the left-hand sidebar for file management and viewing."""

    @st.dialog("Upload your file")
    def upload_dialog():
        uploaded_file = st.file_uploader(
            label="tmp", type="pdf", label_visibility="collapsed"
        )
        if uploaded_file:
            with st.spinner("Uploading to server..."):
                data = {"user_id": st.session_state.user_id}
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(API_URL + "/upload", data=data, files=files)
                if response.status_code == 200:
                    st.session_state.file_list = response.json()
                    st.rerun()
                else:
                    st.error("Error uploading to server")

    @st.dialog("View PDF", width="large")
    def view_file():
        """Display a paginated view of the selected PDF file."""
        if st.session_state.selected_file is None:
            st.write("Please select a file")
            return

        col1, col2, col3 = st.columns([1, 3, 1])
        num = st.session_state.page_num

        with col1:
            if st.button("<", use_container_width=True, disabled=num == 1):
                num -= 1

        with col3:
            if st.button(">", use_container_width=True, disabled=num == st.session_state.selected_file['num']):
                num += 1

        with col2:
            st.markdown(f"<div style='text-align:center; font-size: 18px;'>Page {num} / {st.session_state.selected_file['num']}</div>", unsafe_allow_html=True)

        if num != st.session_state.page_num:
            st.session_state.page_num = num
            with st.spinner("Requesting file..."):
                response = requests.get(API_URL + f"/pdf/{st.session_state.selected_file['id']}/{num}")
                if response.status_code == 200:
                    st.session_state.selected_file["binary"] = response.content
                    st.rerun(scope="fragment")
                else:
                    st.error("Error retrieving page")

        st.markdown(displayPDF(st.session_state.selected_file["binary"]), unsafe_allow_html=True)

    with st.sidebar:
        if st.button("Log-out", use_container_width=True, type="tertiary"):
            st.session_state.clear()
            st.rerun()

        if st.button("Upload", use_container_width=True, type="primary"):
            upload_dialog()

        # Sidebar tab styling
        st.markdown("""
            <style>
            section[data-testid="stSidebar"] [data-testid=stVerticalBlock]{ gap: 0rem; }
            section[data-testid="stSidebar"] button[kind="secondary"]:hover {
                background-color: #e0e0e0;
                color: #000;
            }
            </style>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📁 File selection", "📝 Current file"])

        with tab1:
            st.subheader("Your file list")
            if st.session_state.file_list:
                for idx, file in enumerate(st.session_state.file_list):
                    with stylable_container(key=f"{file['id']}", css_styles="""..."""):
                        if st.button("📄 " + file['name'], use_container_width=True, key=f"{idx}"):
                            with st.spinner("Getting summary..."):
                                try:
                                    res = requests.get(API_URL + f"/smr/{file['id']}")
                                    if res.status_code == 200:
                                        st.session_state.summary = res.json()["summary"]
                                except Exception:
                                    st.error("Error getting summary")

                            if "binary" in st.session_state.selected_file or st.session_state.selected_file is not None:
                                st.session_state.selected_file.pop("binary", None)

                            st.session_state.selected_file = file
                            st.session_state.page_num = 1

                            try:
                                st.session_state.selected_file["binary"] = requests.get(
                                    API_URL + f"/pdf/{file['id']}/1").content
                            except Exception:
                                st.error("Error retrieving PDF")

                            with st.spinner("Retrieving chat history..."):
                                try:
                                    res = requests.get(API_URL + f"/chat/{file['id']}")
                                    if res.status_code == 200:
                                        st.session_state.messages = res.json()["history"]
                                        st.rerun()
                                except Exception:
                                    st.error("Error retrieving chat history")
            else:
                st.caption("empty...")

        with tab2:
            if st.session_state.selected_file:
                if st.button("Open file", key="review", use_container_width=True):
                    view_file()

                st.subheader("Summarize")
                with st.container(height=200):
                    st.write(st.session_state.summary)

                st.subheader("Citation")
                for cite in st.session_state.selected_file["cite"]:
                    if cite["url"] != "null":
                        st.markdown(f"- {cite['title']} [link]({cite['url']})")
                    else:
                        st.markdown(f"- {cite['title']}")
            else:
                st.caption("Select a file first...")


@st.cache_data
def displayPDF(file):
    """Encode PDF binary content as base64 and return HTML iframe for inline display."""
    base64_pdf = base64.b64encode(file).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf"></iframe>'


def chat_page():
    """Render chat interface and handle user prompts."""
    with st.container(height=720):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        data = {"prompt": prompt, "file_id": st.session_state.selected_file["id"]}

        with st.spinner(" a minute..."):
            try:
                response = requests.post(API_URL + "/chatbot", data=data)
                if response.status_code == 200:
                    reply = response.json()
                    st.session_state.messages.append({"role": "assistant", "content": reply["response"]})
                    st.rerun()
                else:
                    st.error("Something went wrong :(")
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    preset_states()
    if st.session_state.user_id is None:
        login_page()
    else:
        st.set_page_config(layout="wide")
        st.markdown("""
            <style>
                .block-container { padding-top: 3rem; padding-bottom: 1rem; padding-left: 5rem; padding-right: 5rem; }
                section[data-testid="stSidebar"] { width: 30% !important; }
            </style>
        """, unsafe_allow_html=True)

        sidebar()
        if st.session_state.selected_file is None:
            entry_page()
        else:
            chat_page()
