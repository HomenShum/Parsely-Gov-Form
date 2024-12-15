import streamlit as st
import pandas as pd
from datetime import datetime, date
import time
from PIL import Image
import io
import os
import base64
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import docx
from io import BytesIO
import markdown
import asyncio
import json
from parsers import Parser
from config import Config
from data_gatherer import DataGatherer

# Initialize parser with config
config = Config()  # This will load from environment variables
parser = Parser(config)

st.set_page_config(
    page_title="Parsely - Government Form Demo",
    page_icon="📜",
    layout="wide"
)


async def process_document(file_obj):
    """Process document and return both PDF and extracted text"""
    try:
        file_ext = os.path.splitext(file_obj.name)[1].lower()
        
        # Create a list with single file for parser
        temp_bytesio = BytesIO(file_obj.getvalue())
        temp_bytesio.name = file_obj.name
        files = [temp_bytesio]
        
        # Parse the file
        add_system_message(f"Processing file: {file_obj.name}")
        await parser.parse_files(files)
        
        if parser.parsed_documents:
            text_chunks = []
            for doc in parser.parsed_documents:
                text = doc.text
                text_chunks.append(text)
            
            text_content = "\n\n".join(text_chunks)
        else:
            text_content = "No text could be extracted from this file."
            
        # For PDFs, return original file
        if file_ext == '.pdf':
            add_system_message(f"Using original PDF file: {file_obj.name}")
            return {
                'pdf': file_obj,
                'text': text_content
            }
        
        # For all other files, create PDF from text
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        # Split content into lines and write to PDF
        y = height - 100
        lines = text_content.split('\n')
        
        for line in lines:
            if line.strip():
                # Check if line looks like a header
                if line.isupper() or len(line) < 50:
                    c.setFont("Helvetica-Bold", 12)
                else:
                    c.setFont("Helvetica", 12)
                
                # Handle long lines
                while line and y > 100:
                    if len(line) > 80:
                        split_at = line[:80].rfind(' ')
                        if split_at == -1:
                            split_at = 80
                        current_line = line[:split_at]
                        line = line[split_at:].lstrip()
                    else:
                        current_line = line
                        line = ''
                    
                    c.drawString(100, y, current_line.strip())
                    y -= 15
                    
                    # New page if needed
                    if y <= 100 and (line or lines.index(line) < len(lines) - 1):
                        c.showPage()
                        y = height - 100
                        c.setFont("Helvetica", 12)
        
        c.save()
        pdf_buffer.seek(0)
        
        # Create a named buffer for the PDF
        pdf_buffer.name = file_obj.name.rsplit('.', 1)[0] + '.pdf'
        
        return {
            'pdf': pdf_buffer,
            'text': text_content
        }
            
    except Exception as e:
        error_msg = f"Error processing {file_obj.name}: {str(e)}"
        st.error(error_msg)
        add_system_message(error_msg)
        return None

def add_system_message(message):
    """Add a message to the system dialog history"""
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    st.session_state.system_messages.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

@st.fragment
def main_content_fragment_DocumentUpload_component(key_prefix=""):
    """Component for document upload and preview functionality"""
    st.header("Upload Documents")
    
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Upload your documents here",
            type=["pdf", "docx", "doc", "txt", "rtf", "csv", "tsv", 
                  "html", "xml", "png", "jpg", "jpeg", "pptx", "xlsx"],
            accept_multiple_files=True,
            key=f"{key_prefix}_file_uploader"
        )
        
        if uploaded_files:
            st.session_state.setdefault('uploaded_files', {})[key_prefix] = uploaded_files

            # Process each file
            processed_files = []
            for file in uploaded_files:
                try:
                    # Check if file was already processed
                    existing_files = st.session_state.get('processed_files', {}).get(key_prefix, [])
                    existing_file = next(
                        (f for f in existing_files if f['pdf'].name == file.name), 
                        None
                    )
                    
                    should_process = True
                    if existing_file:
                        add_system_message(f"File already processed: {file.name}")
                        # Create a unique key for the button
                        button_key = f"reprocess_{key_prefix}_{file.name}"
                        if st.button("Process Again?", key=button_key):
                            add_system_message(f"Action: User clicked Yes on parsing the same named document {file.name}")
                        else:
                            should_process = False
                            processed_files.append(existing_file)
                            add_system_message(f"Using existing processed file: {file.name}")
                    
                    if should_process:
                        # Reset file pointer
                        file.seek(0)
                        
                        # Process the file
                        add_system_message(f"Processing file: {file.name}")
                        result = asyncio.run(process_document(file))
                        if result:
                            processed_files.append(result)
                            add_system_message(f"Successfully processed: {file.name}")
                    
                except Exception as e:
                    error_msg = f"Error processing {file.name}: {str(e)}"
                    st.error(error_msg)
                    add_system_message(error_msg)
            
            # Store processed files in session state
            st.session_state.setdefault('processed_files', {})[key_prefix] = processed_files
            
            # Display files
            if processed_files:
                # File selection dropdown
                file_names = [f['pdf'].name for f in processed_files]
                selected_file = st.selectbox(
                    "Select a file to preview",
                    file_names,
                    index=0,
                    key=f"{key_prefix}_file_select"
                )
                
                # Get selected file data
                file_data = next(
                    (f for f in processed_files if f['pdf'].name == selected_file), 
                    None
                )
                
                if file_data:
                    st.write(f"### 📄 {selected_file}")
                    
                    # Preview tabs
                    pdf_tab, text_tab = st.tabs(["PDF View", "Text View"])
                    
                    with pdf_tab:
                        # PDF preview using iframe
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(file_data["pdf"].getvalue()).decode()}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    
                    with text_tab:
                        # Create a container for the text with scrolling
                        with st.container():
                            # Split the text into paragraphs and display each
                            paragraphs = file_data['text'].split('\n\n')
                            for paragraph in paragraphs:
                                if paragraph.strip():
                                    st.write(paragraph)
                                    st.write("")  # Add spacing between paragraphs
                    
                    # File details and downloads
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Size:** {len(file_data['pdf'].getvalue())/1024:.1f} KB")
                        st.download_button(
                            "Download PDF",
                            file_data['pdf'].getvalue(),
                            file_name=selected_file,
                            mime="application/pdf",
                            key=f"{key_prefix}_download_pdf_{selected_file}"
                        )
                    with col2:
                        st.write("**Type:** PDF")
                        st.download_button(
                            "Download Text",
                            file_data['text'],
                            file_name=selected_file.replace('.pdf', '.txt'),
                            mime="text/plain",
                            key=f"{key_prefix}_download_text_{selected_file}"
                        )
                    
                    st.divider()

@st.fragment(run_every=10)
def sidebar_content_fragment_Settings_component():
    """Settings component in sidebar"""
    # UAuth block with Settings
    with st.container(border=True, height=300):
        st.header("👤 User Account")
        
        # Settings tabs
        settings_tab, usage_tab = st.tabs(["Settings", "Usage"])
        
        with settings_tab:
            st.selectbox("Language", ["English", "Spanish", "Chinese", "Vietnamese"])
            st.checkbox("Enable Voice Assistant", value=True)
            
        with usage_tab:
            st.text("Session Statistics")
            st.metric("Forms Processed", len(st.session_state.get('processed_files', {}).get('', [])))
        
@st.fragment(run_every=10)
def sidebar_content_fragment_SystemDialog_component():
    """System Dialog component in sidebar"""
    with st.container(border=True, height=300):
        st.header("💬 System Dialog")
        
        # Display system messages
        if 'system_messages' in st.session_state and st.session_state.system_messages:
            for message in st.session_state.system_messages[::-1]:
                st.text(message)
        else:
            st.text("No system messages yet")

@st.fragment()
def sidebar_content_fragment_PydanticAIAgentChat_component():
    """AI Agent Chat component in sidebar"""
    with st.container(border=True, height=600):
        st.header("🌱 Parsely AI Assistant")
        
        # Initialize messages in session state if not present
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Chat input area
        col1, col2 = st.columns([5, 1])
        with col1:
            # Using st.chat_input instead of text_input
            prompt = st.chat_input(
                "Ask about forms or assistance...",
                key="chat_input"
            )
        with col2:
            clear = st.button("Clear", use_container_width=True)
                
        # Divider between chat history and input
        st.divider()
        
        # Create a container for chat history with scrolling
        chat_container = st.container(height=400)
        with chat_container:
            # Display previous messages in reverse order
            for message in st.session_state.messages[::-1]:
                # Get timestamp from message or use current time if not present
                timestamp = message.get("timestamp", datetime.now().strftime("%H:%M:%S"))
                
                if message["role"] == "user":
                    # st.markdown(f"🧑 **You** ({timestamp}):")
                    # st.markdown(message["content"])
                    user_message_timestamp = f"🧑 **You** ({timestamp}): \n\n{message['content']}"
                    
                    # Display user and assistant message with timestamp
                    st.markdown(user_message_timestamp)
                    st.markdown(assistant_message_timestamp)
                    st.divider()
                else:
                    # st.markdown(f"🤖 **Assistant** ({timestamp}):")
                    # st.markdown(message["content"])
                    assistant_message_timestamp = f"🤖 **Assistant** ({timestamp}): \n\n{message['content']}"
        
        # Handle input
        if prompt:
            # Get current timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Add user message to chat history with timestamp
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": current_time
            })
            
            # Add assistant response
            assistant_response = """Here's how I can help you with government forms:
            
            1. Form Selection: I can help you choose the right form
            2. Field Guidance: Explain what information goes where
            3. Document Requirements: List what you need to prepare
            4. Eligibility Check: Verify if you qualify
            5. Submission Help: Guide you through the process

            What specific assistance do you need?"""
            
            # Add assistant response to chat history with timestamp
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": current_time
            })
            
            # Rerun to update the chat history
            st.rerun()
        
        # Handle clear button
        if clear:
            st.session_state.messages = []
            st.rerun()

@st.fragment
def main_content_fragment_FormsView_component(key_prefix=""):
    """Component for forms viewing functionality"""
    st.header("View Forms")
    forms_list = [
        "General Intake Form",
        "Housing Assistance Application",
        "Benefits Renewal Form",
        "Service Request Form"
    ]
    selected_form = st.selectbox(
        "Select a form", 
        forms_list,
        key=f"{key_prefix}_form_select"
    )
    
    mode = st.radio(
        "Mode", 
        ["Image Parse Mode", "Analysis Mode"],
        key=f"{key_prefix}_mode_select"
    )
    return selected_form, mode

@st.fragment
def main_content_fragment_DataEditor_component(key_prefix=""):
    """Component for data editing functionality"""
    st.header("Form Data Editor")
    st.write("View and edit form data")
    
    df = pd.DataFrame({
        'Form ID': ['001', '002', '003'],
        'Status': ['Completed', 'In Progress', 'New'],
        'Last Modified': ['2024-12-07', '2024-12-06', '2024-12-05']
    })
    
    edited_df = st.data_editor(
        df,
        key=f"{key_prefix}_data_editor"
    )
    return edited_df

@st.fragment
def main_content_fragment_AgentWebscrapeDisplay_component():
    import streamlit as st
    import asyncio
    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel, Field
    from dataclasses import dataclass
    from typing import Any, List, Optional
    from tavily import TavilyClient, AsyncTavilyClient
    import os
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    @dataclass
    class SearchDataclass:
        max_results: int
        todays_date: str

    @dataclass
    class ResearchDependencies:
        todays_date: str

    class ResourceLink(BaseModel):
        title: str = Field(description="Title of the resource/organization")
        url: str = Field(description="URL of the resource/organization")
        summary: str = Field(description="A short summary of what can be done at this URL")
        instructions: str = Field(description="Step-by-step instructions on how to apply or use the resource")

    class FormField(BaseModel):
        field_name: str
        description: str
        example_input: Optional[str] = None

    class ApplicationForm(BaseModel):
        form_name: str
        form_url: str
        instructions: str
        required_fields: List[FormField]

    class ResearchResult(BaseModel):
        research_title: str = Field(description='A top-level Markdown heading')
        research_main: str = Field(description='Main section with detailed answers')
        research_bullets: str = Field(description='Summary bullet points for quick reference')
        research_resources: List[ResourceLink] = Field(default_factory=list, description='List of resources with URLs and instructions')
        research_forms: List[ApplicationForm] = Field(default_factory=list, description='List of application forms with required fields and instructions')

    tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    search_agent = Agent(
        'openai:gpt-4o',
        deps_type=ResearchDependencies,
        result_type=ResearchResult,
        system_prompt=(
            "You are a helpful research assistant, an expert in research. "
            "When given a user query, you should:\n"
            "1. Identify key resources and perform 3-5 searches (with a query_number) to combine results.\n"
            "2. Return a detailed answer with a heading in research_title.\n"
            "3. In research_main, provide an in-depth explanation.\n"
            "4. In research_bullets, provide a bullet list summary.\n"
            "5. In research_forms and research_resources, provide direct URLs and instructions. "
            "   Prioritize online forms that can be filled out directly.\n"
            "6. Prompt the user for data they will need to provide for these forms.\n"
            "7. Adhere to the given JSON schema."
        )
    )

    @search_agent.tool
    async def get_search(search_data:RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
        """Get the search for a keyword query."""
        print(f"Search query {query_number}: {query}")
        max_results = search_data.deps.max_results
        results = await tavily_client.get_search_context(query=query, max_results=max_results)
        return results

    @search_agent.system_prompt
    async def add_current_date(ctx: RunContext[ResearchDependencies]) -> str:
        todays_date = ctx.deps.todays_date
        system_prompt = (
            "You are a helpful research assistant, you are an expert in research. "
            "If you are given a question, you write strong keywords to do 3-5 searches in total "
            "(each with a query_number) and then combine the results. "
            f"If you need today's date it is {todays_date}. "
            "Return resources, forms, and URLs as requested."
        )
        return system_prompt

    def capture_screenshot(url: str) -> Optional[str]:
        """Use Selenium to open a URL and take a screenshot, returning the local file path."""
        if not url.startswith("http"):
            return None

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_window_size(1200, 800)
            driver.get(url)
            time.sleep(2)  # wait for the page to load
            screenshot_path = f"screenshots/{uuid.uuid4().hex}.png"
            os.makedirs("screenshots", exist_ok=True)
            driver.save_screenshot(screenshot_path)
            driver.quit()
            return screenshot_path
        except Exception as e:
            print(f"Failed to capture screenshot for {url}: {e}")
            return None

    def assist_underserved_communities():
        """Function to assist underserved communities with financial and resource access."""
        st.subheader("🤝 Community Assistance")
        
        # Initialize the research agent if not already done
        if 'research_agent_initialized' not in st.session_state:
            current_date = date.today().isoformat()
            deps = SearchDataclass(max_results=3, todays_date=current_date)
            st.session_state.research_agent_initialized = True
            st.session_state.deps = deps
            
        assistance_categories = {
            "Financial Services": [
                "Bank account access",
                "Low-cost checking accounts",
                "Second chance banking",
                "Credit building programs",
                "Small business loans",
                "Microfinance opportunities"
            ],
            "Housing Resources": [
                "Affordable housing programs",
                "Rent assistance",
                "First-time homebuyer programs",
                "Housing counseling",
                "Emergency shelter",
                "Home repair assistance"
            ],
            "Education & Employment": [
                "Job training programs",
                "Adult education",
                "Financial literacy",
                "Vocational training",
                "Small business development",
                "Career counseling"
            ],
            "Healthcare Access": [
                "Low-cost clinics",
                "Prescription assistance",
                "Mental health services",
                "Health insurance navigation",
                "Preventive care programs",
                "Dental services"
            ],
            "Social Services": [
                "Food assistance",
                "Utility assistance",
                "Legal aid",
                "Transportation assistance",
                "Childcare resources",
                "Senior services"
            ]
        }
        
        selected_category = st.selectbox("Select Assistance Category", list(assistance_categories.keys()))
        selected_service = st.selectbox(f"Select {selected_category} Service", assistance_categories[selected_category])
        location = st.text_input("Enter your location (City, State)", "")
        languages = ["English", "Spanish", "Chinese", "Vietnamese", "Korean", "Other"]
        preferred_language = st.selectbox("Preferred Language", languages)
        specific_needs = st.text_area("Describe any specific needs or circumstances (optional)", help="This helps us provide more relevant resources")
        urgency = st.select_slider("How urgent is your need?", options=["Not urgent", "Somewhat urgent", "Urgent", "Very urgent", "Emergency"], value="Not urgent")
        
        if st.button("Find Resources", type="primary"):
            if not location:
                st.warning("Please enter your location to find relevant resources.")
                return
            
            query = f"""
            Find resources for {selected_service} in {location}.
            Focus on:
            - Programs specifically for underserved communities
            - Services available in {preferred_language}
            - Eligibility requirements and application process
            - Contact information and hours of operation
            - Required documents
            - Any fees or costs involved

            Additional context: {specific_needs if specific_needs else 'No specific needs mentioned'}
            Urgency level: {urgency}

            Also, return resource links (with URLs) and online forms as top priority, with instructions.
            Prompt the user for data needed for these forms.
            """

            try:
                with st.spinner("Researching available resources..."):
                    result = asyncio.run(search_agent.run(query, deps=st.session_state.deps))
                    
                    if result and result.data:
                        with st.expander("📋 Available Resources", expanded=True):
                            st.markdown(result.data.research_title)
                            st.markdown(result.data.research_main)
                            st.markdown(result.data.research_bullets)
                            
                            # Display forms first
                            if result.data.research_forms:
                                st.subheader("📝 Application Forms (Top Priority)")
                                for fdata in result.data.research_forms:
                                    st.markdown(f"**Form Name:** {fdata.form_name}")
                                    # Always display the form URL
                                    st.markdown(f"**[Form URL]({fdata.form_url})**")
                                    st.write("**Instructions:**")
                                    st.write(fdata.instructions)
                                    st.write("**Required Fields:**")
                                    for field in fdata.required_fields:
                                        st.write(f"- **{field.field_name}:** {field.description}")
                                        if field.example_input:
                                            st.write(f"  Example: {field.example_input}")

                                    # Capture and display screenshot of the form URL
                                    ss_path = capture_screenshot(fdata.form_url)
                                    if ss_path:
                                        st.image(ss_path, caption=f"{fdata.form_name} Preview", use_container_width=True)
                                    st.write("---")
                            
                            # Display additional resource links
                            if result.data.research_resources:
                                st.subheader("🔗 Additional Resource Links")
                                for r in result.data.research_resources:
                                    # Always display the resource URL
                                    st.markdown(f"**[{r.title}]({r.url})**")
                                    st.write(r.summary)
                                    st.write("**Instructions:**")
                                    st.write(r.instructions)
                                    # Capture and display screenshot of the resource URL
                                    ss_path = capture_screenshot(r.url)
                                    if ss_path:
                                        st.image(ss_path, caption=f"{r.title} Preview", use_container_width=True)
                                    st.write("---")
                        
                        # Next steps
                        with st.expander("👉 Next Steps", expanded=True):
                            st.write("### Recommended Actions:")
                            st.write("1. Start with the online forms above to apply directly.")
                            st.write("2. Review other resources for additional support.")
                            st.write("3. Gather any required documents mentioned in the requirements.")
                            st.write("4. Contact the organizations directly if needed.")
                            
                            if urgency in ["Urgent", "Very urgent", "Emergency"]:
                                st.warning("""
                                ### Emergency Resources
                                - Call 211 for immediate assistance
                                - Visit your local emergency assistance office
                                - Contact the listed organizations and mention your urgent situation
                                """)

            except Exception as e:
                st.error(f"An error occurred while researching resources: {str(e)}")
        
        with st.expander("ℹ️ Additional Information"):
            st.write("""
            ### Know Your Rights
            - You have the right to access public services regardless of your background
            - Many programs have accommodations for language and disability needs
            - You can request an interpreter if needed

            ### Tips for Success
            - Keep copies of all submitted documents
            - Take notes during phone calls and meetings
            - Ask for written confirmation of applications
            - Follow up regularly on your applications

            ### Need More Help?
            Contact your local legal aid office or community advocacy organization for additional assistance.
            """)
            
    # Run the assist_underserved_communities function
    assist_underserved_communities()

@st.fragment
def main_content_fragment_CaptureScreenshot_component(url):
    """Capture full page screenshot of a webpage using Selenium"""
    try:
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--start-maximized")
        
        # Initialize the Chrome WebDriver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # Navigate to URL
        driver.get(url)
        
        # Get the total height of the page
        total_height = driver.execute_script("""
            return Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.offsetHeight,
                document.body.clientHeight,
                document.documentElement.clientHeight
            );
        """)
        
        # Set window size to capture full page
        width = driver.execute_script("return document.documentElement.scrollWidth")
        driver.set_window_size(width, total_height)
        
        # Wait for any dynamic content to load
        st.spinner("Waiting for page to fully load...")
        driver.implicitly_wait(5)
        
        # Scroll through the page to ensure all content is loaded
        current_height = 0
        while current_height < total_height:
            driver.execute_script(f"window.scrollTo(0, {current_height});")
            current_height += 500
            driver.implicitly_wait(0.1)
        
        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")
        
        # Create temporary file for screenshot
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            screenshot_path = tmp_file.name
            # Take full page screenshot
            driver.save_screenshot(screenshot_path)
        
        driver.quit()
        return screenshot_path
    except Exception as e:
        st.error(f"Error capturing screenshot: {str(e)}")
        return None

@st.fragment
def main_content_fragment():
    """Main application entry point"""
    # Main content area
    st.title("📜 Parsely - Government Form Demo")

    # Two columns for main content
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            all_tab, doc_tab, forms_tab, data_tab = st.tabs([
                "All",
                "Upload Document", 
                "View Forms", 
                "Data Editor"
            ])
            
            # All tab shows everything
            with all_tab:
                main_content_fragment_DocumentUpload_component("all")
                st.divider()
                main_content_fragment_FormsView_component("all")
                st.divider()
                main_content_fragment_DataEditor_component("all")
            
            # Individual tabs
            with doc_tab:
                main_content_fragment_DocumentUpload_component("doc")
                
            with forms_tab:
                main_content_fragment_FormsView_component("forms")
                
            with data_tab:
                main_content_fragment_DataEditor_component("data")

    with col2:
        # Dynamic height adjustment for main content
        with st.container(border=True):
            st.header("Webscrape Contents")
            custom_url_to_scrape = st.text_input("Enter URL to scrape")
            
            # Add form interaction options
            form_interaction = st.checkbox("Enable Form Interaction", value=False)
            auto_fill = st.checkbox("Auto-fill Form Fields", value=False, disabled=not form_interaction)
            show_raw = st.checkbox("Show Raw HTML", value=False)
            
            if st.button("Fetch Contents"):
                if custom_url_to_scrape:
                    try:
                        # Initialize DataGatherer with default config
                        data_gatherer = DataGatherer({})
                        
                        if form_interaction:
                            # Handle form fields and interactions
                            form_data = data_gatherer.handle_form(custom_url_to_scrape)
                            
                            if form_data and form_data.get("forms"):
                                st.success("Successfully detected form fields!")
                                
                                # Display form fields in a hierarchical structure
                                for i, form in enumerate(form_data["forms"]):
                                    with st.expander(f"Form {i+1} Details", expanded=True):
                                        # Create input fields for form data
                                        form_inputs = {}
                                        current_section = None
                                        
                                        for field in form["fields"]:
                                            field_type = field.get("type", "text")
                                            field_label = field.get("label", field.get("name", "Unnamed Field"))
                                            field_section = field.get("section", "")
                                            
                                            # Handle new sections
                                            if field_section and field_section != current_section:
                                                st.subheader(field_section)
                                                current_section = field_section
                                            
                                            # Create a container for the field
                                            with st.container():
                                                # Display field description if available
                                                if field.get("description"):
                                                    st.markdown(f"*{field['description']}*")
                                                
                                                # Handle different field types
                                                if field_type == "select":
                                                    options = field.get("options", [])
                                                    if options:
                                                        # Handle image options
                                                        if any(opt.get("image_url") for opt in options):
                                                            cols = st.columns(len(options))
                                                            for idx, opt in enumerate(options):
                                                                with cols[idx]:
                                                                    if opt.get("image_url"):
                                                                        st.image(opt["image_url"], caption=opt["text"])
                                                                    st.checkbox(opt["text"], key=f"{field['name']}_{idx}")
                                                        else:
                                                            form_inputs[field["name"]] = st.selectbox(
                                                                field_label,
                                                                options=[opt["text"] for opt in options]
                                                            )
                                                
                                                elif field_type == "radio":
                                                    options = field.get("options", ["Yes", "No"])
                                                    # Handle image radio options
                                                    if any(isinstance(opt, dict) and opt.get("image_url") for opt in options):
                                                        st.write(field_label)
                                                        cols = st.columns(len(options))
                                                        for idx, opt in enumerate(options):
                                                            with cols[idx]:
                                                                if isinstance(opt, dict):
                                                                    if opt.get("image_url"):
                                                                        st.image(opt["image_url"], caption=opt["text"])
                                                                    st.radio("", [opt["text"]], key=f"{field['name']}_{idx}")
                                                                else:
                                                                    st.radio("", [opt], key=f"{field['name']}_{idx}")
                                                    else:
                                                        form_inputs[field["name"]] = st.radio(field_label, options)
                                                
                                                elif field_type == "checkbox":
                                                    form_inputs[field["name"]] = st.checkbox(field_label)
                                                
                                                elif field_type == "textarea":
                                                    form_inputs[field["name"]] = st.text_area(field_label)
                                                
                                                else:
                                                    form_inputs[field["name"]] = st.text_input(field_label)
                                                
                                                # Display any validation rules
                                                if field.get("validation"):
                                                    st.caption(f"*{field['validation']}*")
                                        
                                        if st.button(f"Submit Form {i+1}"):
                                            st.write("Form data:", form_inputs)
                                            st.success("Form submitted successfully!")
                            
                            if auto_fill:
                                with st.expander("Auto-fill Suggestions"):
                                    context = {
                                        "url": custom_url_to_scrape,
                                        "form_data": form_data
                                    }
                                    recommendations = data_gatherer.recommend_fill(context)
                                    st.json(recommendations)
                            
                            if show_raw:
                                with st.expander("Raw HTML"):
                                    st.code(data_gatherer.driver.page_source)
                        
                        else:
                            # Regular content extraction
                            content = data_gatherer.extract_text_content(custom_url_to_scrape)
                            if content:
                                st.write(content)
                            else:
                                st.error("No content found or error occurred")
                        
                        # Clean up
                        del data_gatherer
                        
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
                        st.exception(e)  # This will show the full traceback
                else:
                    st.warning("Please enter a URL to scrape")
                    
            st.divider()
            main_content_fragment_AgentWebscrapeDisplay_component()

    # Footer
    st.divider()
    st.caption("Parsely Tool - Alameda County Forms Assistant")

def main():
    """Main application entry point"""
    # Sidebar components
    with st.sidebar:
        sidebar_content_fragment_Settings_component()
        st.divider()
        sidebar_content_fragment_PydanticAIAgentChat_component()
        st.divider()
        sidebar_content_fragment_SystemDialog_component()
    
    # Main content
    main_content_fragment()

if __name__ == "__main__":
    main()