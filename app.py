import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
import docx
import PyPDF2
import openpyxl
import requests
import json
import io
import base64
from typing import Dict, List, Any, Optional
import warnings
import tempfile
import os
import time
import re
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🔍 Data Analyst Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalystAgent:
    def __init__(self, together_api_key: str):
        """Initialize the Data Analyst Agent with Together.ai API key"""
        self.api_key = together_api_key
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"  
        self.data = {}
        self.file_info = {}
        self.conversation_history = []
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
    def _rate_limit(self):
        """Implement rate limiting to avoid too many requests error"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
        
    def _call_llm(self, prompt: str, system_message: str = None, max_retries: int = 3) -> str:
        """Call the Llama model via Together.ai API with retry logic"""
        self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history for context (limited to avoid token limits)
        for msg in self.conversation_history[-4:]:  # Reduced from 6 to 4
            messages.append(msg)
            
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1500,  # Reduced from 2048
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, 
                    headers=headers, 
                    json=payload,
                    timeout=30  # Add timeout
                )
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    return "Error: No response content received"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return "Error: Request timed out. Please try again."
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    wait_time = 2 ** attempt
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return "Error: Rate limit exceeded. Please wait a moment and try again."
                elif e.response.status_code == 401:
                    return "Error: Invalid API key. Please check your Together.ai API key."
                else:
                    return f"Error: HTTP {e.response.status_code} - {str(e)}"
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"Error calling LLM: {str(e)}"
        
        return "Error: Failed to get response after multiple attempts"
    
    def load_file(self, uploaded_file) -> Dict[str, Any]:
        """Load and process different file types from Streamlit uploaded file"""
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == '.csv':
                data = pd.read_csv(uploaded_file)
                self.data[filename] = data
                info = self._analyze_dataframe(data, filename)
                
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
                self.data[filename] = data
                info = self._analyze_dataframe(data, filename)
                
            elif file_ext == '.txt':
                content = str(uploaded_file.read(), "utf-8")
                self.data[filename] = content
                info = self._analyze_text(content, filename)
                
            elif file_ext == '.docx':
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                doc = docx.Document(tmp_path)
                content = '\n'.join([para.text for para in doc.paragraphs])
                os.unlink(tmp_path)  # Clean up temp file
                self.data[filename] = content
                info = self._analyze_text(content, filename)
                
            elif file_ext == '.pdf':
                reader = PyPDF2.PdfReader(uploaded_file)
                content = ''
                for page in reader.pages:
                    content += page.extract_text()
                self.data[filename] = content
                info = self._analyze_text(content, filename)
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = Image.open(uploaded_file)
                # Extract text using OCR
                try:
                    text = pytesseract.image_to_string(img)
                except:
                    text = "OCR not available"
                self.data[filename] = {'image': img, 'text': text}
                info = self._analyze_image(img, text, filename)
                
            else:
                return {"error": f"Unsupported file type: {file_ext}"}
            
            self.file_info[filename] = info
            return {"success": f"File {filename} loaded successfully", "info": info}
            
        except Exception as e:
            return {"error": f"Error loading file {filename}: {str(e)}"}
    
    def _analyze_dataframe(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze a pandas DataFrame"""
        analysis = {
            "type": "tabular_data",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),  # Convert to string for JSON serialization
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "sample_data": df.head(3).to_dict()  # Reduced sample size
        }
        
        # Generate statistical summary for numeric columns
        if analysis["numeric_columns"]:
            try:
                analysis["statistics"] = df[analysis["numeric_columns"]].describe().to_dict()
            except:
                analysis["statistics"] = {}
        
        return analysis
    
    def _analyze_text(self, text: str, filename: str) -> Dict[str, Any]:
        """Analyze text content"""
        words = text.split()
        return {
            "type": "text",
            "length": len(text),
            "word_count": len(words),
            "line_count": text.count('\n') + 1,
            "preview": text[:500] + "..." if len(text) > 500 else text
        }
    
    def _analyze_image(self, img: Image.Image, text: str, filename: str) -> Dict[str, Any]:
        """Analyze image content"""
        return {
            "type": "image",
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
            "extracted_text": text,
            "text_length": len(text)
        }
    
    def _generate_fallback_visualization(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate a fallback visualization when LLM fails"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Simple heuristics for visualization type
            query_lower = query.lower()
            
            if 'correlation' in query_lower or 'corr' in query_lower:
                if len(numeric_cols) >= 2:
                    corr_matrix = data[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation Matrix')
                else:
                    ax.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation', 
                           ha='center', va='center', transform=ax.transAxes)
                    
            elif 'histogram' in query_lower or 'distribution' in query_lower:
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    data[col].hist(ax=ax, bins=20)
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                else:
                    ax.text(0.5, 0.5, 'No numeric columns available for histogram', 
                           ha='center', va='center', transform=ax.transAxes)
                    
            elif 'bar' in query_lower or 'count' in query_lower:
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    value_counts = data[col].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Count of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No categorical columns available for bar chart', 
                           ha='center', va='center', transform=ax.transAxes)
                    
            elif 'scatter' in query_lower:
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    ax.scatter(data[x_col], data[y_col], alpha=0.6)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'{x_col} vs {y_col}')
                else:
                    ax.text(0.5, 0.5, 'Need at least 2 numeric columns for scatter plot', 
                           ha='center', va='center', transform=ax.transAxes)
                    
            else:
                # Default: show basic info about the data
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    data[col].plot(kind='line', ax=ax)
                    ax.set_title(f'{col} over index')
                    ax.set_ylabel(col)
                elif len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    value_counts = data[col].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Count of {col}')
                    plt.xticks(rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No suitable columns found for visualization', 
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            return {"success": "Fallback visualization created", "figure": fig, "code": "# Fallback visualization"}
            
        except Exception as e:
            return {"error": f"Error creating fallback visualization: {str(e)}"}
    
    def create_visualization(self, query: str, filename: str = None) -> Dict[str, Any]:
        """Create visualizations based on user query"""
        if not self.data:
            return {"error": "No data loaded. Please load a file first."}
        
        # Use the first loaded file if no filename specified
        if filename is None:
            filename = list(self.data.keys())[0]
        
        if filename not in self.data:
            return {"error": f"File {filename} not found in loaded data."}
        
        data = self.data[filename]
        
        # Only create visualizations for tabular data
        if not isinstance(data, pd.DataFrame):
            return {"error": "Visualizations are only available for tabular data (CSV, Excel files)."}
        
        # Check if data is empty
        if data.empty:
            return {"error": "Cannot create visualization: data is empty."}
        
        # Get visualization suggestions from LLM
        system_msg = """You are a data visualization expert. Generate ONLY Python code using matplotlib and seaborn.
        Requirements:
        1. Use 'data' as the DataFrame variable name
        2. Create the plot using 'ax' (provided as subplot)
        3. Include plt.tight_layout() at the end
        4. DO NOT include plt.show(), plt.savefig(), or fig.show()
        5. Handle missing values appropriately
        6. Keep the code simple and working
        7. Add appropriate titles and labels
        
        Return ONLY executable Python code, no explanations."""
        
        # Limit data info to avoid token limits
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)[:5]
        categorical_cols = list(data.select_dtypes(include=['object']).columns)[:5]
        
        data_info = f"""
        Data shape: {data.shape}
        Numeric columns: {numeric_cols}
        Categorical columns: {categorical_cols}
        Sample data:
        {data.head(2).to_string()}
        """
        
        prompt = f"""
        Create visualization for: {query}
        Data info: {data_info}
        
        Generate Python code using matplotlib/seaborn. Use 'data' as DataFrame variable and 'ax' for plotting.
        """
        
        # Try to get LLM response
        viz_code = self._call_llm(prompt, system_msg)
        
        # Check if LLM call failed
        if viz_code.startswith("Error:"):
            st.warning("LLM service unavailable, using fallback visualization...")
            return self._generate_fallback_visualization(data, query)
        
        try:
            # Clean the code
            viz_code = self._clean_viz_code(viz_code)
            
            # Execute the visualization code
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create safe execution environment
            safe_globals = {
                'data': data, 
                'plt': plt, 
                'sns': sns, 
                'pd': pd, 
                'np': np, 
                'fig': fig, 
                'ax': ax,
                '__builtins__': {}  # Restrict built-in functions for security
            }
            
            # Execute the code
            exec(viz_code, safe_globals)
            plt.tight_layout()
            
            return {"success": "Visualization created", "code": viz_code, "figure": fig}
            
        except Exception as e:
            # If LLM-generated code fails, try fallback
            # st.warning(f"Generated code failed ({str(e)}), using fallback visualization...")
            plt.close('all')  # Clean up any partial plots
            return self._generate_fallback_visualization(data, query)
    
    def _clean_viz_code(self, code: str) -> str:
        """Clean and validate visualization code"""
        # Remove dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'__import__',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError("Code contains potentially dangerous operations")
        
        # Remove plt.show() and plt.savefig() calls
        code = re.sub(r'plt\.show\(\)', '', code)
        code = re.sub(r'plt\.savefig\([^)]*\)', '', code)
        code = re.sub(r'fig\.show\(\)', '', code)
        
        # Ensure tight_layout is called
        if 'tight_layout' not in code:
            code += '\nplt.tight_layout()'
        
        return code
    
    def analyze_data(self, query: str, filename: str = None) -> str:
        """Perform data analysis based on user query"""
        if not self.data:
            return "No data loaded. Please load a file first."
        
        # Use the first loaded file if no filename specified
        if filename is None:
            filename = list(self.data.keys())[0]
        
        if filename not in self.data:
            return f"File {filename} not found in loaded data."
        
        data = self.data[filename]
        file_info = self.file_info[filename]
        
        # Prepare context for LLM (limit size to avoid token limits)
        context = f"""
        File: {filename}
        File type: {file_info['type']}
        """
        
        if isinstance(data, pd.DataFrame):
            # Limit context size
            context += f"""
        Shape: {data.shape}
        Columns: {list(data.columns)[:10]}
        Data preview:
        {data.head(3).to_string()}
        """
            
            # Add basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    context += f"\nBasic stats:\n{data[numeric_cols].describe().head().to_string()}"
                except:
                    pass
                    
        elif isinstance(data, str):
            context += f"\nText preview:\n{data[:800]}..."
        elif isinstance(data, dict) and 'text' in data:
            context += f"\nExtracted text:\n{data['text'][:800]}..."
        
        system_msg = """You are an expert data analyst. Provide clear, actionable insights based on the data. 
        Be specific with numbers and findings. If you cannot analyze due to limitations, provide what you can 
        and suggest next steps. Keep responses concise but informative."""
        
        prompt = f"""
        Context: {context}
        
        User question: {query}
        
        Provide analysis with specific insights and recommendations.
        """
        
        response = self._call_llm(prompt, system_msg)
        
        # Update conversation history only if response is successful
        if not response.startswith("Error:"):
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Limit conversation history size
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        return response

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'files_loaded' not in st.session_state:
    st.session_state.files_loaded = []

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 from Together.ai")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Together.ai API Key",
            type="password",
            help="Get your API key from https://www.together.ai/"
        )
        
        if api_key:
            if st.session_state.agent is None:
                try:
                    st.session_state.agent = DataAnalystAgent(api_key)
                    st.success("✅ Agent initialized!")
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
            
            st.header("📁 File Upload")
            uploaded_files = st.file_uploader(
                "Upload your data files",
                accept_multiple_files=True,
                type=['csv', 'xlsx', 'xls', 'txt', 'docx', 'pdf', 'jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.files_loaded:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            result = st.session_state.agent.load_file(uploaded_file)
                            if "success" in result:
                                st.success(f"✅ {result['success']}")
                                st.session_state.files_loaded.append(uploaded_file.name)
                            else:
                                st.error(f"❌ {result['error']}")
            
            # Show loaded files
            if st.session_state.files_loaded:
                st.header("📊 Loaded Files")
                for filename in st.session_state.files_loaded:
                    st.info(f"📄 {filename}")
        else:
            st.warning("⚠️ Please enter your Together.ai API key to get started")
            st.info("You can get a free API key from https://www.together.ai/")
    
    # Main content area
    if st.session_state.agent and st.session_state.files_loaded:
        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Visualizations", "📋 Data Summary", "⚡ Quick Analysis"])
        
        with tab1:
            st.header("💬 Chat with your data")
            
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
            
            # Chat input
            user_question = st.text_input("Ask a question about your data:", key="chat_input")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Send", key="send_button"):
                    if user_question:
                        # Add user message to history
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        
                        # Get response from agent
                        with st.spinner("🤔 Thinking..."):
                            response = st.session_state.agent.analyze_data(user_question)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Rerun to update the display
                        st.rerun()
            
            with col2:
                if st.button("Clear Chat", key="clear_button"):
                    st.session_state.chat_history = []
                    if st.session_state.agent:
                        st.session_state.agent.conversation_history = []
                    st.rerun()
        
        with tab2:
            st.header("📊 Create Visualizations")
            
            viz_query = st.text_input(
                "Describe the visualization you want:",
                placeholder="e.g., Create a bar chart showing sales by region",
                key="viz_input"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_file = st.selectbox(
                    "Select file for visualization:",
                    options=st.session_state.files_loaded,
                    key="viz_file_select"
                )
            
            with col2:
                if st.button("Create Visualization", key="create_viz_button"):
                    if viz_query:
                        with st.spinner("🎨 Creating visualization..."):
                            result = st.session_state.agent.create_visualization(viz_query, selected_file)
                        
                        if "success" in result:
                            st.success("✅ Visualization created!")
                            
                            # Display the plot
                            if "figure" in result:
                                st.pyplot(result["figure"])
                                plt.close(result["figure"])  # Clean up
                            
                            # Show the code
                            with st.expander("View generated code"):
                                st.code(result["code"], language="python")
                        else:
                            st.error(f"❌ {result['error']}")
                            if "code" in result:
                                with st.expander("View attempted code"):
                                    st.code(result["code"], language="python")
        
        with tab3:
            st.header("📋 Data Summary")
            
            if st.button("Generate Summary", key="summary_button"):
                with st.spinner("📊 Generating summary..."):
                    if st.session_state.agent.data:
                        for filename, data in st.session_state.agent.data.items():
                            st.subheader(f"📄 {filename}")
                            
                            file_info = st.session_state.agent.file_info[filename]
                            
                            if isinstance(data, pd.DataFrame):
                                # Tabular data summary
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Rows", data.shape[0])
                                with col2:
                                    st.metric("Columns", data.shape[1])
                                with col3:
                                    st.metric("Missing Values", data.isnull().sum().sum())
                                
                                # Data types
                                st.write("**Data Types:**")
                                dtype_df = pd.DataFrame({
                                    'Column': data.dtypes.index,
                                    'Type': data.dtypes.values.astype(str)
                                })
                                st.dataframe(dtype_df, use_container_width=True)
                                
                                # Sample data
                                st.write("**Sample Data:**")
                                st.dataframe(data.head(), use_container_width=True)
                                
                                # Statistics for numeric columns
                                numeric_cols = data.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 0:
                                    st.write("**Statistical Summary:**")
                                    st.dataframe(data[numeric_cols].describe(), use_container_width=True)
                            
                            elif file_info['type'] == 'text':
                                # Text summary
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Characters", file_info['length'])
                                with col2:
                                    st.metric("Words", file_info['word_count'])
                                with col3:
                                    st.metric("Lines", file_info['line_count'])
                                
                                st.write("**Text Preview:**")
                                st.text_area("", value=file_info['preview'], height=200, disabled=True)
                            
                            elif file_info['type'] == 'image':
                                # Image summary
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Image Info:**")
                                    st.write(f"Size: {file_info['size']}")
                                    st.write(f"Mode: {file_info['mode']}")
                                    st.write(f"Format: {file_info['format']}")
                                
                                with col2:
                                    st.image(data['image'], caption=filename, use_column_width=True)
                                
                                if file_info['extracted_text'].strip():
                                    st.write("**Extracted Text:**")
                                    st.text_area("", value=file_info['extracted_text'], height=200, disabled=True)
                            
                            st.divider()
        
        with tab4:
            st.header("⚡ Quick Analysis")
            
            st.write("Get instant insights about your data:")
            
            analysis_options = [
                "What are the main patterns in this data?",
                "Show me key statistics and insights",
                "What correlations exist in the data?",
                "Identify any data quality issues",
                "What are the most important features?",
                "Suggest interesting questions to explore"
            ]
            
            selected_analysis = st.selectbox(
                "Choose a quick analysis:",
                options=analysis_options,
                key="quick_analysis_select"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                analysis_file = st.selectbox(
                    "Select file to analyze:",
                    options=st.session_state.files_loaded,
                    key="analysis_file_select"
                )
            
            with col2:
                if st.button("Run Analysis", key="run_analysis_button"):
                    with st.spinner("🔍 Analyzing data..."):
                        response = st.session_state.agent.analyze_data(selected_analysis, analysis_file)
                    
                    st.markdown("### 📊 Analysis Results")
                    st.write(response)
        
        # Sample data generation for demo
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Demo Data")
        if st.sidebar.button("Generate Sample Data"):
            # Create sample datasets
            sample_data = {
                'sales_data.csv': pd.DataFrame({
                    'Product': ['A', 'B', 'C', 'D', 'E'] * 20,
                    'Sales': np.random.randint(100, 1000, 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr'], 100),
                    'Price': np.random.uniform(10, 100, 100)
                }),
                'customer_data.csv': pd.DataFrame({
                    'CustomerID': range(1, 51),
                    'Age': np.random.randint(18, 80, 50),
                    'Gender': np.random.choice(['M', 'F'], 50),
                    'Income': np.random.randint(30000, 150000, 50),
                    'Satisfaction': np.random.randint(1, 6, 50)
                })
            }
            
            # Load sample data into agent
            for filename, data in sample_data.items():
                st.session_state.agent.data[filename] = data
                st.session_state.agent.file_info[filename] = st.session_state.agent._analyze_dataframe(data, filename)
                if filename not in st.session_state.files_loaded:
                    st.session_state.files_loaded.append(filename)
            
            st.sidebar.success("✅ Sample data generated!")
            st.rerun()
    
    elif st.session_state.agent:
        st.info("📁 Please upload some files using the sidebar to get started!")
        
        # Show supported file types
        st.markdown("### 📋 Supported File Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Tabular Data:**
            - 📊 CSV files (.csv)
            - 📈 Excel files (.xlsx, .xls)
            
            **Text Documents:**
            - 📝 Text files (.txt)
            - 📄 Word documents (.docx)
            - 📑 PDF files (.pdf)
            """)
        
        with col2:
            st.markdown("""
            **Images:**
            - 🖼️ JPEG (.jpg, .jpeg)
            - 🖼️ PNG (.png)
            - 🖼️ Bitmap (.bmp)
            
            *Images are processed with OCR to extract text*
            """)
        
        st.markdown("### 🚀 Agent Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Data Analysis**
            - Statistical insights
            - Pattern recognition
            - Trend analysis
            - Data quality assessment
            """)
        
        with col2:
            st.markdown("""
            **Visualizations**
            - Charts and graphs
            - Correlation heatmaps
            - Distribution plots
            - Custom visualizations
            """)
        
        with col3:
            st.markdown("""
            **Q&A System**
            - Natural language queries
            - Context-aware responses
            - Follow-up questions
            - Multi-file analysis
            """)
    
    else:
        st.info("🔑 Please enter your Together.ai API key in the sidebar to get started.")
        
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Get API Key**: Sign up at [Together.ai](https://www.together.ai/) and get your free API key
        2. **Enter API Key**: Paste your API key in the sidebar
        3. **Upload Files**: Upload your data files using the file uploader
        4. **Start Analyzing**: Ask questions, create visualizations, and explore your data!
        """)
        
        # Add troubleshooting section
        st.markdown("### 🔧 Troubleshooting")
        with st.expander("Common Issues and Solutions"):
            st.markdown("""
            **Chat not responding:**
            - Check your API key is valid
            - Wait a moment between requests (rate limiting)
            - Try rephrasing your question
            
            **Visualizations not working:**
            - Ensure you have tabular data (CSV/Excel)
            - Check that your data has appropriate columns
            - Try simpler visualization requests
            
            **Too many requests error:**
            - Wait 30-60 seconds before trying again
            - The app includes rate limiting to prevent this
            
            **File upload issues:**
            - Check file format is supported
            - Ensure file is not corrupted
            - Try smaller files if having memory issues
            """)

if __name__ == "__main__":
    main()