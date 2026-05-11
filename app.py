"""
Smart Email Summarizer - An AI-powered application to analyze and summarize emails.
Uses Hugging Face Transformers and Streamlit for an interactive user experience.
"""

import streamlit as st
import re
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_priority(email_text):
    """
    Detect the priority level of an email based on keywords.
    
    Args:
        email_text (str): The email content to analyze
        
    Returns:
        str: Priority level - 'High', 'Medium', or 'Low'
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = email_text.lower()
    
    # Define keywords for each priority level
    high_priority_keywords = [
        'urgent', 'asap', 'immediately', 'critical', 'emergency',
        'high priority', 'must', 'important', 'deadline', 'today',
        'immediately', 'right now', 'crucial'
    ]
    
    medium_priority_keywords = [
        'soon', 'important', 'please review', 'needed', 'required',
        'week', 'next week', 'should'
    ]
    
    # Check for high priority keywords
    for keyword in high_priority_keywords:
        if keyword in text_lower:
            return "🔴 High"
    
    # Check for medium priority keywords
    for keyword in medium_priority_keywords:
        if keyword in text_lower:
            return "🟡 Medium"
    
    # Default to low priority
    return "🟢 Low"


def extract_deadlines(email_text):
    """
    Extract deadline and date references from email text using regex.
    
    Args:
        email_text (str): The email content to analyze
        
    Returns:
        list: List of detected deadlines/dates
    """
    deadlines = []
    
    # Regex patterns for common date formats
    patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}',  # Month DD, YYYY
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
        r'\b(?:today|tomorrow|next\s+week|next\s+month|this\s+week|this\s+month|end\s+of\s+week|end\s+of\s+month|by\s+\d{1,2})\b',  # Relative dates
        r'\bby\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # "by Month DD"
    ]
    
    # Search for all matching patterns
    for pattern in patterns:
        matches = re.finditer(pattern, email_text, re.IGNORECASE)
        for match in matches:
            deadline = match.group(0).strip()
            if deadline not in deadlines:
                deadlines.append(deadline)
    
    return deadlines if deadlines else ["No specific deadline found"]


def extract_action_items(email_text, summary_text):
    """
    Extract action items from email and summary text using sentence analysis.
    
    Args:
        email_text (str): The original email content
        summary_text (str): The generated summary text
        
    Returns:
        list: List of action items
    """
    action_items = []
    
    # Keywords that typically indicate action items
    action_keywords = [
        'please', 'can you', 'could you', 'would you', 'will you',
        'should', 'must', 'need to', 'required to', 'have to',
        'action item', 'todo', 'task', 'do', 'send', 'create',
        'update', 'review', 'approve', 'confirm', 'check',
        'submit', 'complete', 'finish', 'prepare', 'schedule'
    ]
    
    # Combine email and summary for analysis
    combined_text = email_text + " " + summary_text
    
    # Tokenize text into sentences
    try:
        sentences = sent_tokenize(combined_text)
    except:
        # Fallback if sentence tokenization fails
        sentences = combined_text.split('. ')
    
    # Check each sentence for action item keywords
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains action keywords
        if any(keyword in sentence_lower for keyword in action_keywords):
            # Clean up the sentence
            cleaned_sentence = sentence.strip()
            if cleaned_sentence and len(cleaned_sentence) > 10:
                # Avoid duplicate action items
                if cleaned_sentence not in action_items:
                    action_items.append(f"• {cleaned_sentence}")
    
    return action_items if action_items else ["No specific action items identified"]


@st.cache_resource
def load_summarizer():
    """Load the BART summarizer model once and cache it."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        return {'tokenizer': tokenizer, 'model': model}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def summarize_email(email_text):
    """
    Summarize the email using the BART model from Hugging Face.
    
    Args:
        email_text (str): The email content to summarize
        
    Returns:
        str: The generated summary
    """
    try:
        import torch
        
        # Load the cached model
        model_data = load_summarizer()
        if model_data is None:
            return "Unable to load the summarization model. Please try again."
        
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        # BART model has a token limit, so we need to truncate if necessary
        # Maximum input is around 1024 tokens
        max_length = 1024
        truncated_text = " ".join(email_text.split()[:max_length])
        
        # Tokenize input
        inputs = tokenizer.encode(truncated_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=50,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode summary
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary_text
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return "Unable to generate summary. Please try again."


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

# Configure page settings
st.set_page_config(
    page_title="SummarizeAI - Smart Email Summarizer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for modern dark theme with glassmorphism
st.markdown("""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Geist:wght@400;500;600&display=swap');

        /* Material Symbols font */
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

        /* Base theme colors */
        :root {
            --primary: #4f46e5;
            --primary-container: #4f46e5;
            --secondary: #4cd7f6;
            --secondary-container: #03b5d3;
            --tertiary: #ffb695;
            --error: #ffb4ab;
            --surface: #0b1326;
            --surface-container: #171f33;
            --surface-container-low: #131b2e;
            --surface-container-high: #222a3d;
            --surface-variant: #2d3449;
            --on-surface: #dae2fd;
            --on-surface-variant: #c7c4d8;
            --outline: #918fa1;
            --outline-variant: #464555;
            --background: #0b1326;
            --on-background: #dae2fd;
        }

        /* Base styles */
        body {
            background-color: var(--background) !important;
            color: var(--on-background) !important;
            font-family: 'Inter', sans-serif !important;
        }

        .main {
            background-color: var(--background) !important;
            color: var(--on-background) !important;
        }

        /* Glass card effect */
        .glass-card {
            background: rgba(23, 31, 51, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 1rem !important;
        }

        /* AI gradient glow */
        .ai-gradient-glow {
            background: linear-gradient(135deg, #4f46e5 0%, #03b5d3 100%) !important;
            box-shadow: 0 0 20px rgba(79, 70, 229, 0.4) !important;
        }

        /* Priority colors */
        .priority-high {
            color: var(--error) !important;
        }

        .priority-medium {
            color: var(--secondary) !important;
        }

        .priority-low {
            color: var(--on-surface-variant) !important;
        }

        /* Material symbols */
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }

        /* Custom button styles */
        .custom-button {
            background: linear-gradient(135deg, rgba(3, 181, 211, 0.1) 0%, rgba(79, 70, 229, 0.1) 100%) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(195, 192, 255, 0.2) !important;
            border-radius: 0.75rem !important;
            color: var(--on-surface) !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        .custom-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3) !important;
        }

        /* Sidebar styling */
        .sidebar-content {
            background: var(--surface-container) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--surface-container);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary);
        }

        /* Input styling */
        .stTextArea textarea {
            background: rgba(45, 52, 73, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 1rem !important;
            color: var(--on-surface) !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
        }

        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #4f46e5 0%, #03b5d3 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 0.75rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4) !important;
        }

        /* Success/info messages */
        .stSuccess, .stInfo, .stWarning {
            background: rgba(23, 31, 51, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 0.75rem !important;
            color: var(--on-surface) !important;
        }

        /* Columns styling */
        .stColumn {
            background: transparent !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--on-surface) !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Decorative elements */
        .ambient-glow-1 {
            position: fixed;
            top: -10%;
            right: -10%;
            width: 500px;
            height: 500px;
            background: rgba(79, 70, 229, 0.1);
            border-radius: 50%;
            filter: blur(120px);
            pointer-events: none;
            z-index: -1;
        }

        .ambient-glow-2 {
            position: fixed;
            bottom: -10%;
            left: 20%;
            width: 400px;
            height: 400px;
            background: rgba(3, 181, 211, 0.05);
            border-radius: 50%;
            filter: blur(120px);
            pointer-events: none;
            z-index: -1;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div style="padding: 1.5rem 0;">
            <div style="padding: 0 1.5rem; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem; font-weight: 700; color: #4f46e5; font-family: 'Inter', sans-serif;">SummarizeAI</h1>
                <p style="font-size: 0.75rem; color: #c7c4d8; font-weight: 500; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em;">Pro Plan</p>
            </div>

            <div style="padding: 0 1.5rem; margin-bottom: 2rem;">
                <button style="width: 100%; background: #4f46e5; color: white; border: none; border-radius: 0.75rem; padding: 0.75rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 0.5rem; cursor: pointer; transition: all 0.3s ease;" onclick="document.querySelector('textarea').focus()">
                    <span class="material-symbols-outlined" style="font-size: 1.25rem;">add</span>
                    <span>New Summary</span>
                </button>
            </div>

            <nav style="flex: 1; display: flex; flex-direction: column; gap: 0.25rem;">
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #4f46e5; font-weight: 700; border-right: 4px solid #4f46e5; background: rgba(79, 70, 229, 0.1); text-decoration: none; border-radius: 0 0.5rem 0.5rem 0;">
                    <span class="material-symbols-outlined">dashboard</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Workspace</span>
                </a>
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #c7c4d8; text-decoration: none; border-radius: 0 0.5rem 0.5rem 0; transition: all 0.2s ease;" onmouseover="this.style.color='#dae2fd'; this.style.background='rgba(45, 52, 73, 0.3)';" onmouseout="this.style.color='#c7c4d8'; this.style.background='transparent';">
                    <span class="material-symbols-outlined">history</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">History</span>
                </a>
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #c7c4d8; text-decoration: none; border-radius: 0 0.5rem 0.5rem 0; transition: all 0.2s ease;" onmouseover="this.style.color='#dae2fd'; this.style.background='rgba(45, 52, 73, 0.3)';" onmouseout="this.style.color='#c7c4d8'; this.style.background='transparent';">
                    <span class="material-symbols-outlined">insights</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Analytics</span>
                </a>
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #c7c4d8; text-decoration: none; border-radius: 0 0.5rem 0.5rem 0; transition: all 0.2s ease;" onmouseover="this.style.color='#dae2fd'; this.style.background='rgba(45, 52, 73, 0.3)';" onmouseout="this.style.color='#c7c4d8'; this.style.background='transparent';">
                    <span class="material-symbols-outlined">settings</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Settings</span>
                </a>
            </nav>

            <div style="margin-top: auto; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1rem;">
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #c7c4d8; text-decoration: none;" onmouseover="this.style.color='#dae2fd';" onmouseout="this.style.color='#c7c4d8';">
                    <span class="material-symbols-outlined">help</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Support</span>
                </a>
                <a href="#" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; color: #c7c4d8; text-decoration: none;" onmouseover="this.style.color='#dae2fd';" onmouseout="this.style.color='#c7c4d8';">
                    <span class="material-symbols-outlined">logout</span>
                    <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Logout</span>
                </a>

                <div style="padding: 1.5rem; display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 2.5rem; height: 2.5rem; border-radius: 50%; background: linear-gradient(135deg, #4f46e5, #03b5d3); display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <span class="material-symbols-outlined" style="color: white; font-size: 1.25rem;">person</span>
                    </div>
                    <div>
                        <p style="font-weight: 700; color: #dae2fd; font-size: 0.875rem; margin: 0;">Alex Rivera</p>
                        <p style="color: #c7c4d8; font-size: 0.625rem; margin: 0; text-transform: uppercase; letter-spacing: 0.025em;">alex.r@summarize.ai</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Header section
st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif; letter-spacing: -0.01em;">Workspace</h1>
        <p style="color: #c7c4d8; font-size: 1rem; margin-top: 0.5rem; font-family: 'Inter', sans-serif;">Paste your email content below to generate a high-precision summary.</p>
    </div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

# Left column - Input section
with col1:
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="font-size: 1.25rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif;">📝 Input Email</h3>
        </div>
    """, unsafe_allow_html=True)

    # Text area for email input
    email_input = st.text_area(
        "Paste your email content here:",
        height=300,
        placeholder="Enter the email text you want to summarize... (e.g. Project updates, client feedback, or long threads)",
        label_visibility="collapsed"
    )

# Right column - Instructions and sample
with col2:
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="font-size: 1.25rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif;">ℹ️ How to Use</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: rgba(23, 31, 51, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span class="material-symbols-outlined" style="color: #4f46e5;">tips_and_updates</span>
                <span style="font-weight: 600; color: #dae2fd; font-size: 0.875rem;">Power Tip</span>
            </div>
            <p style="color: #c7c4d8; font-size: 0.875rem; line-height: 1.4; margin: 0;">Use Command + V to paste instantly and let SummarizeAI auto-detect the context.</p>
        </div>
    """, unsafe_allow_html=True)

    # Show sample email
    if st.checkbox("📌 Show Sample Email"):
        sample_email = """Subject: Urgent: Q2 Project Deadline Update

Hi Team,

I hope this email finds you well. I wanted to reach out regarding the Q2 project status and upcoming deadlines.

As per our discussion, the project needs to be completed by May 31st. We have critical tasks that need immediate attention. Please review the attached requirements and send your feedback by May 15th.

Action items:
1. Please review the project scope document
2. Could you update the timeline spreadsheet?
3. We need to schedule a meeting by next Monday
4. Must confirm budget allocation by end of week

The stakeholders are expecting the presentation on June 1st, so we need to finalize all deliverables as soon as possible.

Let me know if you have any questions or concerns.

Best regards,
John"""
        st.text_area(
            "Sample Email:",
            value=sample_email,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )

# ============================================================================
# ANALYSIS SECTION
# ============================================================================

st.divider()

# Create button for analysis
if st.button("🚀 Analyze Email", use_container_width=True, type="primary"):
    # Validate input
    if not email_input.strip():
        st.warning("⚠️ Please paste an email to analyze")
    elif len(email_input.strip()) < 20:
        st.warning("⚠️ Please enter a longer email (at least 20 characters)")
    else:
        # Show loading spinner while processing
        with st.spinner("🔄 Analyzing email... This may take a moment."):
            # Extract insights
            priority = detect_priority(email_input)
            deadlines = extract_deadlines(email_input)

            # Generate summary
            summary = summarize_email(email_input)

            # Extract action items
            action_items = extract_action_items(email_input, summary)

        # Display results in modern layout
        st.markdown("""
            <div style="margin: 2rem 0;">
                <h2 style="font-size: 1.5rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif; margin-bottom: 1.5rem;">📊 Analysis Results</h2>
            </div>
        """, unsafe_allow_html=True)

        # Create columns for results
        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            # AI Summary Card
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(3, 181, 211, 0.1) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(195, 192, 255, 0.2); border-radius: 1.5rem; padding: 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;">
                    <div style="position: relative; z-index: 10;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                            <span class="material-symbols-outlined" style="color: #4f46e5;">auto_awesome</span>
                            <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem; color: #4f46e5;">AI Synthesis</span>
                        </div>
                        <h3 style="font-size: 1.5rem; font-weight: 600; color: #dae2fd; margin-bottom: 1rem; font-family: 'Inter', sans-serif;">Executive Summary</h3>
                        <p style="font-size: 1.125rem; line-height: 1.6; color: rgba(218, 226, 253, 0.9); font-family: 'Inter', sans-serif;">{summary}</p>
                    </div>
                    <div style="position: absolute; top: -20px; right: -20px; width: 100px; height: 100px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; filter: blur(30px);"></div>
                </div>
            """, unsafe_allow_html=True)

            # Priority Level Card
            priority_color = "#ffb4ab" if "🔴 High" in priority else "#4cd7f6" if "🟡 Medium" in priority else "#c7c4d8"
            priority_icon = "priority_high" if "🔴 High" in priority else "circle" if "🟡 Medium" in priority else "low_priority"
            priority_text = priority.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")

            st.markdown(f"""
                <div style="background: rgba(23, 31, 51, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 1.5rem; padding: 2rem; margin-bottom: 1.5rem;">
                    <h4 style="font-size: 1.25rem; font-weight: 600; color: #dae2fd; margin-bottom: 1rem; font-family: 'Inter', sans-serif;">🎯 Priority Level</h4>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="width: 3rem; height: 3rem; border-radius: 50%; background: rgba({priority_color.replace('#', '').replace('ffb4ab', '255, 180, 171').replace('4cd7f6', '76, 215, 246').replace('c7c4d8', '199, 196, 216')}, 0.1); display: flex; align-items: center; justify-content: center; border: 1px solid {priority_color};">
                            <span class="material-symbols-outlined" style="color: {priority_color}; font-size: 1.5rem;">{priority_icon}</span>
                        </div>
                        <div>
                            <span style="font-size: 2rem; font-weight: 700; color: {priority_color};">{priority_text}</span>
                            <p style="font-size: 0.875rem; color: #c7c4d8; margin: 0.25rem 0 0 0;">Action needed within 4 hours</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with result_col2:
            # Action Items Card
            st.markdown("""
                <div style="background: rgba(23, 31, 51, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 1.5rem; padding: 2rem; margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-between; align-items: start; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <div style="width: 2.5rem; height: 2.5rem; border-radius: 0.5rem; background: rgba(255, 180, 171, 0.1); display: flex; align-items: center; justify-content: center;">
                                <span class="material-symbols-outlined" style="color: #ffb4ab; font-size: 1.25rem;">warning</span>
                            </div>
                            <h4 style="font-size: 1.25rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif;">Critical Tasks</h4>
                        </div>
                        <button style="color: #4f46e5; padding: 0.5rem; border-radius: 0.375rem; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(79, 70, 229, 0.1)'" onmouseout="this.style.background='transparent'">
                            <span class="material-symbols-outlined" style="font-size: 1.25rem;">content_copy</span>
                        </button>
                    </div>
            """, unsafe_allow_html=True)

            if action_items and action_items[0] != "No specific action items identified":
                for item in action_items:
                    if item.startswith("• "):
                        clean_item = item[2:]  # Remove the bullet
                        st.markdown(f"""
                            <div style="display: flex; align-items: start; gap: 0.75rem; margin-bottom: 0.75rem; color: #c7c4d8;">
                                <span class="material-symbols-outlined" style="color: #ffb4ab; font-size: 0.875rem; margin-top: 0.375rem;">priority_high</span>
                                <span style="line-height: 1.4;">{clean_item}</span>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="color: #c7c4d8; font-style: italic;">
                        No specific action items identified
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Deadlines Card
            st.markdown("""
                <div style="background: rgba(23, 31, 51, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 1.5rem; padding: 2rem;">
                    <div style="display: flex; justify-between; align-items: start; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <div style="width: 2.5rem; height: 2.5rem; border-radius: 0.5rem; background: rgba(76, 215, 246, 0.1); display: flex; align-items: center; justify-content: center;">
                                <span class="material-symbols-outlined" style="color: #4cd7f6; font-size: 1.25rem;">event</span>
                            </div>
                            <h4 style="font-size: 1.25rem; font-weight: 600; color: #dae2fd; font-family: 'Inter', sans-serif;">Upcoming Deadlines</h4>
                        </div>
                        <button style="color: #4cd7f6; padding: 0.5rem; border-radius: 0.375rem; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(76, 215, 246, 0.1)'" onmouseout="this.style.background='transparent'">
                            <span class="material-symbols-outlined" style="font-size: 1.25rem;">calendar_add_on</span>
                        </button>
                    </div>
            """, unsafe_allow_html=True)

            if deadlines and deadlines[0] != "No specific deadline found":
                for deadline in deadlines:
                    st.markdown(f"""
                        <div style="margin-bottom: 0.75rem;">
                            <span style="font-weight: 600; color: #dae2fd; font-size: 0.875rem;">{deadline}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="color: #c7c4d8; font-style: italic;">
                        No specific deadline found
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Add divider before additional options
        st.divider()

        # Copy to clipboard functionality
        st.success("✅ Analysis complete! You can copy the results below.")

        # Create a formatted output for easy copying
        formatted_output = f"""
EMAIL ANALYSIS REPORT
{'='*50}

SUMMARY:
{summary}

PRIORITY LEVEL:
{priority_text}

ACTION ITEMS:
{chr(10).join([item[2:] if item.startswith('• ') else item for item in action_items])}

DEADLINES & DATES:
{chr(10).join(deadlines)}
"""

        st.text_area(
            "Formatted Results (Copy & Paste):",
            value=formatted_output,
            height=250,
            disabled=True,
            label_visibility="collapsed"
        )

# ============================================================================
# FOOTER AND ADDITIONAL INFO
# ============================================================================

st.divider()

st.markdown("""
    <div style="background: rgba(23, 31, 51, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 1rem; padding: 1.5rem; margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span class="material-symbols-outlined" style="color: #4f46e5;">tips_and_updates</span>
            <span style="font-weight: 600; color: #dae2fd; font-size: 0.875rem;">Tips for Best Results</span>
        </div>
        <ul style="color: #c7c4d8; font-size: 0.875rem; line-height: 1.6; margin: 0; padding-left: 1.5rem;">
            <li>Use complete sentences for better analysis</li>
            <li>Include dates in standard formats (MM/DD/YYYY or Month DD, YYYY)</li>
            <li>Use clear action verbs (please, should, need to, etc.)</li>
            <li>The model works best with emails longer than 50 words</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #c7c4d8; font-size: 0.75rem; margin-top: 2rem;">
        <p style="margin: 0; font-family: 'Inter', sans-serif;">SummarizeAI v1.0 | Powered by Hugging Face Transformers & Streamlit</p>
        <p style="margin: 0.5rem 0 0 0; font-family: 'Inter', sans-serif;">For bug reports or feature requests, please visit the project repository</p>
    </div>
""", unsafe_allow_html=True)

# Add ambient glow effects
st.markdown("""
    <div class="ambient-glow-1"></div>
    <div class="ambient-glow-2"></div>
""", unsafe_allow_html=True)
