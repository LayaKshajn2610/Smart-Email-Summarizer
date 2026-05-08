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
    page_title="Smart Email Summarizer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for modern styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .email-input {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        .result-box {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 5px;
        }
        .info-box {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Header section
st.markdown("""
    <div class="header">
        <h1>📧 Smart Email Summarizer</h1>
        <p>AI-powered email analysis and summarization powered by Hugging Face Transformers</p>
    </div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

# Left column - Input section
with col1:
    st.subheader("📝 Input Email")
    
    # Text area for email input
    email_input = st.text_area(
        "Paste your email content here:",
        height=250,
        placeholder="Enter the email text you want to summarize...",
        label_visibility="collapsed"
    )

# Right column - Instructions and sample
with col2:
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    **Steps:**
    1. Paste your email text in the input area
    2. Click "Analyze Email" button
    3. Get instant insights:
       - Summary of key points
       - Action items to complete
       - Deadlines and dates
       - Priority level
    
    **Example email format:**
    - Business communications
    - Meeting notes
    - Project updates
    - Task assignments
    """)
    
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
            height=150,
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
        
        # Display results
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        
        # Create columns for results
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.subheader("📋 Summary")
            st.write(summary)
            
            st.subheader("🎯 Priority Level")
            st.markdown(f"### {priority}")
        
        with result_col2:
            st.subheader("✅ Action Items")
            for item in action_items:
                st.markdown(item)
            
            st.subheader("📅 Deadlines & Dates")
            for deadline in deadlines:
                st.markdown(f"• {deadline}")
        
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
{priority}

ACTION ITEMS:
{chr(10).join(action_items)}

DEADLINES & DATES:
{chr(10).join(deadlines)}
"""
        
        st.text_area(
            "Formatted Results (Copy & Paste):",
            value=formatted_output,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )

# ============================================================================
# FOOTER AND ADDITIONAL INFO
# ============================================================================

st.divider()

st.markdown("""
    <div class="info-box">
        <h4>💡 Tips for Best Results:</h4>
        <ul>
            <li>Use complete sentences for better analysis</li>
            <li>Include dates in standard formats (MM/DD/YYYY or Month DD, YYYY)</li>
            <li>Use clear action verbs (please, should, need to, etc.)</li>
            <li>The model works best with emails longer than 50 words</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        <p>Smart Email Summarizer v1.0 | Powered by Hugging Face Transformers & Streamlit</p>
        <p>For bug reports or feature requests, please visit the project repository</p>
    </div>
""", unsafe_allow_html=True)
