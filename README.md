# 📧 Smart Email Summarizer

An AI-powered web application that analyzes and summarizes emails using cutting-edge natural language processing. Built with **Streamlit** and **Hugging Face Transformers**, this application helps you quickly understand email content, identify action items, and prioritize responses.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **AI-Powered Email Summarization**: Uses the `facebook/bart-large-cnn` model to generate concise summaries of long emails
- **Action Item Extraction**: Automatically identifies tasks and action items from email content
- **Deadline Detection**: Extracts dates, deadlines, and time references using advanced regex patterns
- **Priority Classification**: Analyzes email content to determine priority level (High/Medium/Low)
- **Modern UI**: Clean, responsive, and intuitive user interface
- **Real-time Processing**: Get instant analysis with visual feedback
- **Beginner-Friendly Code**: Well-documented code with extensive comments for learning
- **Ready for Deployment**: Can be easily deployed on Streamlit Cloud

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python-based web framework)
- **NLP Model**: Hugging Face Transformers (facebook/bart-large-cnn)
- **Deep Learning**: PyTorch
- **Text Processing**: NLTK, RegEx
- **Language**: Python 3.8+

---

## 📋 Project Structure

```
smart-email-summarizer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python package dependencies
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~2GB disk space (for downloading the model on first run)

### Installation Steps

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/yourusername/smart-email-summarizer.git
   cd smart-email-summarizer
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The first run will download the BART model (~1.6GB), which may take a few minutes.

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

   The application will open in your default browser at `http://localhost:8501`

---

## 💡 How to Use

### Basic Usage

1. **Open the Application**: Navigate to `http://localhost:8501` in your web browser

2. **Paste Your Email**: 
   - Copy an email from your email client
   - Paste it into the "Paste your email content here" text area
   - Click the "🚀 Analyze Email" button

3. **Review Results**:
   - **Summary**: Concise summary of the email
   - **Priority Level**: Quick priority indicator (🔴 High, 🟡 Medium, 🟢 Low)
   - **Action Items**: Tasks you need to complete
   - **Deadlines & Dates**: Important dates and deadlines mentioned

4. **Copy Results**: Use the formatted results box to easily copy and share the analysis

### Sample Test Email

```
Subject: Urgent: Q2 Project Deadline Update

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
John
```

**Expected Output**:
- **Summary**: Project needs completion by May 31st with critical tasks requiring immediate review and feedback by May 15th
- **Priority**: 🔴 High (due to "urgent" and "immediate attention")
- **Action Items**: 
  - Review project scope document
  - Update timeline spreadsheet
  - Schedule meeting
  - Confirm budget allocation
- **Deadlines**: May 15th, May 31st, June 1st, next Monday, end of week

---

## 🔧 Configuration & Customization

### Adjusting Summarization Parameters

Edit the `summarize_email()` function in `app.py` to customize:

```python
summary = summarizer(
    truncated_text,
    max_length=150,      # Maximum length of summary (tokens)
    min_length=50,       # Minimum length of summary (tokens)
    do_sample=False      # Set to True for more creative summaries
)
```

### Changing Priority Keywords

Modify the `detect_priority()` function to add or remove priority keywords:

```python
high_priority_keywords = [
    'urgent', 'asap', 'immediately', 'critical',
    # Add more keywords here
]
```

### Customizing Deadline Patterns

Add regex patterns in `extract_deadlines()` for different date formats:

```python
patterns = [
    r'your_regex_pattern_here',
    # Add more patterns
]
```

---

## 📊 How It Works

### 1. **Email Input**
   - User pastes email text into the web interface

### 2. **Text Processing**
   - Email text is validated for length and content
   - Text is prepared for model input

### 3. **Summarization**
   - Uses the BART (Bidirectional and Auto-Regressive Transformers) model
   - Generates an abstractive summary (not just extractive)
   - Model is efficient and runs on both CPU and GPU

### 4. **Analysis**
   - **Priority Detection**: Keyword-based analysis
   - **Deadline Extraction**: Regex-based pattern matching
   - **Action Item Detection**: Sentence analysis with keyword recognition

### 5. **Result Display**
   - Results are presented in a clean, organized format
   - Can be copied for use in other applications

---

## ⚙️ Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit==1.28.1          # Web framework
transformers==4.34.1       # NLP models
torch==2.0.1              # Deep learning framework
sentencepiece==0.1.99     # Tokenization library
nltk==3.8.1               # Natural Language Toolkit
```

---

## 🐛 Troubleshooting

### Issue: "Model not found" or download takes too long
**Solution**: The first run downloads the BART model (~1.6GB). This is normal and happens only once. Ensure you have good internet connectivity.

### Issue: Out of memory error
**Solution**: The model requires ~4GB RAM. If you have less:
- Close other applications
- Try with shorter emails (less than 1000 words)
- Consider using a GPU if available

### Issue: Application won't start
**Solution**: 
1. Ensure Python 3.8+ is installed: `python --version`
2. Verify all packages installed: `pip list`
3. Try reinstalling requirements: `pip install --upgrade -r requirements.txt`

### Issue: Summary is too short or too long
**Solution**: Adjust the `max_length` and `min_length` parameters in the `summarize_email()` function

---

## 🌟 Future Improvements

- **Multi-language Support**: Add support for emails in different languages
- **Sentiment Analysis**: Detect emotional tone and sentiment of emails
- **Recipient Suggestions**: Auto-suggest who should receive the reply
- **Email Category Classification**: Automatically categorize emails (meeting, task, report, etc.)
- **Custom Model Selection**: Allow users to choose from different summarization models
- **Email Attachment Analysis**: Extract text from attachments and include in analysis
- **Database Integration**: Save analysis history and generate reports
- **Email Integration**: Direct integration with Gmail, Outlook, etc.
- **Advanced NLP**: Named entity recognition for people, organizations, locations
- **Performance Optimization**: Implement model quantization for faster processing
- **Dark Mode**: Add dark theme option
- **Export Options**: PDF, Word, Excel export formats
- **Batch Processing**: Analyze multiple emails at once
- **Smart Routing**: Suggest which team member should handle the email

---

## 📝 Code Documentation

The codebase is extensively commented for beginners:

- **Helper Functions**: Each function has docstrings explaining purpose and parameters
- **Inline Comments**: Complex logic is explained with inline comments
- **Streamlit Sections**: Clearly marked sections for different UI components

### Main Functions

#### `detect_priority(email_text)`
Analyzes keywords to determine email priority level

#### `extract_action_items(email_text, summary_text)`
Identifies tasks and action items from content

#### `extract_deadlines(email_text)`
Finds dates, deadlines, and time references using regex

#### `summarize_email(email_text)`
Generates a concise summary using BART model

---

## 🚀 Deployment

### Deploy on Streamlit Cloud (Recommended)

1. Push your repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository and main file (`app.py`)
5. Click "Deploy"

> The repository includes `requirements.txt` and `streamlit.toml` for smooth deployment.

### Automatic updates on GitHub + Streamlit

Once your repo is connected to Streamlit Cloud, any changes pushed to `main` will automatically trigger a redeploy on Streamlit Cloud.

This repo also includes a GitHub Actions workflow at `.github/workflows/python-ci.yml` that runs on every push and pull request to `main`.

### Deploy on Other Platforms

**Heroku** or **Google Cloud Run**: Create a `Procfile`:
```
web: streamlit run app.py --logger.level=error
```

**Docker**: Create a `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with:
- Bug fixes
- New features
- Code improvements
- Documentation updates

---

## 💬 Support & Questions

- **Report Issues**: Open an issue on GitHub
- **Discussions**: Start a discussion for questions and ideas
- **Documentation**: Check this README for common questions

---

## 🎓 Learning Resources

This project is designed to help beginners learn:
- **Streamlit**: Building interactive web applications with Python
- **Transformers**: Using pre-trained NLP models
- **Natural Language Processing**: Text analysis and processing
- **Regular Expressions**: Pattern matching in text
- **Web Development**: Creating responsive user interfaces

### Recommended Learning Path
1. Read through `app.py` comments
2. Modify parameters and test different inputs
3. Add new helper functions
4. Implement additional features
5. Deploy the application

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BART Model Card](https://huggingface.co/facebook/bart-large-cnn)
- [NLTK Documentation](https://www.nltk.org/)
- [Python Regex Guide](https://docs.python.org/3/library/re.html)

---

## 🎯 Version History

- **v1.0** (Initial Release)
  - Email summarization
  - Action item extraction
  - Deadline detection
  - Priority classification
  - Modern UI

---

**Created with ❤️ for learning and productivity**

Last Updated: May 2026
