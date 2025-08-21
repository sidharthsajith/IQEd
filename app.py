import json
import sys
import re
import logging
import csv
import streamlit as st
import pandas as pd
from groq import Groq
from pydantic import BaseModel, validator
from typing import Literal, List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Helpers (placed early so they are defined before use) ----
def clean_json_string(json_str: str) -> str:
    """Clean and fix common JSON formatting issues in the API response."""
    if not json_str:
        return '[]'
    # Remove any code block markers if present
    json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_str, flags=re.MULTILINE)
    # Remove any leading/trailing whitespace and quotes
    json_str = json_str.strip().strip("'")
    # Try to parse as is first
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    # Try various fixes for common issues
    for _ in range(2):
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<![\\])(")(?=(?:[^"]*"[^"]*")*[^"]*$)', r'\\"', json_str)
        # Fix single quotes around keys and values
        json_str = re.sub(r"'([^']+)"r"'\s*:", r'"\1":', json_str)
        json_str = re.sub(r':\s*\'([^\']+)\'\s*([,\}])', r':"\1"\2', json_str)
        # Fix trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Try parsing again
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    # If we get here, try to extract array content
    array_match = re.search(r'\[\s*\{.*\}\s*\]', json_str, re.DOTALL)
    if array_match:
        return array_match.group(0)
    # Last resort: try to find any valid JSON object/array
    try:
        obj_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', json_str, re.DOTALL)
        if obj_match:
            return f'[{obj_match.group(0)}]'
    except Exception:
        pass
    return '[]'

def sanitize_plain(text: Any) -> str:
    """Convert model text to plain, non-Markdown, non-LaTeX form suitable for CSV.

    - Remove code fences/backticks
    - Strip LaTeX delimiters $...$, $$...$$, \(...\), \[...\]
    - Replace common LaTeX math commands with plain/Unicode
    - Remove \left, \right, \\, \; etc.
    - Collapse whitespace
    """
    s = str(text or "")
    # Remove code fences/backticks
    s = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip('`'), s)
    s = s.replace("`", "")
    # Strip inline/block LaTeX delimiters keeping inner text
    s = re.sub(r"\$\$(.*?)\$\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\$(.*?)\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"\1", s, flags=re.DOTALL)
    # Common replacements
    replacements = {
        r"\\pi": "π",
        r"\\int": "∫",
        r"\\cdot": "·",
        r"\\times": "×",
        r"\\frac": "/",  # crude but avoids TeX
        r"\\,": " ",
        r"\\;": " ",
        r"\\!": "",
        r"\\left": "",
        r"\\right": "",
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
    }
    for pat, rep in replacements.items():
        s = re.sub(pat, rep, s)
    # Remove remaining backslashes from commands
    s = s.replace("\\", "")
    # Remove braces around simple tokens
    s = re.sub(r"\{\s*([^{}]+?)\s*\}", r"\1", s)
    # Normalize some phrasing to sample-like style
    s = s.replace("dx", "dx")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_questions(api_response: str) -> List[Dict[str, Any]]:
    """Parse and validate questions from the API response."""
    try:
        cleaned_json = clean_json_string(api_response)
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError:
            array_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_json, re.DOTALL)
            if array_match:
                data = json.loads(array_match.group(0))
            else:
                return []
        if isinstance(data, dict):
            if 'questions' in data:
                questions = data['questions']
            elif 'value' in data:
                questions = data['value']
            else:
                questions = [v for v in data.values() if isinstance(v, list)][0]
        else:
            questions = data
        if not isinstance(questions, list):
            questions = [questions]
        return questions
    except Exception:
        return []

def validate_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        required = ['topic','difficulty','question','option1','option2','option3','option4','answer','explanation']
        if any(f not in q for f in required):
            continue
        cleaned = {k: str(v).strip() for k, v in q.items()}
        if cleaned.get('difficulty') not in ['easy','medium','hard']:
            cleaned['difficulty'] = 'easy'
        if cleaned.get('answer') not in [cleaned.get(f'option{i}', '') for i in range(1,5)]:
            continue
        validated.append(cleaned)
    return validated

def generate_questions(*, client: Groq, context: Any, num_questions: int, difficulty: str, model: str) -> List[Dict[str, Any]]:
    sys_prompt = (
        "You are an assessment generator. Your output MUST be a RAW JSON array and nothing else. "
        "Do NOT use Markdown, backticks, code fences, or prose. Do NOT prefix or suffix any text. "
        "Do NOT use LaTeX/TeX or math delimiters (e.g., $, $$, \\(...\\), \\[...\\]). "
        "Write equations in simple plain text/Unicode (e.g., '∫ 0 to π sin3x cos2x dx'). "
        "Schema for each array element: {"
        "topic: string, "
        "difficulty: one of ['easy','medium','hard'], "
        "question: string, "
        "option1: string, option2: string, option3: string, option4: string, "
        "answer: string (must be exactly equal to one of option1..option4), "
        "explanation: string}. "
        "Ensure valid JSON with no trailing commas. If unable to comply, return []."
    )
    user_prompt = (
        f"Create exactly {num_questions} {difficulty} multiple-choice questions based on the provided data.\n"
        f"Be faithful to the concepts; vary phrasing and values.\n"
        f"OUTPUT REQUIREMENTS: Return ONLY a raw JSON array that matches the schema described by the system.\n"
        f"Do NOT include markdown, backticks, code blocks, LaTeX/TeX, or any explanation text. Do NOT wrap the JSON in quotes.\n\n"
        f"Context (JSON records):\n{json.dumps(context)[:6000]}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    content = resp.choices[0].message.content if resp.choices else "[]"
    raw_items = parse_questions(content)
    for item in raw_items:
        item.setdefault("difficulty", difficulty)
    validated = validate_questions(raw_items)
    if len(validated) > num_questions:
        validated = validated[:num_questions]
    return validated

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = []
    st.session_state.uploaded_file = None

# Configure page
st.set_page_config(
    page_title="AI Question Generator",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

## Header
st.title("❓ AI Question Generator")
st.caption("Upload CSV, control difficulty distribution, and download validated questions.")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("This app uses Groq's open-source model via a fixed configuration.")
    st.info("Model: openai/gpt-oss-120b (fixed)")
    # Ensure API key exists in secrets
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        st.error("GROQ_API_KEY missing in .streamlit/secrets.toml")
    

# Main UI intro
st.markdown(
    "Generate high-quality multiple-choice questions from your content. "
    "Customize the difficulty distribution below."
)

# File uploader (CSV or Excel)
uploaded_file = st.file_uploader(
    "📁 Upload data file (CSV or Excel)", 
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel file containing the content for question generation"
)

# Initialize session state for file upload
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.get('uploaded_file'):
    try:
        # Detect and read CSV/Excel into DataFrame
        name = st.session_state.uploaded_file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(st.session_state.uploaded_file)
            uploaded_kind = 'csv'
        else:
            # Excel handling: allow sheet selection
            xls = pd.ExcelFile(st.session_state.uploaded_file)
            sheet = st.selectbox("Select Excel sheet", options=xls.sheet_names)
            df = xls.parse(sheet)
            uploaded_kind = 'excel'
        # Normalize column names (e.g., 'option1 ' -> 'option1', 'option 2' -> 'option2')
        df.columns = [re.sub(r"\s+", "", str(c)).strip() for c in df.columns]
        
        # Display file info and preview
        st.success(f"✅ Successfully loaded {len(df)} rows from {st.session_state.uploaded_file.name}")
        
        with st.expander("🔍 View uploaded data", expanded=False):
            st.dataframe(df.head())
            st.caption(f"Total rows: {len(df)}")
            if uploaded_kind == 'excel':
                csv_preview = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download uploaded sheet as CSV",
                    data=csv_preview,
                    file_name=(Path(st.session_state.uploaded_file.name).stem + f"_{sheet}.csv"),
                    mime="text/csv"
                )
        
        # Context records
        records = df.to_dict(orient='records')
        
        # Distribution mode: per-topic or global
        st.subheader("📊 Distribution Mode")
        mode = st.radio("How would you like to specify counts?", ["Global", "Per-topic"], horizontal=True, index=0)

        counts: Dict[str, Dict[str, int]] = {}
        topic_col = None
        total_requested = 0

        if mode == "Per-topic":
            # Topic column selection
            st.subheader("🏷️ Topic Configuration")
            possible_cols = list(df.columns)
            topic_col = st.selectbox(
                "Select the topic column",
                options=possible_cols,
                help="Choose the column that represents the topic/category for each row"
            )
            unique_topics = [t for t in sorted(df[topic_col].dropna().astype(str).unique())]
            st.caption(f"Detected {len(unique_topics)} topics from column '{topic_col}'.")

            st.subheader("Per-topic difficulty counts")
            cols = st.columns(4)
            cols[0].markdown("**Topic**")
            cols[1].markdown("**Easy**")
            cols[2].markdown("**Medium**")
            cols[3].markdown("**Hard**")
            for topic in unique_topics:
                c1, c2, c3, c4 = st.columns(4)
                c1.write(topic)
                easy_n = c2.number_input(f"easy_{topic}", min_value=0, max_value=200, value=0, step=1, label_visibility="collapsed")
                med_n = c3.number_input(f"med_{topic}", min_value=0, max_value=200, value=0, step=1, label_visibility="collapsed")
                hard_n = c4.number_input(f"hard_{topic}", min_value=0, max_value=200, value=0, step=1, label_visibility="collapsed")
                counts[topic] = {"easy": easy_n, "medium": med_n, "hard": hard_n}
            total_requested = int(sum(sum(levels.values()) for levels in counts.values()))
            st.caption(f"Total requested questions: {total_requested}")
        else:
            st.subheader("Global difficulty counts")
            col1, col2, col3 = st.columns(3)
            easy_n = col1.number_input("Easy", min_value=0, max_value=1000, value=0, step=1)
            med_n = col2.number_input("Medium", min_value=0, max_value=1000, value=0, step=1)
            hard_n = col3.number_input("Hard", min_value=0, max_value=1000, value=0, step=1)
            counts = {"__GLOBAL__": {"easy": int(easy_n), "medium": int(med_n), "hard": int(hard_n)}}
            total_requested = int(easy_n + med_n + hard_n)
            st.caption(f"Total requested questions: {total_requested}")
        
        # Generate questions button
        if st.button("✨ Generate Questions", disabled=("GROQ_API_KEY" not in st.secrets or total_requested <= 0)):
            if "GROQ_API_KEY" not in st.secrets:
                st.error("Please configure GROQ_API_KEY in .streamlit/secrets.toml")
                st.stop()
                
            with st.spinner('Generating questions...'):
                try:
                    # Initialize Groq client with user's API key
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

                    MODEL = "openai/gpt-oss-120b"
                    all_questions: List[Dict[str, Any]] = []

                    for topic, level_counts in counts.items():
                        if mode == "Per-topic":
                            topic_context = [r for r in records if str(r.get(topic_col, "")) == str(topic)]
                            effective_context = topic_context if topic_context else records
                        else:
                            effective_context = records
                        for level, n in level_counts.items():
                            n = int(n)
                            if n <= 0:
                                continue
                            with st.status(f"Generating {n} {level} questions for topic '{topic}'..."):
                                qs = generate_questions(
                                    client=client,
                                    context={
                                        "topic": topic,
                                        "topic_column": topic_col,
                                        "records": effective_context,
                                    },
                                    num_questions=n,
                                    difficulty=level,
                                    model=MODEL,
                                )
                                # Ensure topic is set in each question
                                for q in qs:
                                    q.setdefault("topic", topic if mode == "Per-topic" else "")
                                all_questions.extend(qs)

                    # Store questions in session state
                    st.session_state.questions = all_questions
                    st.success(f"✅ Successfully generated {len(all_questions)} questions!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    logger.exception("Error generating questions")
        
        # Display and download questions if available
        if st.session_state.questions:
            st.subheader("📋 Generated Questions")
            
            # Convert to DataFrame for display and download
            try:
                questions_df = pd.DataFrame(st.session_state.questions)
                
                # Display statistics
                difficulty_counts = questions_df['difficulty'].value_counts().to_dict()
                stats_cols = st.columns(len(difficulty_counts) + 1)
                
                for i, (diff, count) in enumerate(difficulty_counts.items()):
                    stats_cols[i].metric(
                        f"{diff.capitalize()} Questions",
                        count,
                        f"{count/len(questions_df)*100:.0f}%"
                    )
                
                # Show data table with expandable rows for long text
                with st.expander("View All Questions", expanded=True):
                    # Format long text columns for better display
                    display_df = questions_df.copy()
                    for col in ['question', 'explanation']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].str.wrap(50).apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        column_config={
                            "question": "Question",
                            "difficulty": st.column_config.SelectboxColumn(
                                "Difficulty",
                                options=["easy", "medium", "hard"],
                                required=True
                            ),
                            "explanation": "Explanation"
                        },
                        hide_index=True
                    )
                
                # Download section
                st.divider()
                st.subheader("💾 Download Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV Download
                    csv = questions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="generated_questions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Excel Download
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        questions_df.to_excel(writer, index=False, sheet_name="questions")
                    st.download_button(
                        label="Download as Excel (.xlsx)",
                        data=output.getvalue(),
                        file_name="generated_questions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with col3:
                    # JSON Download (kept as supplementary)
                    json_data = questions_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=json_data,
                        file_name="generated_questions.json",
                        mime="application/json",
                        use_container_width=True
                    )

                # Sample-format CSV (matches uploaded sample headers, sanitized to plain text)
                st.markdown("\n")
                sample_df = pd.DataFrame({
                    "question": questions_df.get("question", pd.Series(dtype=str)).apply(sanitize_plain),
                    "option1 ": questions_df.get("option1", pd.Series(dtype=str)).apply(sanitize_plain),
                    "option 2": questions_df.get("option2", pd.Series(dtype=str)).apply(sanitize_plain),
                    "option 3": questions_df.get("option3", pd.Series(dtype=str)).apply(sanitize_plain),
                    "option 4": questions_df.get("option4", pd.Series(dtype=str)).apply(sanitize_plain),
                    "correct answer": questions_df.get("answer", pd.Series(dtype=str)).apply(sanitize_plain),
                })
                sample_csv = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as Sample CSV (plain text, no Markdown)",
                    data=sample_csv,
                    file_name="generated_questions (sample).csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Show sample data for debugging
                if st.toggle("Show raw data"):
                    st.json(questions_df.to_dict(orient='records'))
                    
            except Exception as e:
                st.error(f"Error displaying questions: {str(e)}")
                logger.exception("Error displaying questions")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
else:
    st.info("ℹ️ Please upload a CSV file to get started")

# Define question model
class QuestionModel(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    option1: str
    option2: str
    option3: str
    option4: str
    answer: str
    explanation: str

def clean_json_string(json_str: str) -> str:
    """Clean and fix common JSON formatting issues in the API response."""
    if not json_str:
        return '[]'
        
    # Remove any code block markers if present
    json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_str, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace and quotes
    json_str = json_str.strip().strip("'")
    
    # Try to parse as is first
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass  # Continue with fixes
    
    # Try to parse as is first
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass  # Continue with fixes
    
    # Try various fixes for common issues
    for _ in range(2):  # Try a couple of passes
        # Fix unescaped quotes in strings
        json_str = re.sub(
            r'(?<![\\])(")(?=(?:[^"]*"[^"]*")*[^"]*$)', 
            r'\\"', 
            json_str
        )
        
        # Fix single quotes around keys and values
        json_str = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)  # Keys
        # Fix single quotes around values
        json_str = re.sub(r':\s*\'([^\']+)\'\s*([,\}])', r':"\1"\2', json_str)  # Values
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Try parsing again
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    
    # If we get here, try to extract array content
    array_match = re.search(r'\[\s*\{.*\}\s*\]', json_str, re.DOTALL)
    if array_match:
        return array_match.group(0)
    
    # Last resort: try to find any valid JSON object/array
    try:
        # Try to find the first valid JSON object
        obj_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', json_str, re.DOTALL)
        if obj_match:
            return f'[{obj_match.group(0)}]'
    except Exception:
        pass
    
    print("Warning: Could not parse JSON response. Using empty array.")
    return '[]'

def generate_questions(*, client: Groq, context: Any, num_questions: int, difficulty: str, model: str) -> List[Dict[str, Any]]:
    """Call Groq to generate MCQs and return validated list of question dicts.

    Ensures strictly JSON array output with the requested difficulty.
    """
    sys_prompt = (
        "You are an assessment generator. Your output MUST be a RAW JSON array and nothing else. "
        "Do NOT use Markdown, backticks, code fences, or prose. Do NOT prefix or suffix any text. "
        "Do NOT use LaTeX/TeX or math delimiters (e.g., $, $$, \\(...\\), \\[...\\]). "
        "Write equations in simple plain text/Unicode (e.g., '∫ 0 to π sin3x cos2x dx'). "
        "Schema for each array element: {"
        "topic: string, "
        "difficulty: one of ['easy','medium','hard'], "
        "question: string, "
        "option1: string, option2: string, option3: string, option4: string, "
        "answer: string (must be exactly equal to one of option1..option4), "
        "explanation: string}. "
        "Ensure valid JSON with no trailing commas. If unable to comply, return []."
    )

    user_prompt = (
        f"Create exactly {num_questions} {difficulty} multiple-choice questions based on the provided data.\n"
        f"Be faithful to the concepts; vary phrasing and values.\n"
        f"OUTPUT REQUIREMENTS: Return ONLY a raw JSON array that matches the schema described by the system.\n"
        f"Do NOT include markdown, backticks, code blocks, LaTeX/TeX, or any explanation text. Do NOT wrap the JSON in quotes.\n\n"
        f"Context (JSON records):\n{json.dumps(context)[:6000]}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = resp.choices[0].message.content if resp.choices else "[]"
    except Exception as e:
        logger.exception("Groq API call failed")
        raise

    # Parse and validate
    raw_items = parse_questions(content)
    # Enforce difficulty tag
    for item in raw_items:
        item.setdefault("difficulty", difficulty)
    validated = validate_questions(raw_items)
    # Enforce exact count
    if len(validated) > num_questions:
        validated = validated[:num_questions]
    return validated

def parse_questions(api_response: str) -> List[Dict[str, Any]]:
    """Parse and validate questions from the API response."""
    print("\nProcessing API response...")
    
    # Clean and parse the response
    try:
        # Clean the JSON string
        cleaned_json = clean_json_string(api_response)
        
        # Parse the JSON
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print("Attempting to extract JSON array...")
            # Try to extract array content if wrapped in text
            array_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_json, re.DOTALL)
            if array_match:
                data = json.loads(array_match.group(0))
            else:
                print("Could not extract valid JSON array from response.")
                return []
        
        # Extract questions from response
        if isinstance(data, dict):
            # Try common response formats
            if 'questions' in data:
                questions = data['questions']
            elif 'value' in data:
                questions = data['value']
            else:
                # Assume any list value contains the questions
                questions = [v for v in data.values() if isinstance(v, list)][0]
        else:
            questions = data
        
        # Ensure we have a list
        if not isinstance(questions, list):
            questions = [questions]
            
        print(f"Found {len(questions)} questions in response")
        return questions
        
    except Exception as e:
        print(f"Error parsing questions: {str(e)}")
        print(f"Response content (first 500 chars): {api_response[:500]}...")
        return []

def validate_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean question data."""
    validated = []
    errors = []
    
    for i, q in enumerate(questions, 1):
        try:
            if not isinstance(q, dict):
                errors.append(f"Question {i}: Not a dictionary")
                continue
                
            # Ensure all required fields are present
            required_fields = [
                'topic', 'difficulty', 'question', 
                'option1', 'option2', 'option3', 'option4', 
                'answer', 'explanation'
            ]
            
            missing = [f for f in required_fields if f not in q]
            if missing:
                errors.append(f"Question {i}: Missing fields: {', '.join(missing)}")
                continue
                
            # Clean and validate fields
            cleaned = {k: str(v).strip() for k, v in q.items()}
            
            # Validate difficulty
            if cleaned['difficulty'] not in ['easy', 'medium', 'hard']:
                cleaned['difficulty'] = 'easy'  # Default to easy
                
            # Ensure answer is one of the options
            if cleaned['answer'] not in [cleaned[f'option{i}'] for i in range(1, 5)]:
                errors.append(f"Question {i}: Answer must match one of the options")
                continue
                
            validated.append(cleaned)
            
        except Exception as e:
            errors.append(f"Error processing question {i}: {str(e)}")
    
    # Report validation errors
    if errors:
        print(f"\nValidation issues ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    return validated

def write_questions_to_csv(questions: List[Dict[str, str]], filename: str = 'output_questions.csv') -> bool:
    """Write questions to a CSV file."""
    if not questions:
        print("No valid questions to write to CSV")
        return False
        
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'topic', 'difficulty', 'question',
                'option1', 'option2', 'option3', 'option4',
                'answer', 'explanation'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(questions)
            
        print(f"\nSuccessfully wrote {len(questions)} questions to {filename}")
        return True
        
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
        return False
