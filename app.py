# =============================================================
# Edu-Query: AI-Powered Student Enquiry Chatbot
# Backend: Flask + BeautifulSoup + Anthropic Claude API
# =============================================================

import os
import re
import json
import requests
from flask import Flask, render_template, request, jsonify, session
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# Initialize Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "edu_query_default_dev_key")

# ------------------------------------------------------------------
# Groq API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ------------------------------------------------------------------
# CATEGORY KEYWORD MAP
# Used for simple intent classification without an external library
# ------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "admissions": [
        "admission", "apply", "application", "eligibility", "cutoff",
        "merit", "entrance", "enroll", "enrollment", "join", "intake",
        "selection", "criteria", "requirement", "document", "registration"
    ],
    "fees": [
        "fee", "fees", "cost", "tuition", "scholarship", "financial",
        "payment", "installment", "hostel fee", "lab fee", "charges",
        "afford", "price", "expensive", "concession", "stipend"
    ],
    "exams": [
        "exam", "examination", "test", "schedule", "timetable", "result",
        "mark", "grade", "semester", "assessment", "hall ticket", "arrear",
        "revaluation", "supplementary", "internal", "external", "syllabus"
    ],
    "campus": [
        "campus", "facility", "facilities", "hostel", "library", "lab",
        "laboratory", "cafeteria", "sports", "wifi", "transport", "bus",
        "canteen", "gym", "playground", "accommodation", "infrastructure"
    ],
    "courses": [
        "course", "courses", "program", "programme", "degree", "department",
        "branch", "stream", "subject", "bsc", "bca", "bcom", "msc", "mca",
        "mba", "be", "btech", "mtech", "phd", "undergraduate", "postgraduate"
    ],
    "placements": [
        "placement", "job", "career", "recruit", "company", "package",
        "salary", "internship", "campus drive", "hiring", "lpa", "offer"
    ]
}

# ------------------------------------------------------------------
# WELL-KNOWN COLLEGE URL MAP
# Maps common college names to their official websites
# ------------------------------------------------------------------
COLLEGE_URL_MAP = {
    "kpr": "https://kpriet.ac.in",
    "kpr college": "https://kpriet.ac.in",
    "anna university": "https://www.annauniv.edu",
    "iit madras": "https://www.iitm.ac.in",
    "iit bombay": "https://www.iitb.ac.in",
    "iit delhi": "https://home.iitd.ac.in",
    "vit": "https://vit.ac.in",
    "vit vellore": "https://vit.ac.in",
    "srm": "https://www.srmist.edu.in",
    "srm university": "https://www.srmist.edu.in",
    "psg": "https://www.psgtech.edu",
    "psg college": "https://www.psgtech.edu",
    "coimbatore institute": "https://www.citchennai.edu.in",
    "nit trichy": "https://www.nitt.edu",
    "bits pilani": "https://www.bits-pilani.ac.in",
    "manipal": "https://manipal.edu",
    "amrita": "https://www.amrita.edu",
    "sastra": "https://www.sastra.edu",
    "sathyabama": "https://www.sathyabama.ac.in",
    "loyola": "https://www.loyolacollege.edu",
    "mepco": "https://www.mepcoeng.ac.in",
    "kprcas": "https://kprcas.ac.in",
    "kpr arts": "https://kprcas.ac.in",
    "psgcas": "https://www.psgcas.ac.in",
    "psg arts": "https://www.psgcas.ac.in",
}

# ------------------------------------------------------------------
# EXTENDED KNOWLEDGE BASE (For specific requested details)
# ------------------------------------------------------------------
COLLEGE_KNOWLEDGE = {
    "kprcas": {
        "courses": [
            "BBA / BBA (Computer Application)",
            "B.Com / B.Com (CA) / B.Com (PA) / B.Com (BA)",
            "B.Com (Banking and Insurance) / B.Com (E-Commerce) / B.Com (IT)",
            "B.Sc Computer Science / B.Sc Information Technology",
            "B.Sc Computer Science with Data Analytics",
            "Bachelor of Computer Applications (BCA)",
            "M.Com (Master of Commerce)",
            "M.Sc Data Science",
            "Ph.D programs in Commerce and Management"
        ],
        "streams": ["Management", "Commerce", "Computer Science", "Textile Streams"]
    }
}


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def classify_category(message: str) -> str:
    """
    Classifies the user's question into a category using keyword matching.
    Returns the best matching category or 'general'.
    """
    message_lower = message.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in message_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)


def detect_college_name(message: str) -> str | None:
    """
    Detects if a college name is mentioned in the user's message.
    Returns the matched college key or the raw detected name.
    """
    message_lower = message.lower()

    # Check against known college map first
    for college_key in sorted(COLLEGE_URL_MAP.keys(), key=len, reverse=True):
        if college_key in message_lower:
            return college_key

    # Hardened Patterns: Require more than just a question word before 'college/campus'
    stopwords = r'\b(what|how|the|tell|show|any|some|which|me|describe|give|is|are|of|in|for|with)\b'
    
    patterns = [
        # Catch "KPR College", "PSG Institute", etc.
        # Ensure it's not just "What campus" by checking the prefix
        r'\b([a-zA-Z0-9\s]+(?:college|university|institute|academy|school|arts and science|engineering college|medical college|group of institutions|campus|varsity))\b',
        r'\b(iit|nit|bits|vit|srm|psg|iim|aiims|jipmer)\s+([a-zA-Z]+)\b'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, message_lower)
        for match_obj in matches:
            match = match_obj.group(1).strip() if match_obj.groups() else match_obj.group(0).strip()
            
            # STOPWORD FILTER: If the match is just "what campus" or starts with "what", "is", etc.
            # we strip the stopwords and see what remains.
            clean_match = re.sub(stopwords, '', match).strip()
            
            # If after stripping common words, we have nothing or just a suffix, it's a false positive
            suffixes = ['college', 'university', 'campus', 'institute', 'school']
            if not clean_match or clean_match in suffixes:
                continue
                
            return match

    return None


def fetch_college_data(college_key: str, query: str) -> dict:
    """
    Fetches and extracts text content from a college website.
    Dynamically finds the URL if not in the known map.
    """
    url = COLLEGE_URL_MAP.get(college_key)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # Dynamic Discovery: If not in map, try to find the official website
    if not url:
        print(f"🔍 Searching for official website of: {college_key}")
        try:
            # Simple heuristic: Use clearbit or similar or just a google search redirection
            # For this MVP, we use duckduckgo's !bang or a direct search query result
            search_query = f"{college_key} official website"
            # Note: In a production app, use a Search API. For this local logic, we simulate it
            # by fetching the first result from a lightweight search or constructing a very likely URL
            # For now, we'll try to find the domain using common patterns or fall back to a manual link
            
            # Simulated dynamic discovery: if it ends in college/university, try to guess
            clean_name = college_key.replace(" ", "")
            if "iit" in clean_name: url = f"https://www.{clean_name}.ac.in"
            elif "nit" in clean_name: url = f"https://www.{clean_name}.ac.in"
            else:
                # Fallback to a search link if we can't guess with high confidence
                search_url = f"https://www.google.com/search?q={college_key.replace(' ', '+')}+official+website"
                return {
                    "url": search_url,
                    "title": college_key.title(),
                    "text": f"I detected you're asking about **{college_key.title()}**. I couldn't find their official website in my database. Please [click here]({search_url}) to find it, or specify the full name!"
                }
        except Exception:
            pass

    if not url:
        return {"url": None, "title": college_key.title(), "text": "Website not found."}

    try:
        resp = requests.get(url, headers=headers, timeout=10, verify=False) # verify=False for some gov sites with bad SSL
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip noise
        for tag in soup(["script", "style", "nav", "footer", "header", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else college_key.title()
        
        # Priority content extraction: Look for 'about', 'admission', 'courses' sections
        content_parts = []
        for section in soup.find_all(["article", "main", "section", "div"]):
            # Check if section might be relevant
            text = section.get_text(separator=" ", strip=True)
            if len(text) > 100:
                content_parts.append(text)

        # Fallback to standard paragraph extraction
        if not content_parts:
            for p in soup.find_all(["p", "li"]):
                text = p.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)

        # CHECK FOR KNOWLEDGE BASE OVERRIDE
        kb_data = ""
        category = classify_category(query)
        if college_key in COLLEGE_KNOWLEDGE:
            if category == "courses" and "courses" in COLLEGE_KNOWLEDGE[college_key]:
                kb_data = "\nOFFICIAL COURSE LIST:\n- " + "\n- ".join(COLLEGE_KNOWLEDGE[college_key]["courses"])
            elif category == "campus" and "streams" in COLLEGE_KNOWLEDGE[college_key]:
                 kb_data = "\nSTREAMS OFFERED:\n- " + "\n- ".join(COLLEGE_KNOWLEDGE[college_key]["streams"])

        # SMART LINK FOLLOWING (One level only)
        # If the category is 'courses' or 'admissions', try to find a subpage
        if (category in ["courses", "admissions", "fees"]) and len(content_parts) < 15:
            links = soup.find_all("a", href=True)
            target_link = None
            for l in links:
                l_text = l.get_text().lower()
                if category[:-1] in l_text or category in l_text or "academic" in l_text or "program" in l_text:
                    target_link = l['href']
                    if not target_link.startswith("http"):
                        target_link = requests.compat.urljoin(url, target_link)
                    break
            
            if target_link and target_link != url:
                print(f"🔗 Following sublink for better {category} data: {target_link}")
                try:
                    sub_resp = requests.get(target_link, headers=headers, timeout=5, verify=False)
                    sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                    for tag in sub_soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
                    sub_text = sub_soup.get_text(separator=" ", strip=True)
                    content_parts.append("\n[From Subpage " + target_link + "]:\n" + sub_text[:2000])
                except: pass

        extracted = kb_data + "\n\n" + "\n\n".join(content_parts[:40])
        return {
            "url": url,
            "title": title,
            "text": extracted[:6000] # Increased cap for list data
        }

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {
            "url": url,
            "title": college_key.title(),
            "text": f"Could not access the official site ({url}). Please visit it directly for accurate info."
        }


def call_groq_api(system_prompt: str, user_message: str, college_info: dict = None) -> str:
    """
    Calls the Groq API (OpenAI-compatible) and returns the response text.
    Falls back to a rule-based response if the API key is missing.
    """
    if not GROQ_API_KEY:
        return fallback_response(user_message, college_info)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers,
                             json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        return fallback_response(user_message, college_info)


def fallback_response(message: str, college_info: dict = None) -> str:
    """
    Rule-based fallback response when no API key is configured.
    Covers the main categories with generic helpful answers or live data.
    """
    category = classify_category(message)
    
    # If we have scraped college info, prioritize showing that "exact data"
    if college_info and college_info.get("text") and "I couldn't find" not in college_info["text"]:
        source = college_info.get("url", "official website")
        title = college_info.get("title", "the college")
        
        intro = f"I've fetched the following exact information from the {title} official website ({source}):\n\n"
        
        # Simple extraction of the most relevant chunk based on category
        content = college_info["text"]
        # Basic filter to find lines potentially related to the category
        lines = content.split('\n')
        relevant_lines = [l for l in lines if category[:-1] in l.lower() or category in l.lower()]
        
        if relevant_lines:
            exact_data = "\n".join(relevant_lines[:10])
        else:
            exact_data = content[:1000] + "..."
            
        return f"{intro}{exact_data}\n\nFor more details, please visit: {source}"

    fallback_map = {
        "admissions": (
            "I don't have exact admission details for this institution right now. "
            "Please mention a specific college (e.g., 'KPRCAS admission') so I can fetch the latest criteria from their official site."
        ),
        "fees": (
            "Fee structures are specific to each college and course. "
            "Please specify which college you are interested in so I can look up their exact fee schedule for you."
        ),
        "exams": (
            "To provide exact exam schedules or result links, I need to know the college name. "
            "Please specify a college to get their official academic calendar."
        ),
        "campus": (
            "Campus facilities vary widely between institutions. "
            "Please mention a college name to get a detailed list of their labs, hostels, and infrastructure."
        ),
        "courses": (
            "To get the exact course list, please mention a specific college! "
            "If you specify a college name (e.g., 'KPRCAS courses'), I will fetch their live department list for you."
        ),
        "placements": (
            "Placement records and recruiters are unique to each institution. "
            "Please specify a college name to see their latest placement statistics and top recruiters."
        ),
        "general": (
            "I'm Edu-Query, your AI student assistant! I specialize in fetching **Exact Data** from college websites.\n\n"
            "To get accurate information, please ask about a specific college, for example:\n"
            "• 'Tell me about **KPRCAS**'\n"
            "• 'Admission in **VIT**'\n"
            "• 'Placements in **IIT Madras**'\n\n"
            "I will then visit their official site to provide you with verified data!"
        )
    }
    return fallback_map.get(category, fallback_map["general"])


# ==================================================================
# FLASK ROUTES
# ==================================================================

@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    1. Receives user message
    2. Classifies category
    3. Detects college name
    4. Fetches web data if needed
    5. Generates AI response
    6. Returns JSON response
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Step 1: Classify intent category
    category = classify_category(user_message)

    # Step 2: Detect college name
    new_detection = detect_college_name(user_message)
    current_session_college = session.get('active_college')
    
    # Context Logic:
    # 1. If we have a NEW detection that is IN OUR MAP, always use it (High confidence).
    # 2. If no new detection, use session.
    # 3. If new detection is weak (not in map) but session HAS a college, keep session.
    # 4. If new detection is weak and session IS EMPTY, use new detection.
    
    college_key = None
    if new_detection:
        if new_detection in COLLEGE_URL_MAP:
            college_key = new_detection
            session['active_college'] = college_key
            print(f"✅ High confidence detection: {college_key}")
        elif current_session_college:
            college_key = current_session_college
            print(f"🔄 Keeping session context {college_key} over weak detection {new_detection}")
        else:
            college_key = new_detection
            session['active_college'] = college_key
            print(f"💾 Fresh weak detection: {college_key}")
    else:
        college_key = current_session_college
        if college_key:
            print(f"🔄 Using persistent context for: {college_key}")

    # Step 3: Build context and system prompt
    college_info = None
    web_source = None

    if college_key:
        college_info = fetch_college_data(college_key, user_message)
        web_source = college_info.get("url")

        system_prompt = f"""You are Edu-Query, a helpful AI assistant for students seeking information about colleges and education in India.

The student is asking about: {college_key.title()}
Detected query category: {category}
Official website: {college_info['url']}

Here is the extracted content from the college website (PRIORITIZE "OFFICIAL COURSE LIST" IF PRESENT):
\"\"\"
{college_info['text']}
\"\"\"

Instructions:
- Answer the student's question ONLY using the website data provided above.
- IMPORTANT: If the text contains an "OFFICIAL COURSE LIST", you MUST use it for course-related questions.
- Display ALL items in the course list using bullet points.
- DO NOT provide any generic information outside of the provided context.
- Be concise, accurate, and professional.
- If the website data doesn't contain the answer, say so honestly and suggest they contact the college directly.
- Always mention the official website URL at the end.
- Response MUST be under 250 words to ensure complete information is shown.
- Use a friendly tone with emojis where appropriate."""

    else:
        system_prompt = f"""You are Edu-Query, a helpful AI assistant for students seeking information about colleges and higher education in India.

Detected query category: {category}

Instructions:
- DO NOT provide general information about courses, fees, or admissions.
- DO NOT list common undergraduate or postgraduate degrees (like BA, BSc, etc.).
- Strictly inform the student that you can only provide **Exact Data** if they specify a college name.
- Example: "I can fetch exact course lists and fee structures from any official college website. Please mention a specific college (e.g., 'KPRCAS courses') to get started."
- Be helpful but firm about needing a college name to ensure accuracy.
- Keep response under 100 words."""

    # Step 4: Generate AI response
    ai_response = call_groq_api(system_prompt, user_message, college_info)

    # Step 5: Return response
    return jsonify({
        "response": ai_response,
        "category": category,
        "college": college_key.title() if college_key else None,
        "web_source": web_source
    })


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear the college context from the session."""
    session.pop('active_college', None)
    return jsonify({"status": "context cleared"})


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "api_configured": bool(GROQ_API_KEY)
    })


# ==================================================================
# ENTRY POINT
# ==================================================================
if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not set. Using fallback responses.")
    else:
        print("✅ Groq API key detected.")
    print("🚀 Starting Edu-Query on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
