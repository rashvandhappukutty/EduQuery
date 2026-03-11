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


def analyze_question_with_ai(message: str) -> dict:
    """
    Uses the AI API to intelligently detect the college name and intent.
    Returns: {"college": str or None, "intent": str}
    """
    if not GROQ_API_KEY:
        # Fallback to rule-based analysis if no API key
        return {
            "college": detect_college_name(message),
            "intent": classify_category(message)
        }

    system_prompt = """You are an AI Question Analyzer for 'EduQuery'. 
Extract the college name and the intent from the student's question.
Valid intents: Admissions, Courses, Fees, Exams, Facilities, General Information.

Respond ONLY with a JSON object: 
{"college": "College Name", "intent": "Intent Category"}

If no college is mentioned, set "college" to null.
If no specific intent is found, set "intent" to "General Information"."""

    try:
        response_text = call_groq_api(system_prompt, message)
        # Parse JSON from AI response
        import json
        # Handle potential markdown formatting in response
        clean_json = re.search(r'\{.*\}', response_text, re.DOTALL)
        if clean_json:
            result = json.loads(clean_json.group(0))
            # Standardize intent to lowercase to match existing logic
            result["intent"] = result.get("intent", "general").lower()
            return result
    except Exception as e:
        print(f"AI Analysis Error: {e}")
    
    # Final fallback
    return {"college": detect_college_name(message), "intent": classify_category(message)}


def detect_college_name(message: str) -> str | None:
    """
    Rule-based fallback for college name detection.
    """
    message_lower = message.lower()
    for college_key in sorted(COLLEGE_URL_MAP.keys(), key=len, reverse=True):
        if college_key in message_lower:
            return college_key
    patterns = [
        r'\b([a-zA-Z0-9\s]+(?:college|university|institute|academy|school|arts and science|engineering college|medical college|group of institutions|campus|varsity))\b',
        r'\b(iit|nit|bits|vit|srm|psg|iim|aiims|jipmer)\s+([a-zA-Z]+)\b'
    ]
    stopwords = r'\b(what|how|the|tell|show|any|some|which|me|describe|give|is|are|of|in|for|with|about|where|when|can|you|if)\b'
    for pattern in patterns:
        matches = re.finditer(pattern, message_lower)
        for match_obj in matches:
            match = match_obj.group(0).strip()
            clean_match = re.sub(stopwords, '', match).strip()
            suffixes = ['college', 'university', 'campus', 'institute', 'school', 'varsity']
            if not clean_match or clean_match in suffixes: continue
            return clean_match
    return None

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
        
        # Step A: Heuristic Guessing
        clean_name = college_key.lower().replace(" ", "")
        guesses = [
            f"https://www.{clean_name}.ac.in",
            f"https://www.{clean_name.replace('college', '')}.ac.in",
            f"https://www.{clean_name}.edu.in",
            f"https://{clean_name}.in"
        ]
        
        for guess in guesses:
            try:
                check = requests.head(guess, timeout=3, allow_redirects=True, verify=False)
                if check.status_code < 400:
                    url = guess
                    print(f"✅ Guessed valid URL: {url}")
                    break
            except: continue

        # Step B: If still no URL, we return a search link and signal for AI Fallback
        if not url:
            search_url = f"https://www.google.com/search?q={college_key.replace(' ', '+')}+official+website"
            return {
                "url": search_url,
                "title": college_key.title(),
                "text": "NOT_FOUND", # Special keyword for AI Fallback
                "search_link": search_url
            }

    if not url:
        return {"url": None, "title": college_key.title(), "text": "NOT_FOUND"}

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

    # Step 1 & 2: Intelligent Detection using AI (or fallback logic)
    analysis = analyze_question_with_ai(user_message)
    category = analysis.get("intent", "general")
    detected_college = analysis.get("college")

    current_session_college = session.get('active_college')
    
    # Context Logic
    college_key = None
    if detected_college:
        college_key = detected_college.lower().strip()
        session['active_college'] = college_key
        print(f"🤖 AI Detected College: {college_key}")
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

        # UNIVERSAL FALLBACK: If website data is missing or inaccessible
        if college_info['text'] == "NOT_FOUND" or len(college_info['text']) < 100:
            print(f"⚠️ Live data unavailable for {college_key}. Using AI Knowledge Fallback.")
            
            system_prompt = f"""You are Edu-Query, a helpful AI assistant for students.
I could not find the live official website for {college_key.title()} right now.

Instructions:
- Answer the student's question based on your **General AI Knowledge** about {college_key.title()}.
- IMPORTANT: Clearly state at the beginning (using an emoji) that you are using general information because the live site is currently inaccessible.
- If you don't know the specific details (like current fees), provide typical estimates for similar colleges in that region and suggest they verify at the official link.
- Keep response under 250 words.
- Use a friendly tone with emojis 🎓.
- Official Website: {college_info.get('search_link', 'Search Google for official info')}"""

        else:
            # Standard Scraping Prompt
            system_prompt = f"""You are Edu-Query, a helpful AI assistant for students.
The student is asking about: {college_key.title()}
Detected query category: {category}
Official website: {college_info['url']}

Extracted Live Content:
\"\"\"
{college_info['text']}
\"\"\"

Instructions:
- Answer the student's question ONLY using the website data provided above.
- If the answer is in the data, provide it accurately.
- If the data is missing the answer, say so and suggest they visit the official site.
- Be concise, accurate, and professional.
- Response MUST be under 250 words.
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
        "web_source": web_source,
        "is_live": college_info['text'] != "NOT_FOUND" if college_info else False
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
