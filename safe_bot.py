import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
import time

# ---- SETUP ----
# from dotenv import load_dotenv
# load_dotenv()
# from openai import OpenAI
# client = OpenAI(base_url="",api_key="")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- LOAD & EMBED PDFs ----
PDF_PATHS = [
    # "data/engAGED_Online+Safety+1_508.pdf",
    # "data/engAGED_Online+Safety+2_508.pdf",
    # "data/engAGED_Online+Safety+3_508.pdf"
]

if "vector_store" not in st.session_state:
    all_chunks = []
    for pdf_path in PDF_PATHS:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-small", api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
    st.session_state["vector_store"] = vector_store

# ---- UI HEADER ----
st.set_page_config(page_title="Safe Internet Guide Bot", layout="wide")
st.title("🛡️ Safe Internet Guide Bot")
st.caption("Helping older adults learn to stay safe online with tips, quizzes, and chat support.")

# Define questions for Browsing the Internet Confidently
browsing_questions = [
    {
        "question": "What should you check for in a web address before entering personal information?",
        "options": ["http://", "https://", "www", ".com"],
        "answer": "https://",
        "explanation": "Websites using 'https://' are more secure because they encrypt your data."
    },
    {
        "question": "How can you close most pop-up ads safely?",
        "options": ["Click anywhere on it", "Click the X in the corner", "Ignore it", "Click 'Download'"],
        "answer": "Click the X in the corner",
        "explanation": "Clicking the 'X' is the safest way to close pop-ups. Avoid clicking inside the ad."
    },
    {
        "question": "Which is a sign that a website might be fake?",
        "options": ["It has videos", "It loads slowly", "The URL has typos or strange words", "It asks you to log in"],
        "answer": "The URL has typos or strange words",
        "explanation": "Scam websites often use misspelled names or unusual URLs to trick people."
    }
]

protecting_questions = [
    {
        "question": "Which type of Wi-Fi should you avoid when checking your bank or health accounts?",
        "options": ["Public Wi-Fi", "Home Wi-Fi", "Mobile Hotspot", "Office Network"],
        "answer": "Public Wi-Fi",
        "explanation": "Public Wi-Fi is not secure and can be easily hacked. Use secure, private networks for sensitive tasks."
    },
    {
        "question": "Which of the following is an example of a strong password?",
        "options": ["password123", "Myp@ssw0rd!", "january2024", "12345678"],
        "answer": "Myp@ssw0rd!",
        "explanation": "Strong passwords use a mix of letters, numbers, and symbols. Avoid simple or common ones."
    },
    {
        "question": "What does a password manager do?",
        "options": [
            "Stores and organizes your passwords securely",
            "Sends your passwords to your email",
            "Makes your internet faster",
            "Blocks spam emails"
        ],
        "answer": "Stores and organizes your passwords securely",
        "explanation": "A password manager helps you safely store strong, unique passwords for different accounts."
    },
    {
        "question": "What is two-factor authentication (2FA)?",
        "options": [
            "Logging in twice",
            "Using a second method to confirm your identity",
            "Changing your password weekly",
            "Using only numbers in your password"
        ],
        "answer": "Using a second method to confirm your identity",
        "explanation": "2FA adds a second layer of security, like a code sent to your phone, after your password."
    },
    {
        "question": "Which of the following is a sign of a scam email?",
        "options": [
            "Personalized greeting",
            "No spelling mistakes",
            "Sense of urgency and typos",
            "Comes from your bank's exact domain"
        ],
        "answer": "Sense of urgency and typos",
        "explanation": "Scam emails often try to rush you and contain spelling errors or odd links."
    }
]

socializing_questions = [
    {
        "question": "Who should you accept friend requests from?",
        "options": [
            "Anyone who sends one",
            "People with many mutual friends",
            "Only people you know in real life",
            "People who say they're family"
        ],
        "answer": "Only people you know in real life",
        "explanation": "Strangers can pretend to be someone else. Accept requests only from people you personally know."
    },
    {
        "question": "Which information should you never share on social media?",
        "options": [
            "Favorite foods",
            "Your full address or credit card number",
            "Photos of pets",
            "TV shows you like"
        ],
        "answer": "Your full address or credit card number",
        "explanation": "Personal information like address or card numbers can be used to steal your identity."
    },
    {
        "question": "What’s a safe way to check if an online relationship is real?",
        "options": [
            "Send them money to build trust",
            "Ask for a video call",
            "Believe their profile description",
            "Add them to your account logins"
        ],
        "answer": "Ask for a video call",
        "explanation": "Romance scammers avoid video calls. Asking for one helps confirm if they’re genuine."
    },
    {
        "question": "How can you keep virtual meetings safe?",
        "options": [
            "Use a passcode and waiting room",
            "Keep the camera off",
            "Let anyone join",
            "Use the same link forever"
        ],
        "answer": "Use a passcode and waiting room",
        "explanation": "These features help prevent uninvited guests from joining your meetings."
    },
    {
        "question": "What should you do if someone online makes you uncomfortable?",
        "options": [
            "Block and report them",
            "Ignore it and hope it stops",
            "Give them another chance",
            "Share your number to talk offline"
        ],
        "answer": "Block and report them",
        "explanation": "You have the right to feel safe. Block and report anyone who acts suspicious or threatening."
    }
]

# ---- Initialize session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m your online safety guide. Ask me anything about staying safe on the internet."
            )
        }
    ]

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

if "quiz_high_scores" not in st.session_state:
    st.session_state.quiz_high_scores = {
        "Browsing": 0,
        "Protecting": 0,
        "Socializing": 0
    }

if "quiz_scores" not in st.session_state:
    st.session_state.quiz_scores = {
        "Browsing": 0,
        "Protecting": 0,
        "Socializing": 0
    }

if "browsing_quiz" not in st.session_state:
    st.session_state.browsing_quiz = {
        'step': 'tips',
        'question_index': 0,
        'correct': False,
        'score': 0
    }

if "protecting_quiz" not in st.session_state:
    st.session_state.protecting_quiz = {
        'step': 'tips',
        'question_index': 0,
        'correct': False,
        'score': 0
    }

if "socializing_quiz" not in st.session_state:
    st.session_state.socializing_quiz = {
        'step': 'tips',
        'question_index': 0,
        'correct': False,
        'score': 0
    }

# ---- TAB NAVIGATION ----
tab1, tab2, tab3 = st.tabs(["📘 Learn & Quiz", "💬 Ask the Bot", "📊 Dashboard"])

# ---- LEARN TAB ----
with tab1:

    with st.container():
        st.header("🧓 Learn About Online Safety")

        st.image("images/banner2.png", use_container_width=True, caption="Staying safe online is for everyone!")

        st.markdown("### ✅ Simple Tips for Safer Internet Use")
        st.write("This guide is especially designed for seniors. Each section is easy to explore and has tips that are simple to follow.")

        with st.expander("🌐 **Browsing the Internet Confidently**"):
            quiz_state = st.session_state.browsing_quiz

            if quiz_state['step'] == 'tips':
                st.image("images/safe_internet.png", width=300)
                st.markdown("""
                **Tips:**
                - ✅ Look for **`https://`** before entering personal information.
                - 🔍 Use browser safety tools (e.g., Chrome’s Safe Browsing).
                - ❌ Close pop-ups by clicking the tiny **‘X’** (usually top-right).
                - 📢 Watch out for **sponsored ads** at the top of search results.
                - 🔗 Avoid fake websites: check spelling and unusual web addresses.
                - 📜 Ask yourself: *Is this website trustworthy?*
                """)
                if st.button("📝 Take the Quiz", key="browse_quiz_start"):
                    quiz_state['step'] = 'quiz'
                    quiz_state['question_index'] = 0
                    quiz_state['correct'] = False
                    quiz_state['score'] = 0
                    quiz_state['tried_wrong'] = False  # ✅ initialize tracking for first try
                    st.rerun()

            elif quiz_state['step'] == 'quiz':
                q = browsing_questions[quiz_state['question_index']]
                st.markdown(f"**Q{quiz_state['question_index'] + 1}: {q['question']}**")

                # Initialize tracking flag if not present
                if 'tried_wrong' not in quiz_state:
                    quiz_state['tried_wrong'] = False

                for option in q['options']:
                    if st.button(option, key=f"browse_q{quiz_state['question_index']}_{option}"):
                        if option == q['answer']:
                            st.success("✅ Correct!")
                            if not quiz_state['tried_wrong']:
                                quiz_state['score'] += 1  # ✅ Only if no wrong attempts
                            time.sleep(0.5)
                            if quiz_state['question_index'] + 1 < len(browsing_questions):
                                quiz_state['question_index'] += 1
                                quiz_state['tried_wrong'] = False  # ✅ Reset for next question
                            else:
                                quiz_state['step'] = 'complete'
                            st.rerun()
                        else:
                            st.error(f"❌ Incorrect. {q['explanation']}")
                            quiz_state['tried_wrong'] = True  # ✅ Mark this question as tried wrong

            elif quiz_state['step'] == 'complete':
                score = quiz_state['score']
                if score > st.session_state.quiz_high_scores["Browsing"]:
                    st.session_state.quiz_high_scores["Browsing"] = score

                st.success("🎉 You've completed the quiz!")
                st.markdown(f"**✅ Your Score:** {score} / {len(browsing_questions)}")
                st.markdown(f"🏆 **Highest Score:** {st.session_state.quiz_high_scores['Browsing']} / {len(browsing_questions)}")

                if st.button("🔁 Retake Quiz", key="browse_quiz_retry"):
                    quiz_state['step'] = 'quiz'
                    quiz_state['question_index'] = 0
                    quiz_state['correct'] = False
                    quiz_state['score'] = 0
                    quiz_state['tried_wrong'] = False
                    st.rerun()

                if st.button("📖 Review Tips", key="browse_quiz_back"):
                    quiz_state['step'] = 'tips'
                    st.rerun()



        with st.expander("🔐 **Protecting Your Personal Information**"):
            protect_state = st.session_state.protecting_quiz

            if protect_state['step'] == 'tips':
                st.image("images/safe_password.png", width=300)
                st.markdown("""
                **Tips:**
                - 🚫 Never share your **Social Security number** or **bank info** in emails.
                - 📶 Use secure Wi-Fi, not public Wi-Fi, for banking or health info.
                - 🔐 Create **strong passwords** (8+ characters, use symbols and numbers).
                - 🧠 Consider using a **password manager** like LastPass or Bitwarden.
                - 📲 Turn on **two-factor authentication** for your accounts.
                - 🛑 Watch for scam signs: **typos**, **“urgent” requests**, **odd links**.
                - 💡 If something feels off, it probably is. **Trust your instincts**.
                """)
                if st.button("📝 Take the Quiz", key="protect_quiz_start"):
                    protect_state['step'] = 'quiz'
                    protect_state['question_index'] = 0
                    protect_state['correct'] = False
                    protect_state['score'] = 0
                    protect_state['tried_wrong'] = False
                    st.rerun()

            elif protect_state['step'] == 'quiz':
                q = protecting_questions[protect_state['question_index']]
                st.markdown(f"**Q{protect_state['question_index'] + 1}: {q['question']}**")

                if 'tried_wrong' not in protect_state:
                    protect_state['tried_wrong'] = False

                for option in q['options']:
                    if st.button(option, key=f"protect_q{protect_state['question_index']}_{option}"):
                        if option == q['answer']:
                            st.success("✅ Correct!")
                            if not protect_state['tried_wrong']:
                                protect_state['score'] += 1
                            time.sleep(0.5)
                            if protect_state['question_index'] + 1 < len(protecting_questions):
                                protect_state['question_index'] += 1
                                protect_state['tried_wrong'] = False
                            else:
                                protect_state['step'] = 'complete'
                            st.rerun()
                        else:
                            st.error(f"❌ Incorrect. {q['explanation']}")
                            protect_state['tried_wrong'] = True

            elif protect_state['step'] == 'complete':
                score = protect_state['score']
                if score > st.session_state.quiz_high_scores["Protecting"]:
                    st.session_state.quiz_high_scores["Protecting"] = score

                st.success("🎉 You've completed the quiz!")
                st.markdown(f"**✅ Your Score:** {score} / {len(protecting_questions)}")
                st.markdown(f"🏆 **Highest Score:** {st.session_state.quiz_high_scores['Protecting']} / {len(protecting_questions)}")

                if st.button("🔁 Retake Quiz", key="protect_quiz_retry"):
                    protect_state['step'] = 'quiz'
                    protect_state['question_index'] = 0
                    protect_state['correct'] = False
                    protect_state['score'] = 0
                    protect_state['tried_wrong'] = False
                    st.rerun()

                if st.button("📖 Review Tips", key="protect_quiz_back"):
                    protect_state['step'] = 'tips'
                    st.rerun()



        with st.expander("👥 **Socializing Safely Online**"):
            social_state = st.session_state.socializing_quiz

            if social_state['step'] == 'tips':
                st.image("images/safe_3.png", width=300)
                st.markdown("""
                **Tips:**
                - 🗞️ Only share what you’d be okay seeing in a newspaper.
                - 🛡️ Check your **privacy settings** on Facebook, Instagram, etc.
                - 🚷 Don’t accept friend requests from people you don’t know.
                - 🔒 Never share your **address**, **credit card**, or **SSN**.
                - 💔 Watch out for **romance scams** – video chat before trusting.
                - 🚨 Report and block anyone suspicious.
                - 🔑 For Zoom or online events: use **passcodes and waiting rooms**.
                """)
                if st.button("📝 Take the Quiz", key="social_quiz_start"):
                    social_state['step'] = 'quiz'
                    social_state['question_index'] = 0
                    social_state['correct'] = False
                    social_state['score'] = 0
                    social_state['tried_wrong'] = False
                    st.rerun()

            elif social_state['step'] == 'quiz':
                q = socializing_questions[social_state['question_index']]
                st.markdown(f"**Q{social_state['question_index'] + 1}: {q['question']}**")

                if 'tried_wrong' not in social_state:
                    social_state['tried_wrong'] = False

                for option in q['options']:
                    if st.button(option, key=f"social_q{social_state['question_index']}_{option}"):
                        if option == q['answer']:
                            st.success("✅ Correct!")
                            if not social_state['tried_wrong']:
                                social_state['score'] += 1
                            time.sleep(0.5)
                            if social_state['question_index'] + 1 < len(socializing_questions):
                                social_state['question_index'] += 1
                                social_state['tried_wrong'] = False
                            else:
                                social_state['step'] = 'complete'
                            st.rerun()
                        else:
                            st.error(f"❌ Incorrect. {q['explanation']}")
                            social_state['tried_wrong'] = True

            elif social_state['step'] == 'complete':
                score = social_state['score']
                if score > st.session_state.quiz_high_scores["Socializing"]:
                    st.session_state.quiz_high_scores["Socializing"] = score

                st.success("🎉 You've completed the quiz!")
                st.markdown(f"**✅ Your Score:** {score} / {len(socializing_questions)}")
                st.markdown(f"🏆 **Highest Score:** {st.session_state.quiz_high_scores['Socializing']} / {len(socializing_questions)}")

                if st.button("🔁 Retake Quiz", key="social_quiz_retry"):
                    social_state['step'] = 'quiz'
                    social_state['question_index'] = 0
                    social_state['correct'] = False
                    social_state['score'] = 0
                    social_state['tried_wrong'] = False
                    st.rerun()

                if st.button("📖 Review Tips", key="social_quiz_back"):
                    social_state['step'] = 'tips'
                    st.rerun()


# ---- Ask TAB ----
with tab2:
    st.header("Ask Me Anything")

    chat_placeholder = st.container()
    status_placeholder = st.empty()
    input_placeholder = st.empty()

    with chat_placeholder:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    disabled = st.session_state.awaiting_response
    with input_placeholder:
        question = st.chat_input("Ask something about internet safety", disabled=disabled)

    if question and not st.session_state.awaiting_response:
        st.session_state.awaiting_response = True
        st.session_state.messages.append({"role": "user", "content": question})

        with status_placeholder:
            st.info("⏳ Assistant is thinking... Please wait.")

        with chat_placeholder:
            st.chat_message("user").write(question)


        # Retrieve context via RAG
        retriever = st.session_state.vector_store.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        with chat_placeholder:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model="openai.gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a friendly, patient internet safety guide for older adults. "
                                "Use the following context to answer clearly and simply."
                                f"\n\n{context}"
                            )
                        },
                        *st.session_state.messages
                    ],
                    stream=True
                )
                collected_response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": collected_response})
        st.session_state.awaiting_response = False
        status_placeholder.empty()

with tab3:
    st.header("📊 Your Progress Dashboard")

    scores = st.session_state.quiz_high_scores

    st.markdown("### 🏆 Highest Scores")
    st.markdown(f"- 🌐 **Browsing the Internet Confidently**: {scores['Browsing']} / {len(browsing_questions)}")
    st.markdown(f"- 🔐 **Protecting Your Personal Information**: {scores['Protecting']} / {len(protecting_questions)}")
    st.markdown(f"- 👥 **Socializing Safely Online**: {scores['Socializing']} / {len(socializing_questions)}")

    st.markdown("### 🧠 Suggested Study Focus")

    # Suggest based on lowest score
    min_score = min(scores.values())
    weakest_topic = [k for k, v in scores.items() if v == min_score]

    if min_score == max(len(browsing_questions), len(protecting_questions), len(socializing_questions)):
        st.success("👏 Great job! You've scored full marks in every category.")
    else:
        for topic in weakest_topic:
            if topic == "Browsing":
                st.warning("🔁 Consider reviewing tips on identifying safe websites, pop-ups, and sponsored ads.")
            elif topic == "Protecting":
                st.warning("🔁 Brush up on password safety, email scams, and Wi-Fi risks.")
            elif topic == "Socializing":
                st.warning("🔁 Revisit social media safety, privacy settings, and avoiding scams.")

    st.markdown("You can always return to the **Learn & Quiz** tab to review and try again.")
