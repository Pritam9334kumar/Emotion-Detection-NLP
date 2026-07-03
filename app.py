from collections import deque

import streamlit as st

from src.prediction import predict_emotion_details


st.set_page_config(
    page_title="Emotion Detection NLP",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "recent_results" not in st.session_state:
    st.session_state.recent_results = deque(maxlen=5)


SAMPLE_TEXTS = {
    "Joy": "I finally finished the project and I feel amazing.",
    "Sadness": "I missed the deadline and I feel really down today.",
    "Anger": "This situation is so frustrating and unfair.",
    "Fear": "I am worried something bad is going to happen.",
    "Neutral": "I will review the notes and continue with the task.",
}


EMOTION_COPY = {
    "anger": {
        "title": "Strong frustration detected",
        "body": "The message reads as intense, direct, and confrontational.",
    },
    "fear": {
        "title": "Anxious or uncertain tone detected",
        "body": "The text suggests worry, caution, or concern about an outcome.",
    },
    "joy": {
        "title": "Positive emotional tone detected",
        "body": "The message sounds optimistic, satisfied, or celebratory.",
    },
    "love": {
        "title": "Warm and affectionate tone detected",
        "body": "The text suggests appreciation, care, or connection.",
    },
    "sadness": {
        "title": "Low-mood or disappointed tone detected",
        "body": "The message feels reflective, heavy, or emotionally subdued.",
    },
    "surprise": {
        "title": "Unexpected tone detected",
        "body": "The wording suggests something sudden or noteworthy.",
    },
    "neutral": {
        "title": "Balanced tone detected",
        "body": "The message is informational and does not lean strongly emotional.",
    },
}

CLASS_LABELS = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
}


def normalize_emotion_label(emotion):
    if isinstance(emotion, str):
        return emotion.strip().lower()

    try:
        return CLASS_LABELS.get(int(emotion), str(emotion).strip().lower())
    except (TypeError, ValueError):
        return str(emotion).strip().lower()


def set_sample_text(sample_key):
    st.session_state.input_text = SAMPLE_TEXTS[sample_key]


def clear_text():
    st.session_state.input_text = ""


def describe_emotion(emotion):
    label = normalize_emotion_label(emotion)

    return EMOTION_COPY.get(
        label,
        {
            "title": f"Detected emotion: {emotion}",
            "body": "The model returned a label that is shown below.",
        },
    )


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }

        .hero {
            padding: 2.2rem 2rem 1.4rem 2rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(16px);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.72rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.1rem, 3vw, 3.4rem);
            line-height: 1.05;
            color: #0f172a;
        }

        .hero p {
            margin: 0.85rem 0 0 0;
            max-width: 60ch;
            color: #334155;
            font-size: 1.02rem;
        }

        .panel {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 24px;
            padding: 1.3rem 1.3rem 1rem 1.3rem;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.06);
        }

        .result-card {
            border-radius: 24px;
            padding: 1.25rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.22);
        }

        .result-label {
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.72rem;
            color: rgba(226, 232, 240, 0.78);
            margin-bottom: 0.35rem;
        }

        .result-card h2 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }

        .result-card p {
            margin-top: 0.7rem;
            color: rgba(226, 232, 240, 0.9);
        }

        .subtle {
            color: #475569;
            font-size: 0.96rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Emotion Detection NLP</div>
        <h1>Turn raw text into emotion insight.</h1>
        <p>Paste a message, choose a sample, and get a cleaner, more readable emotion analysis experience with model feedback and recent results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.write("")


with st.sidebar:
    st.markdown("### Quick Samples")
    st.caption("Use these to test the model and compare how different tones are detected.")

    for sample_name in SAMPLE_TEXTS:
        st.button(
            sample_name,
            use_container_width=True,
            on_click=set_sample_text,
            args=(sample_name,),
        )

    st.divider()
    st.markdown("### What this app does")
    st.write("- Cleans text before prediction")
    st.write("- Runs the saved scikit-learn model")
    st.write("- Shows the latest predictions")

    if st.session_state.recent_results:
        st.divider()
        st.markdown("### Recent Results")
        for item in list(st.session_state.recent_results):
            st.write(f"**{normalize_emotion_label(item['emotion']).title()}** · {item['preview']}")


left_col, right_col = st.columns([1.7, 1], gap="large")


with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Analyze text")
    st.text_area(
        "Enter your sentence",
        key="input_text",
        placeholder="Example: I finally finished the project and I feel incredible.",
        height=180,
        label_visibility="collapsed",
    )

    action_col, clear_col = st.columns([1, 0.35])
    predict_clicked = action_col.button("Predict Emotion", use_container_width=True, type="primary")
    clear_col.button("Clear", use_container_width=True, on_click=clear_text)

    current_text = st.session_state.input_text.strip()
    if current_text:
        char_count = len(current_text)
        word_count = len(current_text.split())
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Characters", char_count)
        metric_right.metric("Words", word_count)
    else:
        st.markdown('<p class="subtle">Tip: try one of the examples in the sidebar or paste a message here.</p>', unsafe_allow_html=True)

    if predict_clicked:
        if current_text:
            result = predict_emotion_details(current_text)
            emotion = result["emotion"]
            emotion_label = normalize_emotion_label(emotion)
            confidence = result["confidence"]
            preview = current_text[:55].rstrip()
            if len(current_text) > 55:
                preview += "..."

            st.session_state.recent_results.appendleft(
                {
                    "emotion": emotion_label,
                    "preview": preview,
                }
            )

            copy = describe_emotion(emotion)
            confidence_text = "Unavailable"
            if confidence is not None:
                confidence_text = f"{confidence * 100:.0f}%"

            st.markdown(
                f'''
                <div class="result-card">
                    <div class="result-label">Detected Emotion</div>
                    <h2>{emotion_label.title()}</h2>
                    <p>{copy["title"]}</p>
                    <p>{copy["body"]}</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Confidence", confidence_text)
            result_col2.metric("Cleaned length", len(result["cleaned_text"]))

            with st.expander("See cleaned text"):
                st.write(result["cleaned_text"] or "All tokens were filtered out during cleaning.")

            st.success(f"Prediction complete: {emotion_label}")
        else:
            st.warning("Please enter some text before predicting.")

    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Why this feels better")
    st.write("This version keeps the focus on the prediction while adding stronger visual hierarchy, examples, and feedback.")
    st.write("")
    st.markdown("##### Improvements")
    st.write("- Cleaner layout with a hero section")
    st.write("- Sidebar shortcuts for sample inputs")
    st.write("- Result card with clearer emphasis")
    st.write("- Recent prediction history")
    st.write("- Live character and word counts")

    st.divider()
    st.markdown("##### Best results")
    st.caption("Short, natural sentences usually produce the clearest emotion labels.")
    st.caption("Try emotionally distinct examples to see how the model behaves.")
    st.markdown("</div>", unsafe_allow_html=True)