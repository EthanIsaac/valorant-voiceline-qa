"""VALORANT Voice-Line QA & Localization Assistant (HKUST ISOM5240 course project, educational only).

Entry point: page setup, shared layout and navigation. Two workflows:
- QA check (app_pages/qa_check.py): a tester checks one recording.
- Performance test (app_pages/performance.py): measures the app's accuracy and runtime on held-out test clips;
  kept out of the tester's navigation and linked from the footer.
Business logic lives in voiceline_qa/ and the visual components in ui/.
"""

import streamlit as st

from ui import theme

st.set_page_config(page_title="Voice-Line QA | Educational project", page_icon="🎧", layout="centered")

qa_check = st.Page("app_pages/qa_check.py", title="QA check", icon=":material/graphic_eq:", default=True)
performance = st.Page("app_pages/performance.py", title="Performance test", icon=":material/speed:",
                      url_path="performance")
page = st.navigation([qa_check, performance], position="hidden")

theme.apply_styles()
theme.banner()

page.run()

if page.url_path == performance.url_path:
    theme.footer(qa_check, "Back to the QA check")
else:
    theme.footer(performance, "Performance test: accuracy of this app on held-out test clips")
