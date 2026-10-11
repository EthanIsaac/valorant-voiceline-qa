"""Agent names and callouts that a Spanish subtitle must keep (the check of notebook 04)."""

import re

from voiceline_qa.agents import AGENTS

_NAME_PATTERNS = {name: re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
                  for name in [agent.name for agent in AGENTS.values()] + ["Mid"]}
# Site letters are matched in capitals: in Spanish a lower-case "a" is a preposition
_SITE_PATTERNS = {site: re.compile(rf"(?<!\w){site}(?!\w)") for site in ["A", "B", "C"]}
# In English "A" is also an article ("A little help.") or part of a word ("A-okay"): in the source line it only
# counts as a site when not followed by an ordinary lower-case word (callout words such as "long" excepted)
_SOURCE_SITE_PATTERNS = {**_SITE_PATTERNS, "A": re.compile(
    r"(?<!\w)A(?![\w-])(?! (?!(?:site|long|short|main|link|lobby|heaven)\b)[a-z])")}
_TARGET_PATTERNS = {**_NAME_PATTERNS, **_SITE_PATTERNS}


def protected_terms(english_line: str) -> list[str]:
    """Agent names, Mid and site letters that appear in an English line."""
    names = [name for name, pattern in _NAME_PATTERNS.items() if pattern.search(english_line)]
    sites = [site for site, pattern in _SOURCE_SITE_PATTERNS.items() if pattern.search(english_line)]
    return names + sites


def lost_terms(english_line: str, spanish_line: str) -> list[str]:
    """Protected terms of the English line that the Spanish line no longer contains."""
    return [term for term in protected_terms(english_line) if not _TARGET_PATTERNS[term].search(spanish_line)]
