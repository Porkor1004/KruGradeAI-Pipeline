"""Conservative lexical guards, not a semantic or mathematical verifier."""
import re
import unicodedata


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


# Explicit scoring prescriptions are forbidden even if they happen to match a
# current rubric. Adaptation instructions must defer allocations to the rubric.
_SCORING = (
    r"\b(?:always|automatically)\s+(?:score|award|assign|give)\s+\d",
    r"\b(?:award|assign|give|grant)\b.{0,40}\d+(?:\.\d+)?\s*%",
    r"\b(?:rewrite|edit|update|replace|modify|alter|change)\s+(?:the\s+)?(?:original\s+)?(?:question|reference\s+solution|(?:verified\s+)?(?:ta|human)\s+(?:labels?|scores?))\b",
    r"\b(?:award|assign|give|deduct|add|subtract|grant)\b.{0,55}\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|half|full|maximum|all|no)\s*(?:points?|marks?|credit)\b",
    r"\b(?:set|force|override|change|adjust|replace|increase|decrease|modify|alter|rescale|multiply)\b.{0,60}\b(?:score|scores|weight|weights|maximum|maxima|max_score|partial.credit|allocation|allocations)\b",
    r"\b(?:ignore|disregard|override|replace|modify|alter|change)\s+(?:the\s+)?rubric\b",
    r"\b(?:change|replace|modify|alter|remove|add)\s+(?:the\s+)?(?:rubric\s+)?criteri(?:a|on)\b",
    r"\b(?:score|credit|points?|marks?|weight|weights|maximum|max_score)\s*(?:=|:=|:|to)\s*\d",
    r"\b(?:do not|don't|never)\s+(?:follow|use|respect|apply)\s+(?:the\s+)?rubric\b",
)
_OVERFIT = (
    r"\bfor\s+(?:the\s+)?student\s+(?!answers?\b|responses?\b|work\b)\S+",
    r"\b(?:student|response|submission)\s+(?:id\b|#|[=:])",
    r"\b(?:the\s+)?(?:correct|expected)\s+answer\s+(?:is|=|:)\s*\S+",
    r"\banswer\s+key\s*(?:is|=|:)\s*\S+",
    r"\baccept\s+.{1,60}\s+as\s+(?:the\s+)?correct\s+answer\b",
    r"\b(?:if|when)\s+(?:the\s+)?(?:student\s+)?answer\s+(?:contains|equals|matches)\s+\S+",
)
# Bracketed references and explicit 'criterion ID' references are unambiguous.
_CRITERION_REF = re.compile(r"\bcriterion\s+(?:id\s*[:=]?\s*[`\"'\[]?([\w.-]+)|[`\"'\[]([\w.-]+)[`\"'\]])", re.IGNORECASE)


def validate_rule_text(text: str, known_ids: set[str], answer_quotes: tuple[str, ...]) -> None:
    clean = normalized(text)
    for pattern in _SCORING:
        for match in re.finditer(pattern, clean):
            # 'Never override the rubric' is a guard, not an override. Only
            # immediate negation of a prohibited action is exempted.
            prefix = clean[max(0, match.start() - 12):match.start()]
            if re.search(r"(?:never|do not|must not|don't)\s+$", prefix):
                continue
            raise ValueError("Rule contains an explicit scoring or rubric override")
    if any(re.search(pattern, clean) for pattern in _OVERFIT):
        raise ValueError("Rule contains detectable student-specific content or an answer key")
    for match in _CRITERION_REF.finditer(text):
        key = match.group(1) or match.group(2)
        if key not in known_ids:
            raise ValueError("Rule text references an unknown rubric criterion")
    # Short/common words are not identifying answer content. Short mathematical
    # excerpts still identify worked answers and must not escape the length gate.
    for quote in answer_quotes:
        excerpt = normalized(quote)
        mathematical = len(excerpt) >= 4 and re.search(r"\d", excerpt) and re.search(r"[=+*/^<>-]", excerpt)
        copied = excerpt in clean
        if mathematical:
            copied = re.sub(r"\s+", "", excerpt) in re.sub(r"\s+", "", clean)
        if (len(excerpt) >= 8 or mathematical) and copied:
            raise ValueError("Rule copies student-answer or reference-solution evidence")
