def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words=None,
):
    stop_words = stop_words or []
    code = code.replace("\r", "")
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    if language_type and language_type.lower() == "python":
        stop_words = ["\ndef ", "\nclass ", "\nif __name__", "\nprint("]
    return _truncate_code_at_stopwords(code, stop_words).strip()


def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]


def postprocess_code_generation(raw_response: str):
    raw = (raw_response or "").replace("\r", "")
    for marker in ("[DONE]", "<|endoftext|>", "<|eot_id|>", "</s>", "\n### Instruction:", "\n### Response:"):
        index = raw.find(marker)
        if index >= 0:
            raw = raw[:index]
    raw = raw.strip("\n")

    fenced = False
    code = None
    import re

    for pattern in (
        r"```(?:python|py)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            code = match.group(1).strip("\n")
            fenced = True
            break

    if code is None:
        lines = raw.splitlines()
        start = 0
        start_re = re.compile(
            r"^\s*(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+\s*\(|class\s+\w+|"
            r"[A-Za-z_]\w*\s*=|[A-Za-z_]\w*\s*\()"
        )
        for index, line in enumerate(lines):
            if start_re.match(line):
                start = index
                break
        code = "\n".join(lines[start:]).strip("\n")

    kept = []
    trailing_removed = False
    prose_re = re.compile(
        r"^(?:here\s+is|this\s+function|the\s+function|it\s+works|explanation|note:|"
        r"example\s+usage|test\s+cases)",
        flags=re.IGNORECASE,
    )
    for line in code.splitlines():
        stripped = line.strip()
        top_level = line == line.lstrip()
        if stripped.startswith("```"):
            trailing_removed = True
            break
        if top_level and (
            stripped.startswith("if __name__")
            or stripped.startswith("print(")
            or prose_re.match(stripped)
        ):
            if kept:
                trailing_removed = True
                break
        kept.append(line)
    code = "\n".join(kept).strip("\n")

    compile_ok = True
    compile_error = ""
    try:
        compile(code + "\n", "<mbpp_generation>", "exec")
    except SyntaxError as exc:
        compile_ok = False
        compile_error = f"{type(exc).__name__}: {exc}"

    return {
        "raw_response": raw_response or "",
        "generation": code.strip() + "\n" if code.strip() else "",
        "postprocess_meta": {
            "fenced_code_extracted": fenced,
            "trailing_text_removed": trailing_removed,
            "compile_ok": compile_ok,
            "compile_error": compile_error,
        },
    }
