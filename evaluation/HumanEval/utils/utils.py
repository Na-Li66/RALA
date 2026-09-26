import ast
import re
import textwrap

languge_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'cpp': {
        'full_name': 'cpp',
        'indent': 0,
        'main': "int main()",
    },
    'java': {
        'full_name': 'Java',
        'indent': 4,
        'main': "public static void main",
    },
    'cs': {
        'full_name': "csharp",
        'indent': 0,
        'main': "public static void Main",
    },
    'php': {
        'full_name': "PHP",
        'indent': 0,
    },
    'ts': {
        'full_name': "TypeScript",
        'indent': 0,
    },
    'js': {
        'full_name': "JavaScript",
        'indent': 0
    },
    'sh': {
        'full_name': "Bash",
        'indent': 0
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix

    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(example: str, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = languge_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']

    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))

        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]

        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1

            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ['php', 'ts', 'js']:
            body += '\n' + ' '*indent + '}'

        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output

    return example

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        # Do not truncate on Python syntax such as "\ndef" or "\nif".
        # HumanEval candidates may legitimately contain helper functions,
        # classes, comments, or top-level imports. Postprocessing below handles
        # wrappers and demo code without changing valid solution structure.
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]


_CODE_START_RE = re.compile(
    r"^\s*(?:"
    r"from\s+\S+\s+import\s+|"
    r"import\s+\S+|"
    r"def\s+\w+\s*\(|"
    r"class\s+\w+|"
    r"return\b|"
    r"if\b|"
    r"for\b|"
    r"while\b|"
    r"try:|"
    r"with\b|"
    r"raise\b|"
    r"[A-Za-z_]\w*\s*=|"
    r"[A-Za-z_]\w*\s*\(|"
    r"\[|\{"
    r")"
)


def infer_python_entry_point(prompt: str) -> str:
    matches = list(re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\(", prompt, flags=re.MULTILINE))
    return matches[-1].group(1) if matches else ""


def split_prompt_before_entry_def(prompt: str, entry_point: str) -> str:
    if not entry_point:
        return ""
    pattern = re.compile(
        r"^def\s+" + re.escape(entry_point) + r"\s*\(",
        flags=re.MULTILINE,
    )
    match = pattern.search(prompt)
    if not match:
        return ""
    return prompt[: match.start()].rstrip()


def _cut_answer_delimiters(text: str) -> str:
    delimiters = [
        "[DONE]",
        "<|endoftext|>",
        "<|eot_id|>",
        "</s>",
        "\n### Instruction:",
        "\n### Response:",
    ]
    end = len(text)
    for marker in delimiters:
        index = text.find(marker)
        if 0 <= index < end:
            end = index
    return text[:end]


def _extract_fenced_code(text: str):
    for pattern in (
        r"```(?:python|py)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip("\n"), True
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip("\n"), True
    return None, False


def _drop_leading_prose(text: str):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if _CODE_START_RE.match(line):
            return "\n".join(lines[index:]).strip("\n"), index > 0
    return text.strip("\n"), False


def _drop_trailing_prose_and_demos(code: str):
    lines = code.splitlines()
    kept = []
    truncated = False
    prose_re = re.compile(
        r"^(?:"
        r"here\s+is|"
        r"this\s+function|"
        r"the\s+function|"
        r"it\s+works|"
        r"explanation|"
        r"note:|"
        r"example\s+usage|"
        r"test\s+cases"
        r")",
        flags=re.IGNORECASE,
    )
    for line in lines:
        stripped = line.strip()
        top_level = line == line.lstrip()
        if stripped.startswith("```"):
            truncated = True
            break
        if top_level and (
            stripped.startswith("if __name__")
            or stripped.startswith("print(")
            or stripped.startswith("assert ")
            or prose_re.match(stripped)
        ):
            if kept:
                truncated = True
                break
        kept.append(line)
    return "\n".join(kept).strip("\n"), truncated


def _remove_prompt_echo(code: str, prompt: str):
    normalized_code = code.lstrip("\n")
    candidates = [
        prompt,
        prompt.strip(),
        prompt.rstrip(),
    ]
    for candidate in candidates:
        if candidate and normalized_code.startswith(candidate):
            return normalized_code[len(candidate):].lstrip("\n"), True
    return code, False


def _has_top_level_target_def(code: str, entry_point: str) -> bool:
    if not entry_point:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(
            re.search(
                r"^def\s+" + re.escape(entry_point) + r"\s*\(",
                code,
                flags=re.MULTILINE,
            )
        )
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _keep_solution_top_level_nodes(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code.strip("\n")

    lines = code.splitlines()
    segments = []
    saw_target = False
    allowed_before_target = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    allowed_after_target = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)

    for node in tree.body:
        is_allowed = isinstance(node, allowed_after_target if saw_target else allowed_before_target)
        is_target = isinstance(node, ast.FunctionDef) and node.name == entry_point
        if is_allowed:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None:
                segments.append("\n".join(lines[start - 1:end]))
            if is_target:
                saw_target = True
            continue
        if saw_target:
            break

    if saw_target and segments:
        return "\n\n".join(segment.strip("\n") for segment in segments if segment.strip()).strip("\n")
    return code.strip("\n")


def _indent_function_body(code: str) -> str:
    body = textwrap.dedent(code).strip("\n")
    if not body:
        return "    pass"
    return "\n".join(("    " + line if line.strip() else line) for line in body.splitlines())


def postprocess_humaneval_generation(raw_response: str, original_prompt: str):
    raw = (raw_response or "").replace("\r", "")
    raw = _cut_answer_delimiters(raw).strip("\n")
    entry_point = infer_python_entry_point(original_prompt)

    fenced_code, fenced = _extract_fenced_code(raw)
    if fenced_code is None:
        clean, prose_removed = _drop_leading_prose(raw)
    else:
        clean = fenced_code
        prose_removed = raw.strip() != fenced_code.strip()

    clean, prompt_echo_removed = _remove_prompt_echo(clean, original_prompt)
    clean, trailing_removed = _drop_trailing_prose_and_demos(clean)
    kind = "full_function" if _has_top_level_target_def(clean, entry_point) else "body"

    if kind == "full_function":
        solution = _keep_solution_top_level_nodes(clean, entry_point)
        prefix = split_prompt_before_entry_def(original_prompt, entry_point)
        if prefix:
            generation = prefix + "\n\n" + solution.rstrip() + "\n"
        else:
            generation = solution.rstrip() + "\n"
        completion = "\n" + solution.rstrip() + "\n"
    else:
        body = _indent_function_body(clean)
        generation = original_prompt.rstrip() + "\n" + body.rstrip() + "\n"
        completion = body.rstrip() + "\n"

    compile_ok = True
    compile_error = ""
    try:
        compile(generation, "<humaneval_generation>", "exec")
    except SyntaxError as exc:
        compile_ok = False
        compile_error = f"{type(exc).__name__}: {exc}"

    return {
        "raw_response": raw_response or "",
        "clean_response": clean,
        "completion": completion,
        "generation": generation,
        "postprocess_meta": {
            "entry_point": entry_point,
            "kind": kind,
            "fenced_code_extracted": bool(fenced),
            "leading_prose_removed": bool(prose_removed),
            "prompt_echo_removed": bool(prompt_echo_removed),
            "trailing_text_removed": bool(trailing_removed),
            "compile_ok": compile_ok,
            "compile_error": compile_error,
        },
    }
