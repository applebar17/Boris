import re


def _strip_code_fence(s: str) -> str:
    """Remove triple backtick fences and leading 'json' hints."""
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        # ```json\n{...}\n``` or ```\n{...}\n```
        s = re.sub(r"^```(?:json)?\s*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _extract_top_level_json(s: str) -> tuple[str, list[str], bool]:
    """Best-effort extraction of the first top-level JSON value.

    Returns (substring, remaining_stack, in_string_at_end).
    • remaining_stack empty ⇒ balanced (a full JSON value).
    • If stack non-empty, caller may append matching closers.
    """
    started = False
    stack: list[str] = []
    out_chars: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if not started:
            if ch in "[{":
                started = True
                stack.append(ch)
                out_chars.append(ch)
                continue
            else:
                # skip junk until the first bracket
                continue
        # once started, always append
        out_chars.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "]}":
                if stack and (
                    (stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]")
                ):
                    stack.pop()
                # If we closed the outermost value, stop and ignore any trailing junk
                if not stack:
                    break
    # If we ended inside a string, close it
    if in_string:
        out_chars.append('"')
    return ("".join(out_chars), stack, in_string)


def _sanitize_json_candidate(s: str) -> str:
    """Minor cleanups: remove trailing commas, fix trailing backslash in strings."""
    # Remove trailing commas before closing ] or }
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    # If ends with an odd number of backslashes, add one
    if re.search(r"(?<!\\)\\$", s):
        s += "\\"
    return s


def _close_stack(s: str, stack: list[str]) -> str:
    for opener in reversed(stack):
        s += "}" if opener == "{" else "]"
    return s
