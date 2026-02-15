"""Phase 1: Lexical Pre-processor.

Sanitises .ajml files so they can be parsed by xml.etree.ElementTree.
Handles <condition> and <system_prompt> content that may contain
characters invalid in strict XML (comparison operators, angle brackets, etc.).
"""

import re
import xml.etree.ElementTree as ET


def _escape_content(text: str) -> str:
    """Escape XML-invalid characters in condition/prompt content."""
    # Replace bare & that aren't already escaped entities
    text = re.sub(r"&(?!amp;|lt;|gt;|apos;|quot;|#)", "&amp;", text)
    # Replace < but not </ (closing tags)
    text = re.sub(r"<(?!/)", "&lt;", text)
    # Replace >
    text = text.replace(">", "&gt;")
    return text


def _process_tag_content(raw: str, tag_name: str) -> str:
    """Find all occurrences of <tag_name>...</tag_name> and escape their content."""
    pattern = re.compile(
        rf"(<{tag_name}>)(.*?)(</{tag_name}>)",
        re.DOTALL,
    )

    def replacer(match):
        open_tag = match.group(1)
        content = match.group(2)
        close_tag = match.group(3)
        return open_tag + _escape_content(content) + close_tag

    return pattern.sub(replacer, raw)


def preprocess(raw_text: str) -> ET.Element:
    """Pre-process raw AJML text and return an ElementTree root element.

    Steps:
    1. Escape content within <condition> and <system_prompt> tags.
    2. Parse with xml.etree.ElementTree.

    Returns:
        ET.Element: The parsed root element.

    Raises:
        ET.ParseError: If the sanitised text still fails to parse.
    """
    sanitised = raw_text
    sanitised = _process_tag_content(sanitised, "condition")
    sanitised = _process_tag_content(sanitised, "system_prompt")
    return ET.fromstring(sanitised)


def unescape_content(text: str) -> str:
    """Un-escape content that was escaped during preprocessing.

    Used during code generation to restore original characters
    in condition expressions and system prompts.
    """
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    return text


def count_sanitised_conditions(raw_text: str) -> int:
    """Count how many <condition> blocks needed sanitisation."""
    pattern = re.compile(r"<condition>(.*?)</condition>", re.DOTALL)
    count = 0
    for match in pattern.finditer(raw_text):
        content = match.group(1)
        if "<" in content or ">" in content or "&" in content:
            count += 1
    return count
