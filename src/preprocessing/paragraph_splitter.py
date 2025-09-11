import re

import re

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split raw text into clean paragraphs while keeping table blocks intact.
    Table blocks look like:
        [Table_PageX]
        ...
        [/Table_PageX]
    """
    paragraphs = []
    buffer = []
    inside_table = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            # End of paragraph (only if not inside a table)
            if not inside_table and buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue

        # Start of table
        if re.match(r"\[Table_Page\d+\]", stripped):
            inside_table = True
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            buffer.append(stripped)
            continue

        # End of table
        if re.match(r"\[/Table_Page\d+\]", stripped):
            buffer.append(stripped)
            paragraphs.append("\n".join(buffer).strip())
            buffer = []
            inside_table = False
            continue

        # Normal line
        buffer.append(stripped)

    # Flush remaining buffer
    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    return paragraphs
