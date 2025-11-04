import re
from typing import List
from collections import Counter, defaultdict
import fitz  # PyMuPDF
from typing import List


# ---------------------
# Heuristics / helpers
# ---------------------
def is_junk_text(text: str, punct_threshold: float = 0.6) -> bool:
    """Return True for text that's mostly punctuation/dots or decorative."""
    if not text or not text.strip():
        return True
    s = text.strip()
    # remove common unicode whitespace/bullet characters for ratio check
    cleaned = re.sub(r"[\s·•◦▪▫]+", "", s)
    if not cleaned:
        return True
    # punctuation ratio
    punct_count = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
    if punct_count / max(len(s), 1) >= punct_threshold:
        return True
    # repeated dot leaders (like ". . . . .")
    if re.fullmatch(r"[.\s·•─\-]{3,}", s):
        return True
    return False


def token_stats(lines: List[str]):
    """Return simple stats used by table heuristic."""
    num_lines = len(lines)
    numeric_lines = sum(1 for l in lines if any(ch.isdigit() for ch in l))
    comma_lines = sum(1 for l in lines if "," in l)
    tab_lines = sum(1 for l in lines if "\t" in l or re.search(r"\s{2,}", l))
    avg_len = sum(len(l) for l in lines) / max(1, num_lines)
    max_len = max((len(l) for l in lines), default=0)
    return {
        "num_lines": num_lines,
        "numeric_lines": numeric_lines,
        "comma_lines": comma_lines,
        "tab_lines": tab_lines,
        "avg_len": avg_len,
        "max_len": max_len,
    }


def looks_like_table(text: str) -> bool:
    """
    Heuristic to classify a block as part of a table.
    Works best when a block already has multiple rows (csv-like or numeric-heavy).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    num_lines = len(lines)
    numeric_lines = sum(1 for line in lines if any(ch.isdigit() for ch in line))
    comma_lines = sum(1 for line in lines if "," in line)
    tab_lines = sum(1 for line in lines if "\t" in line or "  " in line)

    if numeric_lines / num_lines > 0.5:
        return True
    if comma_lines / num_lines > 0.5:
        return True
    if tab_lines / num_lines > 0.4:
        return True

    avg_len = sum(len(l) for l in lines) / num_lines
    if avg_len < 40 and max(len(l) for l in lines) < avg_len * 1.8:
        return True

    return False


def group_table_blocks(blocks, y_tol: float = 5.0) -> List[str]:
    """
    Try to reconstruct a table from individual blocks using their bounding boxes.
    - Cluster blocks by Y (rows)
    - Sort within row by X (columns)
    - Join cells with ' | '
    Returns list of rows as strings.
    """
    # group by y position
    rows = defaultdict(list)
    for b in blocks:
        x0, y0, x1, y1, text = b
        y_key = round(y0 / y_tol)  # bucket by y
        rows[y_key].append((x0, text))

    # sort rows by y, then sort cells by x
    ordered_rows = []
    for y_key in sorted(rows.keys()):
        cells = sorted(rows[y_key], key=lambda c: c[0])
        row_text = " | ".join(cell[1].strip() for cell in cells if cell[1].strip())
        if row_text:
            ordered_rows.append(row_text)

    return ordered_rows


# ---------------------
# Rich PyMuPDF backend
# ---------------------

def extract_text_pymupdf(
    filepath: str,
    merge_tables: bool = True,
    header_footer_threshold: float = 0.7,
    top_bottom_margin: float = 0.08,
) -> List[str]:
    """
    Extract text and tables from PDF using PyMuPDF while preserving inline order.
    - Removes headers/footers
    - Filters junk
    - Reconstructs multi-block tables using coordinates
    - Wraps tables in [Table_PageX] ... [/Table_PageX]
    """
    doc = fitz.open(filepath)
    num_pages = len(doc)

    raw_blocks_by_page = defaultdict(list)
    header_footer_counter = Counter()
    header_footer_positions = defaultdict(lambda: {"top": 0, "bottom": 0})

    # first pass: collect raw blocks
    for page_num, page in enumerate(doc, start=1):
        page_h = page.rect.height
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], (b[4] or "").strip()
            if not text:
                continue
            top_norm = y0 / max(page_h, 1)
            bottom_norm = y1 / max(page_h, 1)
            block = (x0, y0, x1, y1, text)
            raw_blocks_by_page[page_num].append(block)

            # candidate headers/footers
            if len(text.split()) <= 8:
                if top_norm <= top_bottom_margin:
                    header_footer_counter[text] += 1
                    header_footer_positions[text]["top"] += 1
                elif bottom_norm >= (1 - top_bottom_margin):
                    header_footer_counter[text] += 1
                    header_footer_positions[text]["bottom"] += 1

    # decide headers/footers
    hf_threshold = max(1, int(num_pages * header_footer_threshold))
    headers_footers = set()
    for txt, count in header_footer_counter.items():
        top_count = header_footer_positions[txt]["top"]
        bottom_count = header_footer_positions[txt]["bottom"]
        if count >= hf_threshold or (top_count + bottom_count) >= hf_threshold:
            headers_footers.add(txt)

    # second pass: build content
    content_blocks: List[str] = []
    for page_num in range(1, num_pages + 1):
        blocks = raw_blocks_by_page.get(page_num, [])
        table_buffer: List[str] = []
        candidate_table_blocks = []

        for blk in blocks:
            x0, y0, x1, y1, text = blk
            if not text.strip():
                continue

            if text in headers_footers:
                continue
            if re.fullmatch(r"\d{1,3}", text) and (
                (y0 / doc[page_num - 1].rect.height) < top_bottom_margin
                or (y1 / doc[page_num - 1].rect.height) > (1 - top_bottom_margin)
            ):
                continue

            # --- NEW TABLE DETECTION ---
            if merge_tables:
                if looks_like_table(text):
                    rows = [re.sub(r"\s{2,}", " | ", r).strip() for r in text.splitlines() if r.strip()]
                    table_buffer.extend(rows)
                    continue
                # also catch very short blocks (possible cell text)
                if len(text.split()) <= 4 and len(text) <= 20:
                    candidate_table_blocks.append(blk)
                    continue

            # flush candidate cell-block table if any
            if candidate_table_blocks:
                rows = group_table_blocks(candidate_table_blocks)
                if len(rows) > 1:
                    table_tag = f"[Table_Page{page_num}]\n" + "\n".join(rows) + f"\n[/Table_Page{page_num}]"
                    content_blocks.append(table_tag)
                else:
                    content_blocks.extend(r[4] for r in candidate_table_blocks)
                candidate_table_blocks = []

            # flush table buffer if any
            if table_buffer:
                table_tag = f"[Table_Page{page_num}]\n" + "\n".join(table_buffer) + f"\n[/Table_Page{page_num}]"
                content_blocks.append(table_tag)
                table_buffer = []

            # add normal text
            content_blocks.append(text)

        # end of page flushes
        if candidate_table_blocks:
            rows = group_table_blocks(candidate_table_blocks)
            if len(rows) > 1:
                table_tag = f"[Table_Page{page_num}]\n" + "\n".join(rows) + f"\n[/Table_Page{page_num}]"
                content_blocks.append(table_tag)
            else:
                content_blocks.extend(r[4] for r in candidate_table_blocks)
            candidate_table_blocks = []

        if table_buffer:
            table_tag = f"[Table_Page{page_num}]\n" + "\n".join(table_buffer) + f"\n[/Table_Page{page_num}]"
            content_blocks.append(table_tag)
            table_buffer = []

    # cleanup
    final_blocks = []
    for b in content_blocks:
        b_clean = re.sub(r"[ \t]+", " ", b)
        b_clean = re.sub(r"\n{3,}", "\n\n", b_clean).strip()
        if not b_clean:
            continue
        if len(b_clean) <= 2 and re.fullmatch(r"[^\w\s]+", b_clean):
            continue
        final_blocks.append(b_clean)

    return final_blocks


# ---------------------
# Router / unified API
# ---------------------
def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from PDF file using PyMuPDF.
    Returns a list[str] of text blocks.
    """
    return extract_text_pymupdf(pdf_path, merge_tables=True)
