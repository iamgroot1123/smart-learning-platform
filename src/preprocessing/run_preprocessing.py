import argparse
import os
from .preprocessing import preprocess_material

DEFAULT_PDF = "data/raw/sample.pdf"

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing on a PDF")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=DEFAULT_PDF,
        help=f"Path to the PDF file (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--backend",
        choices=["pypdf2", "pymupdf"],
        default="pypdf2",
        help="Backend to use for PDF extraction (default: pypdf2)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        return

    print(f"🔧 Using backend: {args.backend}")
    print(f"📂 Processing file: {args.pdf_path}\n")

    paragraphs = preprocess_material(args.pdf_path, backend=args.backend)

    print(f"✅ Total clean paragraphs extracted: {len(paragraphs)}\n")
    print("📖 First 3 clean paragraphs:\n")
    for i, para in enumerate(paragraphs, start=1):
        print(f"{i}. {para}\n")


if __name__ == "__main__":
    main()
