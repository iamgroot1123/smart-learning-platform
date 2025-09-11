from src.preprocessing.preprocessing import preprocess_material

if __name__ == "__main__":
    pdf_path = "data/sample_test.pdf"  
    paragraphs = preprocess_material(pdf_path)

    print(f"✅ Total clean paragraphs extracted: {len(paragraphs)}\n")

    print("📖 First 3 clean paragraphs:\n")
    for i, para in enumerate(paragraphs[:3], start=1):
        print(f"{i}. {para}\n")
