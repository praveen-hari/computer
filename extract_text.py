"""
Extract images from PDF files, run OCR on each image,
and save the extracted text as unit-wise text files.
"""

import os
import sys
import glob
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Configuration
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE, "extracted_text")
IMAGES_DIR = os.path.join(WORKSPACE, "extracted_images")

# Minimum image dimensions to filter out tiny icons/artifacts
MIN_WIDTH = 50
MIN_HEIGHT = 50


def extract_images_from_pdf(pdf_path):
    """Extract all images from a PDF file and return them as PIL Image objects."""
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Filter out tiny images (icons, bullets, etc.)
                if pil_image.width < MIN_WIDTH or pil_image.height < MIN_HEIGHT:
                    continue

                # Convert to RGB if necessary (for OCR compatibility)
                if pil_image.mode not in ("RGB", "L"):
                    pil_image = pil_image.convert("RGB")

                images.append({
                    "image": pil_image,
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "ext": image_ext,
                    "width": pil_image.width,
                    "height": pil_image.height,
                })
            except Exception as e:
                print(f"  ⚠ Could not extract image {img_index + 1} on page {page_num + 1}: {e}")

    doc.close()
    return images


def extract_page_text_from_pdf(pdf_path):
    """Extract embedded text from each page of the PDF (non-image text)."""
    doc = fitz.open(pdf_path)
    page_texts = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            page_texts[page_num + 1] = text.strip()
    doc.close()
    return page_texts


def render_page_as_image(pdf_path, page_num, dpi=300):
    """Render a full PDF page as an image for OCR (fallback for scanned pages)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    # Create a high-resolution pixmap
    zoom = dpi / 72  # 72 is the default DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def ocr_image(pil_image, lang="eng"):
    """Run OCR on a PIL Image and return the extracted text."""
    try:
        text = pytesseract.image_to_string(pil_image, lang=lang)
        return text.strip()
    except Exception as e:
        return f"[OCR Error: {e}]"


def process_pdf(pdf_path, unit_name):
    """Process a single PDF: extract text + images, OCR images, save results."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    print(f"{'='*60}")

    # --- Step 1: Extract embedded text from PDF pages ---
    print("\n📄 Extracting embedded text from PDF pages...")
    page_texts = extract_page_text_from_pdf(pdf_path)
    print(f"   Found embedded text on {len(page_texts)} pages")

    # --- Step 2: Extract images from PDF ---
    print("\n🖼  Extracting images from PDF...")
    images = extract_images_from_pdf(pdf_path)
    print(f"   Found {len(images)} images (filtered by min size {MIN_WIDTH}x{MIN_HEIGHT})")

    # Save extracted images (optional, for reference)
    unit_images_dir = os.path.join(IMAGES_DIR, unit_name)
    os.makedirs(unit_images_dir, exist_ok=True)

    # --- Step 3: OCR on extracted images ---
    print("\n🔍 Running OCR on extracted images...")
    image_ocr_texts = []
    for i, img_data in enumerate(images):
        img = img_data["image"]
        page = img_data["page"]
        idx = img_data["index"]

        # Save image for reference
        img_filename = f"page{page}_img{idx}.{img_data['ext']}"
        img_path = os.path.join(unit_images_dir, img_filename)
        try:
            img.save(img_path)
        except Exception:
            # Convert and retry
            img = img.convert("RGB")
            img_path = os.path.join(unit_images_dir, f"page{page}_img{idx}.png")
            img.save(img_path)

        ocr_text = ocr_image(img)
        if ocr_text:
            image_ocr_texts.append({
                "page": page,
                "index": idx,
                "text": ocr_text,
                "dimensions": f"{img_data['width']}x{img_data['height']}",
            })
            print(f"   ✅ Page {page}, Image {idx} ({img_data['width']}x{img_data['height']}): {len(ocr_text)} chars")
        else:
            print(f"   ⬜ Page {page}, Image {idx}: no text found")

    # --- Step 4: For pages with no embedded text, do full-page OCR ---
    print("\n📝 Running full-page OCR on scanned pages (no embedded text)...")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    fullpage_ocr_texts = {}
    pages_with_text = set(page_texts.keys())
    scanned_pages = [p for p in range(1, total_pages + 1) if p not in pages_with_text]

    if scanned_pages:
        print(f"   Found {len(scanned_pages)} pages without embedded text")
        for page_num in scanned_pages:
            page_img = render_page_as_image(pdf_path, page_num - 1, dpi=300)
            ocr_text = ocr_image(page_img)
            if ocr_text:
                fullpage_ocr_texts[page_num] = ocr_text
                print(f"   ✅ Page {page_num}: {len(ocr_text)} chars via full-page OCR")
            else:
                print(f"   ⬜ Page {page_num}: no text found")
    else:
        print("   All pages have embedded text — skipping full-page OCR")

    # --- Step 5: Combine all text in page order ---
    print("\n📋 Combining extracted text...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combined_text = []
    combined_text.append(f"# {unit_name.upper()} — Extracted Text\n")
    combined_text.append(f"Source: {os.path.basename(pdf_path)}")
    combined_text.append(f"Total pages: {total_pages}")
    combined_text.append(f"Images extracted: {len(images)}")
    combined_text.append(f"Images with OCR text: {len(image_ocr_texts)}")
    combined_text.append("")
    combined_text.append("=" * 60)
    combined_text.append("")

    for page_num in range(1, total_pages + 1):
        combined_text.append(f"\n{'─'*40}")
        combined_text.append(f"## PAGE {page_num}")
        combined_text.append(f"{'─'*40}\n")

        # Embedded text for this page
        if page_num in page_texts:
            combined_text.append("### Embedded Text\n")
            combined_text.append(page_texts[page_num])
            combined_text.append("")

        # Full-page OCR text (for scanned pages)
        if page_num in fullpage_ocr_texts:
            combined_text.append("### Full-Page OCR Text\n")
            combined_text.append(fullpage_ocr_texts[page_num])
            combined_text.append("")

        # OCR text from images on this page
        page_image_texts = [t for t in image_ocr_texts if t["page"] == page_num]
        if page_image_texts:
            combined_text.append("### Image OCR Text\n")
            for img_text in page_image_texts:
                combined_text.append(f"**Image {img_text['index']}** ({img_text['dimensions']}):\n")
                combined_text.append(img_text["text"])
                combined_text.append("")

    # Write output file
    output_path = os.path.join(OUTPUT_DIR, f"{unit_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_text))

    print(f"\n✅ Saved: {output_path}")
    print(f"   Total text length: {len(chr(10).join(combined_text))} characters")

    return output_path


def main():
    """Find all unit PDFs and process them."""
    print("🚀 PDF Image Extraction + OCR Pipeline")
    print(f"   Workspace: {WORKSPACE}")

    # Find all unit PDF files
    pdf_files = sorted(glob.glob(os.path.join(WORKSPACE, "unit*.pdf")))

    if not pdf_files:
        print("❌ No unit*.pdf files found in the workspace!")
        sys.exit(1)

    print(f"   Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")

    output_files = []
    for pdf_path in pdf_files:
        # Derive unit name from filename (e.g., "unit1.pdf" -> "unit1")
        unit_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = process_pdf(pdf_path, unit_name)
        output_files.append(output_path)

    print(f"\n{'='*60}")
    print("🎉 All done! Output files:")
    for f in output_files:
        size = os.path.getsize(f)
        print(f"   📄 {f} ({size:,} bytes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
