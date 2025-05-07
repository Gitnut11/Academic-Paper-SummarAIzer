"""
PDF utility functions for section extraction, page manipulation, and conversion.
"""

from io import BytesIO
import pymupdf4llm
import PyPDF2


def get_sections(file):
    """
    Extracts section titles from a PDF file's outline (table of contents).
    
    Args:
        file: Path to the PDF file or file object.
    
    Returns:
        List of section titles.
    """
    reader = PyPDF2.PdfReader(file)
    result = []
    for item in reader.outline:
        if not isinstance(item, list):
            result.append(item.title)
    return result


def convert_to_markdown(file):
    """
    Converts a PDF to Markdown format using pymupdf4llm.

    Args:
        file: Path to the PDF file or file object.

    Returns:
        str: Markdown representation of the PDF.
    """
    return pymupdf4llm.to_markdown(file)


def extract_page_as_binary(input_pdf_path, page_number):
    """
    Extract a single page from a PDF and return it as binary.

    Args:
        input_pdf_path (str): Path to the PDF file.
        page_number (int): Page number to extract (1-based index).

    Returns:
        bytes: PDF binary data of the extracted page.
    """
    reader = PyPDF2.PdfReader(input_pdf_path)
    writer = PyPDF2.PdfWriter()

    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError("Invalid page number.")

    writer.add_page(reader.pages[page_number - 1])

    output_buffer = BytesIO()
    writer.write(output_buffer)
    pdf_binary = output_buffer.getvalue()
    output_buffer.close()

    return pdf_binary


def get_pdf_page_count(input_pdf_path):
    """
    Returns the number of pages in a PDF file.

    Args:
        input_pdf_path (str): Path to the PDF file.

    Returns:
        int: Total number of pages.
    """
    reader = PyPDF2.PdfReader(input_pdf_path)
    return len(reader.pages)


if __name__ == "__main__":
    file = "2006.11371v2.pdf"
    print(get_sections(file))
