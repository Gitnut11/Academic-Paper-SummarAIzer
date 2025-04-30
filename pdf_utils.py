import PyPDF2
import pymupdf4llm
from io import BytesIO

def get_sections(file):
    reader = PyPDF2.PdfReader(file)
    result = []
    for item in reader.outline:
        if not isinstance(item, list):
            result.append(item.title)            
    return result

def convert_to_markdown(file):
    return pymupdf4llm.to_markdown(file)

def extract_page_as_binary(input_pdf_path, page_number):
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
    reader = PyPDF2.PdfReader(input_pdf_path)
    return len(reader.pages)


if __name__ == "__main__":
    file = '2006.11371v2.pdf'
    print(get_sections(file))