import PyPDF2
import pymupdf4llm

def get_sections(file):
    reader = PyPDF2.PdfReader(file)
    result = []
    for item in reader.outline:
        if not isinstance(item, list):
            result.append(item.title)            
    return result

def convert_to_markdown(file):
    return pymupdf4llm.to_markdown(file)

if __name__ == "__main__":
    file = '2006.11371v2.pdf'
    print(get_sections(file))