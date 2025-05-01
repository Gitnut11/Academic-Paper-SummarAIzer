import pymupdf4llm
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
import re
import requests
import json
from dotenv import load_dotenv


load_dotenv()

def setup_gemini():
    """Initialize the Gemini model via LangChain's GoogleGenerativeAI."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=4096
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini: {e}")

def extract_markdown(pdf_path):
    """Convert a PDF file to markdown using pymupdf4llm."""
    try:
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        return markdown_text
    except Exception as e:
        raise Exception(f"Failed to extract markdown from PDF: {e}")

def find_references_section(markdown_text):
    """Isolate the references section to reduce LLM input size."""
    references_start = max(markdown_text.lower().rfind("references"), markdown_text.lower().rfind("bibliography"))
    if references_start == -1:
        return markdown_text
    return markdown_text[references_start:]

def create_llm_prompt():
    """Create a prompt template for extracting citation titles and URLs."""
    response_schemas = [
        ResponseSchema(
            name="citations",
            description="List of citation objects",
            type="array",
            items={
                "type": "object",
                "properties": {
                    "index": {"type": "string", "description": "Citation index (e.g., [1])"},
                    "title": {"type": "string", "description": "Title of the cited paper"},
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs (arXiv or DOI), or null if none",
                        "nullable": True
                    }
                }
            }
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt_template = """
        You are a research assistant tasked with extracting citation information from a markdown-formatted references section of a research paper. For each citation, provide:

        1. The citation index (e.g., [1], [2], etc., as it appears in the references).
        2. The title of the cited paper.
        3. A list of URLs derived from arXiv IDs (e.g., https://arxiv.org/abs/1607.06450) or DOIs (e.g., https://doi.org/10.1000/xyz123). If no arXiv ID or DOI is present, set urls to null.

        Rules:
        - Extract information only from citation entries in the references section.
        - Ignore inline URLs or DOIs not part of citations.
        - Do not include direct URLs (e.g., https://example.com) unless they are arXiv URLs.
        - Return a JSON object with a single key "citations" containing a list of citation objects.
        - Each citation object should have keys: "index", "title", and "urls".

        Here is the markdown text to process:

        ```
        {references_text}
        ```

        {format_instructions}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["references_text"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    return prompt, output_parser

def extract_citations_with_gemini(markdown_text, llm, prompt, output_parser):
    """Use Gemini to extract citation titles and URLs from markdown text."""
    try:
        references_text = find_references_section(markdown_text)
        formatted_prompt = prompt.format(references_text=references_text)
        response = llm.invoke(formatted_prompt)
        parsed_output = output_parser.parse(response)
        # Convert list to dictionary with index as key
        citations_dict = [{"title": citation["title"], "urls": citation["urls"]} for citation in parsed_output.get("citations", [])]
        return citations_dict
    except Exception as e:
        raise Exception(f"Gemini processing failed: {e}")

def get_dictionary_of_urls(pdf_path):
    """Extract citation titles and URLs from a PDF, plus citing papers' URLs."""
    try:
        llm = setup_gemini()
        prompt, output_parser = create_llm_prompt()
        markdown_text = extract_markdown(pdf_path)
        citations_dict = extract_citations_with_gemini(markdown_text, llm, prompt, output_parser)

        print("Citations Dictionary:")
        if not citations_dict:
            print("No citations found.")
        else:
            print(json.dumps(citations_dict, indent=2))
        print("\n")

        return citations_dict
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    a = get_dictionary_of_urls("test.pdf")
    print(a)