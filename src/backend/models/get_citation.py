import logging

# Converts PDF to markdown optimized for LLM input
import pymupdf4llm

# LangChain modules for output formatting and prompt templating
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

# Import environment config and custom prompt template
from utils.config import GEMINI_API_KEY
from utils.prompt import CITATION_PROMPT

# Initialize module-level logger
logger = logging.getLogger(__name__)


def setup_gemini():
    """
    Initialize the Gemini 1.5 Pro model using the GoogleGenerativeAI wrapper from LangChain.

    Raises:
        ValueError: if GEMINI_API_KEY is not set.
        Exception: for any other initialization errors.
    Returns:
        GoogleGenerativeAI: LLM object ready for use.
    """
    try:
        api_key = GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. Please set it."
            )
        # Initialize Gemini with desired parameters
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=4096,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
        raise Exception(f"Failed to initialize Gemini: {e}")


def extract_markdown(pdf_path):
    """
    Convert a PDF file into markdown for LLM-friendly text representation.

    Args:
        pdf_path (str): Path to the input PDF.

    Returns:
        str: Markdown representation of the PDF.
    """
    try:
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        return markdown_text
    except Exception as e:
        logger.error(f"Failed to extract markdown from PDF: {str(e)}")
        raise Exception(f"Failed to extract markdown from PDF: {e}")


def find_references_section(markdown_text):
    """
    Extract just the references/bibliography section to reduce prompt size.

    Args:
        markdown_text (str): Full markdown of the PDF.

    Returns:
        str: Substring starting from 'References' or 'Bibliography' if found.
    """
    references_start = max(
        markdown_text.lower().rfind("references"),
        markdown_text.lower().rfind("bibliography"),
    )
    if references_start == -1:
        # If no section found, return full text
        return markdown_text
    return markdown_text[references_start:]


def create_llm_prompt():
    """
    Create a LangChain prompt and structured output parser for extracting citations.

    Returns:
        tuple: (PromptTemplate, StructuredOutputParser)
    """
    response_schemas = [
        ResponseSchema(
            name="citations",
            description="List of citation objects",
            type="array",
            items={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "Citation index (e.g., [1])",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the cited paper",
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs (arXiv or DOI), or null if none",
                        "nullable": True,
                    },
                },
            },
        )
    ]
    # Create a parser to enforce structured JSON-like output
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Bind the parser’s format instructions into the prompt
    prompt = PromptTemplate(
        template=CITATION_PROMPT,
        input_variables=["references_text"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    return prompt, output_parser


def extract_citations_with_gemini(markdown_text, llm, prompt, output_parser):
    """
    Use Gemini to extract citations (titles and URLs) from markdown text.

    Args:
        markdown_text (str): Full markdown of the PDF.
        llm (GoogleGenerativeAI): Initialized Gemini model.
        prompt (PromptTemplate): Prompt template with formatting.
        output_parser (StructuredOutputParser): Schema-based parser.

    Returns:
        list[dict]: List of citation dictionaries with title and URL.
    """
    try:
        references_text = find_references_section(markdown_text)
        formatted_prompt = prompt.format(references_text=references_text)
        response = llm.invoke(formatted_prompt)
        parsed_output = output_parser.parse(response)

        # Convert structured output to list of dicts
        citations_dict = [
            {
                "title": citation["title"],
                "url": (
                    citation["urls"][0]
                    if citation["urls"] is not None and len(citation["urls"]) > 0
                    else "null"
                ),
            }
            for citation in parsed_output.get("citations", [])
        ]
        return citations_dict
    except Exception as e:
        logger.error(f"Gemini processing failed: {str(e)}")
        raise Exception(f"Gemini processing failed: {e}")


def get_list_of_urls(pdf_path):
    """
    Top-level function to extract citation titles and URLs from a PDF.

    Args:
        pdf_path (str): Path to the PDF.

    Returns:
        list[dict]: Citation entries with titles and resolved URLs.
    """
    try:
        llm = setup_gemini()
        prompt, output_parser = create_llm_prompt()
        markdown_text = extract_markdown(pdf_path)
        citations_dict = extract_citations_with_gemini(
            markdown_text, llm, prompt, output_parser
        )
        return citations_dict
    except Exception as e:
        logger.error(f"Error: {e}")
        return []


# Run as a standalone script
if __name__ == "__main__":
    a = get_list_of_urls("test.pdf")
    print(a)
