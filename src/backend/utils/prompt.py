# Store all the prompt templates here

# Question-Answering Prompt
QNA_PROMPT = """
You are a professional AI assistant.

Below are several reference paragraphs provided as context. 
Based solely on the information contained in these paragraphs, answer the user's question.

**Instructions:**
- Only use information from the provided paragraphs. Do not make up or hallucinate any additional information.
- Answer using the same language as the user's question, if the question's language and the reference paragraphs' language are different, translate the answer to the user's language.
- Include the references in the answer. Use the format: `References: [{{excerpt}}]` where `{{excerpt}}` is the relevant excerpt from the reference paragraphs.
- If you use multiple references, list them together (e.g., `[{{excerpt1}}][{{excerpt2}}]`).
- If you cannot find sufficient information in the provided paragraphs, respond exactly: "No relevant information found in the provided paragraphs."
- If the answer has math formulas, use Markdown format for math (e.g., `$E=mc^2$`).

---

**Reference Paragraphs:**
{context}
---

**User Question:**
{question}

---

**Answer Format Example:**
Answer: {{your answer here}}
References: [{{excerpt1}}][{{excerpt2}}]
""".strip()

# ==============================================================================

# Safety Classification Prompt
SAFETY_PROMPT = """
You are a helpful and precise assistant tasked with classifying questions based on their safety.

Given the following question:
{question}

Your task is to analyze the question carefully and classify it strictly as either "safe" or "unsafe".

Please respond with only one word: either "safe" or "unsafe", without any additional explanation or text.
""".strip()

# ==============================================================================

# Help Message
HELPER = """
- `/summarize` + text or [sections_id]: to summarize text or sections.
- `/sections`: show list of detected sections.
- `/help`: show this help message.
Other inputs are treated as Q&A prompt for this paper!
""".strip()

# ==============================================================================
CITATION_PROMPT = """
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
""".strip()
