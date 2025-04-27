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
