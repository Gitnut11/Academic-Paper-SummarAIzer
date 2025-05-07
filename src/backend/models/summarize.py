import gc
import logging

# ======== Comment when using Docker ========
# import nltk
# nltk.download("punkt")
# nltk.download("punkt_tab")
# ===========================================

import torch
from nltk.tokenize import sent_tokenize
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
BASE_MODEL = "allenai/led-base-16384"
REPO_NAME = "Mels22/led-scisummnet"
CHUNK_SIZE = 8192
OVERLAP_SIZE = 512
MAX_TARGET_LENGTH = 1024

# Logger setup
logger = logging.getLogger(__name__)


class LEDInference:
    """
    Inference class for summarizing long documents using a LED-based model.
    Splits long input into overlapping chunks, summarizes each, and recombines.
    """

    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP_SIZE,
        max_gen_len=MAX_TARGET_LENGTH,
        batch_size=4,
    ):
        """
        Initializes the LED summarizer with specified chunking and generation settings.

        Args:
            chunk_size (int): Max tokens per chunk for input.
            overlap (int): Token overlap between chunks for smoother transitions.
            max_gen_len (int): Max tokens in each generated summary.
            batch_size (int): Number of chunks processed in parallel.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        self.max_len = max_gen_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading summarization model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        self.model = PeftModel.from_pretrained(base_model, REPO_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on device: {self.device}")

    def _semantic_split(self, text):
        """
        Splits text into overlapping chunks based on sentence boundaries and token limits.

        Args:
            text (str): Input document text.

        Returns:
            List[str]: List of text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > self.chunk_size:
                # Save the current chunk
                chunks.append(" ".join(current_chunk))

                # Handle overlap logic
                if self.chunk_overlap > 0:
                    overlap_sentences = []
                    total_overlap = 0
                    for prev_sentence in reversed(current_chunk):
                        tokens = self.tokenizer.tokenize(prev_sentence)
                        total_overlap += len(tokens)
                        overlap_sentences.insert(0, prev_sentence)
                        if total_overlap >= self.chunk_overlap:
                            break
                    current_chunk = overlap_sentences + [sentence]
                    current_length = total_overlap + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Append final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _batch_summarize(self, texts):
        """
        Summarizes a list of texts in batches.

        Args:
            texts (List[str]): List of input chunks.

        Returns:
            List[str]: List of generated summaries.
        """
        summaries = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.chunk_size,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            # LED requires global attention on the first token
            global_attention_mask = torch.zeros_like(inputs["attention_mask"]).to(self.device)
            global_attention_mask[:, 0] = 1

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    global_attention_mask=global_attention_mask,
                    max_new_tokens=self.max_len,
                    do_sample=False,
                    num_beams=2,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                    length_penalty=1.0,
                    early_stopping=True,
                )

            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend([s.strip() for s in batch_summaries])

            # Free GPU memory
            del inputs, global_attention_mask, outputs
            torch.cuda.empty_cache()

        return summaries

    def infer(self, text):
        """
        Performs summarization on the input text. Handles both short and long documents.

        Args:
            text (str): Full document text.

        Returns:
            str: Final summarized text.
        """
        gc.collect()
        torch.cuda.empty_cache()

        try:
            tokenized_text = self.tokenizer.tokenize(text)
            if len(tokenized_text) <= self.chunk_size:
                logger.info("Summarizing with a single pass (no chunking needed).")
                return self._batch_summarize([text])[0]

            logger.info("Input is long, performing semantic splitting...")
            chunks = self._semantic_split(text)

            logger.info(f"Created {len(chunks)} chunks. Summarizing each chunk...")
            chunk_summaries = self._batch_summarize(chunks)

            combined_summary = " ".join(chunk_summaries)
            logger.info("Re-summarizing the combined chunks...")
            final_summary = self._batch_summarize([combined_summary])[0]

            return final_summary

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise


# Instantiate a global inference object
LED = LEDInference()


def summarize(text):
    """
    Public summarization function using a preloaded LED model.

    Args:
        text (str): Input document text.

    Returns:
        str | None: Summary or None if failed.
    """
    try:
        return LED.infer(text)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return None
