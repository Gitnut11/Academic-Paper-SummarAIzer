import gc
import logging

import nltk
import torch
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

BASE_MODEL = "allenai/led-base-16384"
REPO_NAME = "Mels22/led-scisummnet"

CHUNK_SIZE = 8192
OVERLAP_SIZE = 512
MAX_TARGET_LENGTH = 512


class LEDInference:
    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP_SIZE,
        max_gen_len=MAX_TARGET_LENGTH,
        batch_size=4,  # New param: safe batch size
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        self.max_len = max_gen_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        self.model = PeftModel.from_pretrained(base_model, REPO_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _semantic_split(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))

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

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _batch_summarize(self, texts):
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

            del inputs, global_attention_mask, outputs
            torch.cuda.empty_cache()

        return summaries

    def infer(self, text):
        gc.collect()
        torch.cuda.empty_cache()

        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) <= self.chunk_size:
            # Direct summarization
            return self._batch_summarize([text])[0]

        # Otherwise, semantic split
        chunks = self._semantic_split(text)

        # Batch summarize chunks
        chunk_summaries = self._batch_summarize(chunks)

        combined_summary = " ".join(chunk_summaries)

        # Final summarization
        final_summary = self._batch_summarize([combined_summary])[0]

        return final_summary


class SmrRequest(BaseModel):
    text: str


LED = LEDInference()


def summarize(request: SmrRequest):
    try:
        summary = LED.infer(request.text)
        return {"summary": summary}
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
