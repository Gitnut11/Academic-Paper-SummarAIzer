import logging
import re
import uuid

import pymupdf4llm
import spacy
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

from utils.config import GEMINI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from utils.prompt import QNA_PROMPT

# ======== Uncomment for local use (Docker handles separately)
# spacy.cli.download("en_core_web_sm")
# ========

# Configure logging
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction using pymupdf4llm."""
    def __init__(self): ...
    def extract_text(self, pdf_path):
        """
        Extracts text from a PDF and returns it in Markdown format.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted Markdown-formatted text.
        """
        try:
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            logger.info(f"Extracted Markdown text ({len(markdown_text)} characters) from '{pdf_path}'")
            return markdown_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from '{pdf_path}': {e}")
            raise


class TextSplitter:
    """Splits Markdown text into semantically coherent chunks using LangChain's SemanticChunker."""

    def __init__(self, model_name="models/text-embedding-004", breakpoint_threshold_type="percentile"):
        self.splitter = SemanticChunker(
            GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=GEMINI_API_KEY),
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def split_text(self, markdown_text):
        """
        Splits text into smaller sections using heading structure and semantic logic.

        Args:
            markdown_text (str): Markdown-formatted document text.

        Returns:
            list[str]: List of chunked strings.
        """
        sections = []
        current_section = []

        for line in markdown_text.splitlines():
            if line.startswith("## "):
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        chunks = []
        for section in sections:
            section_chunks = self.splitter.split_text(section)
            chunks.extend(section_chunks)

        logger.info(f"Split text into {len(chunks)} semantic chunks")
        return chunks


class Embedder:
    """Embeds text using Gemini's embedding model via LangChain."""

    def __init__(self, model_name="models/text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=GEMINI_API_KEY)

    def embed(self, text):
        """Embeds a single query string."""
        try:
            embedding = self.embeddings.embed_query(text)
            logger.info(f"Generated embedding for query of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def embed_documents(self, texts):
        """Embeds a list of document chunks."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embedded {len(texts)} document chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise


class EntityExtractor:
    """Extracts named entities from text using spaCy."""

    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        """
        Identifies named entities in the text.

        Args:
            text (str): A single text chunk.

        Returns:
            list[str]: List of named entities.
        """
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]


class DatabaseConnector:
    """Handles connection to Neo4j and all related database operations."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j database")

    def close(self):
        """Closes the Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")

    def store_embeddings(self, chunks, embeddings, pdf_id, entities_list):
        """
        Stores chunks, embeddings, and entity relationships in Neo4j.

        Args:
            chunks (list[str])
            embeddings (list[list[float]])
            pdf_id (str)
            entities_list (list[list[str]])
        """
        try:
            with self.driver.session() as session:
                for chunk, embedding, entities in zip(chunks, embeddings, entities_list):
                    chunk_id = str(uuid.uuid4())

                    # Create Chunk node with embedding
                    session.run(
                        """
                        CREATE (c:Chunk {id: $id, text: $text, embedding: $embedding, pdf_id: $pdf_id})
                        """,
                        id=chunk_id,
                        text=chunk,
                        embedding=embedding,
                        pdf_id=pdf_id,
                    )

                    # Create/merge entity nodes and relations
                    for entity in entities:
                        session.run(
                            """
                            MERGE (e:Entity {name: $entity})
                            WITH e
                            MATCH (c:Chunk {id: $chunk_id})
                            CREATE (c)-[:MENTIONS]->(e)
                            """,
                            entity=entity,
                            chunk_id=chunk_id,
                        )
            logger.info(f"Stored {len(chunks)} chunks and entities for PDF ID: {pdf_id}")
        except Exception as e:
            logger.error(f"Error storing data in Neo4j: {e}")
            raise

    def retrieve_similar(self, query_embedding, pdf_id, top_k=5):
        """
        Retrieves top-k similar chunks from Neo4j using cosine similarity.

        Returns:
            list[str]: Relevant text chunks.
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    // Find top-k similar chunks
                    MATCH (c:Chunk {pdf_id: $pdf_id})
                    WHERE c.embedding IS NOT NULL
                    WITH c, $query_embedding AS q
                    WITH c, REDUCE(s = 0.0, i IN RANGE(0, SIZE(c.embedding)-1) | s + c.embedding[i] * q[i]) /
                            (SQRT(REDUCE(s = 0.0, x IN c.embedding | s + x*x)) * SQRT(REDUCE(s = 0.0, x IN q | s + x*x))) AS score
                    ORDER BY score DESC
                    LIMIT $top_k
                    WITH collect(c) AS top_chunks
                    // Find related chunks via entities
                    UNWIND top_chunks AS tc
                    MATCH (tc)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Chunk {pdf_id: $pdf_id})
                    WHERE related <> tc
                    WITH top_chunks, collect(DISTINCT related) AS related_chunks
                    // Combine and return unique chunks
                    WITH top_chunks + related_chunks AS all_chunks
                    UNWIND all_chunks AS chunk
                    RETURN DISTINCT chunk.text AS text
                    """,
                    query_embedding=query_embedding,
                    pdf_id=pdf_id,
                    top_k=top_k,
                )
                chunks = [record["text"].strip() for record in result]
                logger.info(f"Retrieved {len(chunks)} total related chunks for PDF ID: {pdf_id}")
                return chunks
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            raise

    def delete_pdf_graph(self, pdf_id):
        """
        Deletes all chunks and orphan entities for a given PDF ID.
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (c:Chunk {pdf_id: $pdf_id})
                    OPTIONAL MATCH (c)-[r:MENTIONS]->(e:Entity)
                    DELETE r, c
                    """,
                    pdf_id=pdf_id
                )
                logger.info(f"Deleted graph nodes for PDF ID: {pdf_id}")

                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                    DELETE e
                    """
                )
                logger.info("Deleted orphaned Entity nodes")
        except Exception as e:
            logger.error(f"Error deleting graph for PDF ID {pdf_id}: {e}")
            raise


class Retriever:
    """Wrapper class to access retrieval logic via the DatabaseConnector."""

    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector

    def retrieve(self, query_embedding, pdf_id, top_k=5):
        """Fetches top-k relevant chunks from the database for a specific PDF."""
        return self.db_connector.retrieve_similar(query_embedding, pdf_id, top_k)


def clean_context(context):
    # 1. Normalize spaces
    context = context.strip()
    context = re.sub(
        r"\s+", " ", context
    )  # collapse all whitespace (including \n) into single spaces

    # 2. Re-split into paragraphs (assuming paragraphs end with a dot + space or line breaks)
    paragraphs = re.split(r"(?<=[.?!])\s+", context)

    # 3. Strip each paragraph individually
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 4. Optionally: Deduplicate paragraphs

    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        if para not in seen:
            unique_paragraphs.append(para)
            seen.add(para)

    # 5. Re-join nicely
    cleaned_context = "\n\n".join(unique_paragraphs)

    return cleaned_context


class Generator:
    """Uses Gemini to generate answers from questions and context."""

    def __init__(self, model_name="models/gemini-2.0-flash"):
        self.llm = GoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)

    def generate_answer(self, question, chunks):
        """
        Generates a structured answer from the LLM.

        Args:
            question (str)
            chunks (list[str])

        Returns:
            dict: Contains question, answer, and references.
        """
        try:
            context = "\n".join(chunks)
            prompt = QNA_PROMPT.format(
                question=question, context=clean_context(context)
            )
            answer = self.llm.invoke(prompt)
            logger.info(f"Generated answer for question: {question}")
            response = {}
            if "Answer" in answer and "References: " in answer:
                ans, ref = answer.split("References: ")
                response.update(
                    {
                        "question": question,
                        "answer": ans.replace("Answer: ", "").strip(),
                        "references": ref.strip(),
                    }
                )
            else:
                response.update(
                    {
                        "question": question,
                        "answer": answer.strip(),
                        "references": None,
                    }
                )

            # return answer
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


class RAGSystem:
    """Main orchestrator for the GraphRAG pipeline."""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder(model_name="models/text-embedding-004")
        self.db_connector = DatabaseConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        self.retriever = Retriever(self.db_connector)
        self.generator = Generator(model_name="models/gemini-2.0-flash")

    def process_pdf(self, pdf_path):
        """Processes a PDF file, extracts entities, and stores embeddings in Neo4j."""
        try:
            pdf_id = str(uuid.uuid4())
            text = self.pdf_processor.extract_text(pdf_path)
            chunks = self.text_splitter.split_text(text)
            # Extract entities for each chunk
            entity_extractor = EntityExtractor()
            entities_list = [
                entity_extractor.extract_entities(chunk) for chunk in chunks
            ]
            embeddings = self.embedder.embed_documents(chunks)
            self.db_connector.store_embeddings(
                chunks, embeddings, pdf_id, entities_list
            )
            # print(f"Stored {len(chunks)} chunks with embeddings and entities")
            logger.info(
                f"Processed and stored embeddings and entities for {pdf_path} with PDF ID ({pdf_id})"
            )
            return pdf_id
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def answer_question(self, question, pdf_id):
        """Answers a user question based on the stored data from a specific PDF."""
        try:
            query_embedding = self.embedder.embed(question)
            relevant_chunks = self.retriever.retrieve(query_embedding, pdf_id, top_k=5)
            answer = self.generator.generate_answer(question, relevant_chunks)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        self.db_connector.close()


# ========= API Interface Functions ==========

PDF_ID = None
RAG_SYSTEM = RAGSystem()



def current_pdf():
    global PDF_ID
    return PDF_ID


def read_pdf(pdf_path: str):
    """Reads a PDF file and processes it."""
    # global PDF_ID
    pdf_id = RAG_SYSTEM.process_pdf(pdf_path)
    return pdf_id


def clear_pdf(pdf_id: str):
    """Clears the processed PDF data."""
    if pdf_id is not None:
        RAG_SYSTEM.db_connector.delete_pdf_graph(pdf_id)
        logger.info("Cleared processed PDF data")
        return True
    else:
        logger.warning("No PDF data to clear")
        return False


def qna(question: str, pdf_id: str):
    """Answers a question based on the processed PDF."""
    if pdf_id is None:
        raise ValueError("No PDF has been processed. Please upload a PDF first.")
    return RAG_SYSTEM.answer_question(question, pdf_id)


# ========= Local Testing Block ==========

if __name__ == "__main__":
    pdf_path = "pro.pdf"
    question = "What is YOLO's loss function?"

    pdf_id = read_pdf(pdf_path)
    print(f"PDF ID: {pdf_id}")

    answer = qna(question, pdf_id)
    print(f"Answer: {answer}")

    clear_pdf(pdf_id)
    RAG_SYSTEM.close()
