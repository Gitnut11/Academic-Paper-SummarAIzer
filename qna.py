import logging
import os
import re
import uuid

import fitz
import spacy
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from prompt import QNA_PROMPT

#======== Comment when using docker
# spacy.cli.download("en_core_web_sm")
#========

# ==================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class PDFProcessor:
    def __init__(self):
        # Set up basic logging
        logging.basicConfig(level=logging.INFO)

    def extract_text(self, pdf_path, margin=0.1):
        """
        Extract text from a PDF, handling two-column layouts with PyMuPDF.

        Args:
            pdf_path (str): Path to the PDF file.
            margin (float): Fraction of page height to exclude as top/bottom margins.

        Returns:
            str: Extracted text with left column followed by right column.
        """
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            full_text = ""

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                height = page.rect.height
                width = page.rect.width

                # Define content area (exclude top and bottom margins)
                top = margin * height
                bottom = (1 - margin) * height

                # Get text blocks
                blocks = page.get_text("blocks")

                # Filter blocks within the content area
                content_blocks = [
                    block for block in blocks if block[1] >= top and block[3] <= bottom
                ]

                if not content_blocks:
                    continue

                # Determine column layout by splitting at the page midpoint
                x_coords = [block[0] for block in content_blocks]
                mid_point = width / 2
                left_blocks = [
                    block for block in content_blocks if block[0] < mid_point
                ]
                right_blocks = [
                    block for block in content_blocks if block[0] >= mid_point
                ]

                if left_blocks and right_blocks:
                    # Two-column layout detected
                    # Sort blocks in each column by y-coordinate (top to bottom)
                    left_blocks.sort(key=lambda b: b[1])
                    right_blocks.sort(key=lambda b: b[1])

                    # Extract text from left and right columns
                    left_text = "\n".join([block[4] for block in left_blocks])
                    right_text = "\n".join([block[4] for block in right_blocks])

                    # Combine columns
                    page_text = left_text + "\n" + right_text
                else:
                    # Single-column layout
                    content_blocks.sort(key=lambda b: b[1])
                    page_text = "\n".join([block[4] for block in content_blocks])

                full_text += page_text + "\n"

            doc.close()
            logging.info(
                f"Extracted text from {pdf_path}, length: {len(full_text)} characters"
            )
            return full_text.strip()

        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            raise


class TextSplitter:
    def __init__(
        self,
        model_name="models/text-embedding-004",
        breakpoint_threshold_type="percentile",
    ):
        self.splitter = SemanticChunker(
            GoogleGenerativeAIEmbeddings(
                model=model_name, google_api_key=os.getenv("GEMINI_API_KEY")
            ),
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def split_text(self, text):
        chunks = self.splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")
        return chunks


class Embedder:
    """Generates embeddings using the Gemini embedding model via LangChain."""

    def __init__(self, model_name="models/text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name, google_api_key=GEMINI_API_KEY
        )

    def embed(self, text):
        """Embeds a single text."""
        try:
            embedding = self.embeddings.embed_query(text)
            logging.info(f"Embedded query text, length: {len(embedding)}")
            return embedding
        except Exception as e:
            logging.error(f"Error embedding query: {e}")
            raise

    def embed_documents(self, texts):
        """Embeds multiple texts."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logging.info(f"Embedded {len(texts)} documents")
            return embeddings
        except Exception as e:
            logging.error(f"Error embedding documents: {e}")
            raise


class EntityExtractor:
    """Extracts named entities from text using spaCy for GraphRAG."""

    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        """Extracts named entities from a text chunk."""
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities


class DatabaseConnector:
    """Manages connection and operations with Neo4j Aura database."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.info("Connected to Neo4j Aura")

    def close(self):
        """Closes the database connection."""
        self.driver.close()
        logging.info("Closed Neo4j Aura connection")

    def store_embeddings(self, chunks, embeddings, pdf_id, entities_list):
        """Stores text chunks, embeddings, and entities in Neo4j with a unique PDF ID."""
        try:
            with self.driver.session() as session:
                for chunk, embedding, entities in zip(
                    chunks, embeddings, entities_list
                ):
                    chunk_id = str(uuid.uuid4())
                    session.run(
                        """
                        CREATE (c:Chunk {id: $id, text: $text, embedding: $embedding, pdf_id: $pdf_id})
                        """,
                        id=chunk_id,
                        text=chunk,
                        embedding=embedding,
                        pdf_id=pdf_id,
                    )
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
            logging.info(
                f"Stored {len(chunks)} chunks with embeddings and entities for PDF ID: {pdf_id}"
            )
        except Exception as e:
            logging.error(f"Error storing embeddings and entities in Neo4j: {e}")
            raise

    def retrieve_similar(self, query_embedding, pdf_id, top_k=5):
        """Retrieves top-k similar chunks and expands context using GraphRAG from Neo4j."""
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
                logging.info(
                    f"Retrieved {len(chunks)} chunks (including related) for PDF ID: {pdf_id}"
                )
                return chunks
        except Exception as e:
            logging.error(f"Error retrieving similar chunks: {e}")
            raise

    def delete_pdf_graph(self, pdf_id):
        """Deletes all Chunk nodes and their relationships for a given PDF ID, and optionally deletes orphaned Entity nodes."""
        try:
            with self.driver.session() as session:
                # Delete Chunk nodes and their MENTIONS relationships for the given pdf_id
                session.run(
                    """
                    MATCH (c:Chunk {pdf_id: $pdf_id})
                    OPTIONAL MATCH (c)-[r:MENTIONS]->(e:Entity)
                    DELETE r, c
                    """,
                    pdf_id=pdf_id,
                )
                logging.info(
                    f"Deleted Chunk nodes and relationships for PDF ID: {pdf_id}"
                )

                # Optional: Delete orphaned Entity nodes that no longer have any incoming MENTIONS relationships
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                    DELETE e
                    """
                )
                logging.info("Deleted orphaned Entity nodes")
        except Exception as e:
            logging.error(f"Error deleting PDF graph for ID {pdf_id}: {e}")
            raise


class Retriever:
    """Retrieves relevant document chunks based on query embedding."""

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
    """Generates answers using the Gemini-2.0-flash model via LangChain."""

    def __init__(self, model_name="models/gemini-2.0-flash"):
        self.llm = GoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)

    def generate_answer(self, question, chunks):
        """Generates an answer based on the question and retrieved chunks."""
        try:
            context = "\n".join(chunks)
            prompt = QNA_PROMPT.format(
                question=question, context=clean_context(context)
            )
            answer = self.llm.invoke(prompt)
            logging.info(f"Generated answer for question: {question}")
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
            logging.error(f"Error generating answer: {e}")
            raise


class RAGSystem:
    """Orchestrates the RAG pipeline using LangGraph-like workflow with GraphRAG."""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder(model_name="models/text-embedding-004")
        self.db_connector = DatabaseConnector(
            uri=NEO4J_URI,
            user=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
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
            logging.info(
                f"Processed and stored embeddings and entities for {pdf_path} with PDF ID: {pdf_id}"
            )
            return pdf_id
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def answer_question(self, question, pdf_id):
        """Answers a user question based on the stored data from a specific PDF."""
        try:
            query_embedding = self.embedder.embed(question)
            relevant_chunks = self.retriever.retrieve(query_embedding, pdf_id, top_k=5)
            answer = self.generator.generate_answer(question, relevant_chunks)
            return answer
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        self.db_connector.close()




PDF_ID = None
RAG_SYSTEM = RAGSystem()

def current_pdf():
    global PDF_ID
    return PDF_ID

def read_pdf(pdf_path: str):
    """Reads a PDF file and processes it."""
    global PDF_ID
    PDF_ID = RAG_SYSTEM.process_pdf(pdf_path)
    return PDF_ID


def clear_pdf():
    """Clears the processed PDF data."""
    global PDF_ID
    if PDF_ID is not None:
        RAG_SYSTEM.db_connector.delete_pdf_graph(PDF_ID)
        PDF_ID = None
        logging.info("Cleared processed PDF data")
        return True
    else:
        logging.warning("No PDF data to clear")
        return False


def qna(question):
    """Answers a question based on the processed PDF."""
    if PDF_ID is None:
        raise ValueError("No PDF has been processed. Please upload a PDF first.")
    return RAG_SYSTEM.answer_question(question, PDF_ID)


# Này để test th, import thì ko chạy đâu đừng lo
if __name__ == "__main__":
    PDF_ID = None
    RAG_SYSTEM = RAGSystem()

    # Example usage
    pdf_path = "pro.pdf"  # Replace with your PDF file path
    question = "What is YOLO's loss function?"

    # Read and process the PDF
    pdf_id = read_pdf(pdf_path=pdf_path)
    print(f"Processed PDF ID: {pdf_id}")

    # Ask a question
    answer = qna(question=question)
    print(f"Answer: {answer}")

    # Clear the processed PDF data
    clear_pdf()

    RAG_SYSTEM.close()
