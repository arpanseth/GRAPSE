import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from classes import Pipeline
from langchain.prompts import PromptTemplate
from argparse import ArgumentParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableMap
)
from langchain_core.messages import AIMessage, HumanMessage
import warnings
import re
import base64
from langchain.schema import Document
from datetime import datetime
import html

warnings.filterwarnings("ignore")

# Load the environment variables used for evaluation
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

__all__ = ["MainPipeline"]

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the Author, Algorithm, Software, Solver, Formula, Institution, and Benchmark entities that" 
        "appear in the text",
    )

class CitationInfo(BaseModel):
    """Citation information for a retrieved document."""
    paper_title: str
    paper_authors: str
    paper_year: str
    document_id: str
    content: str
    retrieval_source: str  # "semantic", "entity", or "figure"
    entity_matched: str = None  # For entity-based retrieval

class FigureInfo(BaseModel):
    """Information for a retrieved figure."""
    figure_id: str
    description: str
    image_path: str
    image_base64: str = ""
    paper_title: str = "Unknown Title"
    paper_authors: str = "Unknown Authors"
    paper_year: str = "Unknown"
    citation_number: int = None  # Track which citation this figure belongs to

class MainPipeline(Pipeline):

    def __init__(self, model_name):
        self.model_name = model_name
        self.retrieved_docs = None
        self.embeddings = OpenAIEmbeddings()
        self.citation_counter = 0
        self.unique_papers = {}  # Track unique papers for bibliography
        self.citation_figures = {}  # Track figures by citation number: {citation_num: [figure_paths]}

    def format_docs(self, docs):
        """
        Utility to format docs with citation information.
        """
        if self.retrieved_docs is None:
            self.retrieved_docs = []
        
        formatted_content = []
        for doc in docs:
            if isinstance(doc, dict) and 'content' in doc:
                self.retrieved_docs.append(doc)
                formatted_content.append(doc['content'])
            else:
                self.retrieved_docs.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                formatted_content.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
        
        return "\n\n".join(formatted_content)

    def extract_entities(self, question: str) -> List[str]:
        """
        Extract PSE-specific entities from the question using the LLM.
        """
        try:
            entities = self.entity_chain.invoke({"question": question})
            return entities.names
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def remove_lucene_chars_preserving_hyphen(self, input_str: str) -> str:
        """
        Removes Lucene special characters except hyphens.
        Escapes hyphens to prevent them from being treated as operators.
        """
        lucene_special_chars = r'([+\!(){}\[\]^"~*?:\\/])'
        cleaned = re.sub(lucene_special_chars, '', input_str)
        # Escape hyphens
        cleaned = re.sub(r'(-)', r'\\\1', cleaned)
        return cleaned

    def generate_full_text_query(self, input_str: str) -> str:
        """
        Generate a full-text search query for a given input string.
        """
        cleaned_input = self.remove_lucene_chars_preserving_hyphen(input_str)
        words = [el for el in cleaned_input.split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        return full_text_query.strip()

    def semantic_document_retrieval(self, question: str, k: int = 10) -> List[CitationInfo]:
        """
        Perform semantic document retrieval with paper metadata for citations.
        """
        try:
            # Custom retrieval query that includes Paper information
            retrieval_query = """
                MATCH (n:Document)<-[:CONTAINS]-(p:Paper)
                WITH n, p, vector.similarity.cosine(n.embedding, $embedding) AS score
                WHERE score > 0.6
                RETURN n.text AS text,
                       n.id AS document_id,
                       p.title AS paper_title,
                       p.authors AS paper_authors,
                       COALESCE(toString(p.year), 'Unknown') AS paper_year,
                       score
                ORDER BY score DESC
                LIMIT $k
            """
            
            embedding = self.embeddings.embed_query(question)
            results = self.graph.query(retrieval_query, {
                "embedding": embedding,
                "k": k
            })
            
            citations = []
            for result in results:
                # Try to extract year if missing
                paper_year = str(result["paper_year"]) if result["paper_year"] else "Unknown"
                if paper_year == "Unknown":
                    paper_year = self.extract_year_from_title_or_content(
                        result["paper_title"] or "Unknown Title", 
                        result["text"]
                    )
                
                citation = CitationInfo(
                    paper_title=result["paper_title"] or "Unknown Title",
                    paper_authors=result["paper_authors"] or "Unknown Authors", 
                    paper_year=paper_year,
                    document_id=result["document_id"],
                    content=result["text"],
                    retrieval_source="semantic"
                )
                citations.append(citation)
                
                # Track unique papers for bibliography (deduplicate by title+authors)
                paper_base_key = f"{citation.paper_title}_{citation.paper_authors}"
                paper_key = f"{citation.paper_title}_{citation.paper_authors}_{citation.paper_year}"
                
                # Check if this is a duplicate paper with different year
                existing_key = None
                for existing_paper_key in self.unique_papers.keys():
                    if existing_paper_key.startswith(paper_base_key + "_"):
                        existing_key = existing_paper_key
                        break
                
                if existing_key:
                    # Update to use the most recent year if we have a better one
                    existing_year = self.unique_papers[existing_key]["year"]
                    current_year = citation.paper_year
                    
                    # Use most recent year (or replace Unknown with actual year)
                    if (current_year != "Unknown" and existing_year == "Unknown") or \
                       (current_year != "Unknown" and existing_year != "Unknown" and current_year > existing_year):
                        # Update existing entry with newer year
                        citation_num = self.unique_papers[existing_key]["number"]
                        del self.unique_papers[existing_key]
                        self.unique_papers[paper_key] = {
                            "number": citation_num,
                            "title": citation.paper_title,
                            "authors": citation.paper_authors,
                            "year": citation.paper_year
                        }
                    # Otherwise keep the existing one and reuse its number
                    else:
                        citation_num = self.unique_papers[existing_key]["number"]
                        self.unique_papers[paper_key] = {
                            "number": citation_num,
                            "title": citation.paper_title,
                            "authors": citation.paper_authors,
                            "year": citation.paper_year
                        }
                else:
                    # New unique paper
                    self.citation_counter += 1
                    self.unique_papers[paper_key] = {
                        "number": self.citation_counter,
                        "title": citation.paper_title,
                        "authors": citation.paper_authors,
                        "year": citation.paper_year
                    }
            
            return citations
            
        except Exception as e:
            print(f"Error in semantic document retrieval: {e}")
            return []

    def entity_based_retrieval(self, question: str, k: int = 5) -> List[CitationInfo]:
        """
        Perform entity-based retrieval with document traversal and paper metadata.
        """
        try:
            entities = self.extract_entities(question)
            if not entities:
                return []
                
            citations = []
            
            for entity in entities[:k]:  # Limit to top 3 entities to avoid noise
                query = self.generate_full_text_query(entity)
                if not query:
                    continue
                    
                # Entity-based retrieval query
                entity_query = """
                    CALL db.index.fulltext.queryNodes('entity', $query, {limit: 3})
                    YIELD node, score
                    MATCH (node)<-[:MENTIONS]-(d:Document)<-[:CONTAINS]-(p:Paper)
                    RETURN DISTINCT d.text AS text,
                           d.id AS document_id,
                           node.id AS entity_matched,
                           labels(node)[0] AS entity_type,
                           p.title AS paper_title,
                           p.authors AS paper_authors,
                           COALESCE(toString(p.year), 'Unknown') AS paper_year,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """
                
                try:
                    results = self.graph.query(entity_query, {
                        "query": query,
                        "k": k
                    })
                    
                    for result in results:
                        # Try to extract year if missing
                        paper_year = str(result["paper_year"]) if result["paper_year"] else "Unknown"
                        if paper_year == "Unknown":
                            paper_year = self.extract_year_from_title_or_content(
                                result["paper_title"] or "Unknown Title", 
                                result["text"]
                            )
                        
                        citation = CitationInfo(
                            paper_title=result["paper_title"] or "Unknown Title",
                            paper_authors=result["paper_authors"] or "Unknown Authors",
                            paper_year=paper_year,
                            document_id=result["document_id"],
                            content=result["text"],
                            retrieval_source="entity",
                            entity_matched=result["entity_matched"]
                        )
                        citations.append(citation)
                        
                        # Track unique papers for bibliography (deduplicate by title+authors)
                        paper_base_key = f"{citation.paper_title}_{citation.paper_authors}"
                        paper_key = f"{citation.paper_title}_{citation.paper_authors}_{citation.paper_year}"
                        
                        # Check if this is a duplicate paper with different year
                        existing_key = None
                        for existing_paper_key in self.unique_papers.keys():
                            if existing_paper_key.startswith(paper_base_key + "_"):
                                existing_key = existing_paper_key
                                break
                        
                        if existing_key:
                            # Update to use the most recent year if we have a better one
                            existing_year = self.unique_papers[existing_key]["year"]
                            current_year = citation.paper_year
                            
                            # Use most recent year (or replace Unknown with actual year)
                            if (current_year != "Unknown" and existing_year == "Unknown") or \
                               (current_year != "Unknown" and existing_year != "Unknown" and current_year > existing_year):
                                # Update existing entry with newer year
                                citation_num = self.unique_papers[existing_key]["number"]
                                del self.unique_papers[existing_key]
                                self.unique_papers[paper_key] = {
                                    "number": citation_num,
                                    "title": citation.paper_title,
                                    "authors": citation.paper_authors,
                                    "year": citation.paper_year
                                }
                            # Otherwise keep the existing one and reuse its number
                            else:
                                citation_num = self.unique_papers[existing_key]["number"]
                                self.unique_papers[paper_key] = {
                                    "number": citation_num,
                                    "title": citation.paper_title,
                                    "authors": citation.paper_authors,
                                    "year": citation.paper_year
                                }
                        else:
                            # New unique paper
                            self.citation_counter += 1
                            self.unique_papers[paper_key] = {
                                "number": self.citation_counter,
                                "title": citation.paper_title,
                                "authors": citation.paper_authors,
                                "year": citation.paper_year
                            }
                            
                except Exception as e:
                    print(f"Error querying entity {entity}: {e}")
                    continue
                    
            return citations
            
        except Exception as e:
            print(f"Error in entity-based retrieval: {e}")
            return []

    def extract_year_from_title_or_content(self, paper_title: str, content: str) -> str:
        """
        Extract year from paper title or content using regex patterns.
        """
        import re
        
        # Common year patterns in academic papers
        year_patterns = [
            r'\b(20[0-2]\d)\b',    # Years 2000-2029
            r'\b(19[89]\d)\b',     # Years 1980-1999  
            r'\((\d{4})\)',        # Years in parentheses
            r'(\d{4})',            # Any 4-digit number that could be a year
        ]
        
        # Try to find year in title first
        for pattern in year_patterns:
            matches = re.findall(pattern, paper_title)
            for match in matches:
                year = match if isinstance(match, str) else match[0] if match else None
                if year and 1980 <= int(year) <= 2030:  # Reasonable range for academic papers
                    return year
        
        # Try to find year in content (first 1000 chars)
        content_sample = content[:1000] if content else ""
        for pattern in year_patterns:
            matches = re.findall(pattern, content_sample)
            for match in matches:
                year = match if isinstance(match, str) else match[0] if match else None
                if year and 1980 <= int(year) <= 2030:  # Reasonable range for academic papers
                    return year
        
        return "Unknown"

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        """
        try:
            # Fix the path if it contains the old GRAPSE folder name
            corrected_path = image_path.replace("/Users/aseth/Projects/GRAPSE/", "/Users/aseth/Projects/GRAPSE_Dev/")
            
            with open(corrected_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {corrected_path}: {e}")
            return None

    def figure_retrieval(self, question: str, k: int = 3) -> List[FigureInfo]:
        """
        Perform vector search on Figure nodes to find relevant figures.
        """
        try:
            # Custom retrieval query for figures with paper information
            figure_query = """
                MATCH (f:Figure)
                WITH f, vector.similarity.cosine(f.embedding, $embedding) AS score
                WHERE score > 0.5
                OPTIONAL MATCH (f)-[:MENTIONS]-(d:Document)<-[:CONTAINS]-(p:Paper)
                RETURN f.id AS figure_id,
                       f.text AS description,
                       f.path AS image_path,
                       p.title AS paper_title,
                       p.authors AS paper_authors,
                       COALESCE(toString(p.year), 'Unknown') AS paper_year,
                       score
                ORDER BY score DESC
                LIMIT $k
            """
            
            embedding = self.embeddings.embed_query(question)
            results = self.graph.query(figure_query, {
                "embedding": embedding,
                "k": k
            })
            
            figures = []
            for result in results:
                # Try to extract year if missing
                paper_year = result["paper_year"] if result["paper_year"] else "Unknown"
                if paper_year == "Unknown" and result["description"]:
                    paper_year = self.extract_year_from_title_or_content(
                        result["paper_title"] or "Unknown Title", 
                        result["description"]
                    )
                
                # Encode image to base64
                image_base64 = ""
                if result["image_path"]:
                    # Fix the path if it contains the old GRAPSE folder name
                    corrected_path = result["image_path"].replace("/Users/aseth/Projects/GRAPSE/", "/Users/aseth/Projects/GRAPSE_Dev/")
                    if os.path.exists(corrected_path):
                        encoded = self.encode_image(result["image_path"])  # Pass original path, encode_image will fix it
                        image_base64 = encoded if encoded else ""
                
                # Use corrected path for the FigureInfo object
                corrected_image_path = result["image_path"].replace("/Users/aseth/Projects/GRAPSE/", "/Users/aseth/Projects/GRAPSE_Dev/") if result["image_path"] else ""
                
                figure = FigureInfo(
                    figure_id=result["figure_id"],
                    description=result["description"] or "No description available",
                    image_path=corrected_image_path,
                    image_base64=image_base64,
                    paper_title=result["paper_title"] or "Unknown Title",
                    paper_authors=result["paper_authors"] or "Unknown Authors",
                    paper_year=paper_year
                )
                figures.append(figure)
                
                # Track unique papers for bibliography if we have paper info
                if result["paper_title"]:
                    paper_base_key = f"{figure.paper_title}_{figure.paper_authors}"
                    paper_key = f"{figure.paper_title}_{figure.paper_authors}_{figure.paper_year}"
                    
                    # Check if this is a duplicate paper with different year
                    existing_key = None
                    for existing_paper_key in self.unique_papers.keys():
                        if existing_paper_key.startswith(paper_base_key + "_"):
                            existing_key = existing_paper_key
                            break
                    
                    if existing_key:
                        # Update to use the most recent year if we have a better one
                        existing_year = self.unique_papers[existing_key]["year"]
                        current_year = figure.paper_year
                        
                        # Use most recent year (or replace Unknown with actual year)
                        if (current_year != "Unknown" and existing_year == "Unknown") or \
                           (current_year != "Unknown" and existing_year != "Unknown" and current_year > existing_year):
                            # Update existing entry with newer year
                            citation_num = self.unique_papers[existing_key]["number"]
                            del self.unique_papers[existing_key]
                            self.unique_papers[paper_key] = {
                                "number": citation_num,
                                "title": figure.paper_title,
                                "authors": figure.paper_authors,
                                "year": figure.paper_year
                            }
                        # Otherwise keep the existing one and reuse its number
                        else:
                            citation_num = self.unique_papers[existing_key]["number"]
                            self.unique_papers[paper_key] = {
                                "number": citation_num,
                                "title": figure.paper_title,
                                "authors": figure.paper_authors,
                                "year": figure.paper_year
                            }
                            
                        # Track figure path by citation number (for both update cases)
                        citation_num = self.unique_papers[paper_key]["number"]
                        figure.citation_number = citation_num
                        
                        if citation_num not in self.citation_figures:
                            self.citation_figures[citation_num] = []
                        self.citation_figures[citation_num].append({
                            "path": figure.image_path,
                            "description": figure.description,
                            "figure_id": figure.figure_id
                        })
                    else:
                        # New unique paper
                        self.citation_counter += 1
                        self.unique_papers[paper_key] = {
                            "number": self.citation_counter,
                            "title": figure.paper_title,
                            "authors": figure.paper_authors,
                            "year": figure.paper_year
                        }
                    
                    # Track figure path by citation number
                    citation_num = self.unique_papers[paper_key]["number"]
                    figure.citation_number = citation_num
                    
                    if citation_num not in self.citation_figures:
                        self.citation_figures[citation_num] = []
                    self.citation_figures[citation_num].append({
                        "path": figure.image_path,
                        "description": figure.description,
                        "figure_id": figure.figure_id
                    })
            
            return figures
            
        except Exception as e:
            print(f"Error in figure retrieval: {e}")
            return []

    def format_context_with_citations(self, semantic_citations: List[CitationInfo], entity_citations: List[CitationInfo], figures: List[FigureInfo] = None) -> Tuple[str, List]:
        """
        Format retrieved content with clear citation attribution.
        Returns both text context and list of images for multimodal processing.
        """
        all_citations = semantic_citations + entity_citations
        
        # Remove duplicates based on document_id
        seen_docs = set()
        unique_citations = []
        for citation in all_citations:
            if citation.document_id not in seen_docs:
                unique_citations.append(citation)
                seen_docs.add(citation.document_id)
        
        formatted_context = ""
        citation_list = []
        image_list = []
        
        # Process all citations and group by citation number
        citation_content = {}
        
        for citation in unique_citations:
            paper_key = f"{citation.paper_title}_{citation.paper_authors}_{citation.paper_year}"
            if paper_key in self.unique_papers:
                citation_num = self.unique_papers[paper_key]["number"]
                if citation_num not in citation_content:
                    citation_content[citation_num] = {
                        "semantic": [],
                        "entity": []
                    }
                    citation_list.append(citation_num)
                
                if citation.retrieval_source == "semantic":
                    citation_content[citation_num]["semantic"].append(citation.content)
                elif citation.retrieval_source == "entity":
                    entity_info = f" (matched entity: {citation.entity_matched})" if citation.entity_matched else ""
                    citation_content[citation_num]["entity"].append(f"{entity_info}: {citation.content}")
        
        # Process figures
        if figures:
            formatted_context += "RELEVANT FIGURES:\n\n"
            for i, figure in enumerate(figures, 1):
                # Get citation number for this figure if available
                if figure.paper_title and figure.paper_title != "Unknown Title":
                    paper_key = f"{figure.paper_title}_{figure.paper_authors}_{figure.paper_year}"
                    if paper_key in self.unique_papers:
                        citation_num = self.unique_papers[paper_key]["number"]
                        if citation_num not in citation_list:
                            citation_list.append(citation_num)
                        formatted_context += f"Figure {i} [Citation {citation_num}]: {figure.description}\n\n"
                    else:
                        formatted_context += f"Figure {i}: {figure.description}\n\n"
                else:
                    formatted_context += f"Figure {i}: {figure.description}\n\n"
                
                # Add image to image list for multimodal processing
                if figure.image_base64 and len(figure.image_base64) > 0:
                    image_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{figure.image_base64}"
                        }
                    })
            
            formatted_context += "---\n\n"
        
        # Format context with grouped content
        for citation_num in sorted(citation_list):
            if citation_num in citation_content:
                content_info = citation_content[citation_num]
                
                formatted_context += f"[Citation {citation_num}] Sources:\n"
                
                if content_info["semantic"]:
                    formatted_context += "SEMANTIC SEARCH:\n"
                    for content in content_info["semantic"]:
                        formatted_context += f"{content}\n\n"
                
                if content_info["entity"]:
                    formatted_context += "ENTITY-BASED SEARCH:\n"
                    for content in content_info["entity"]:
                        formatted_context += f"{content}\n\n"
                
                formatted_context += "---\n\n"
        
        # Add instruction about using relevant citations
        formatted_context += f"\nAvailable citations: {citation_list}\n"
        formatted_context += "INSTRUCTIONS: Use only the citations and figures that are relevant to answering the question. You do not need to use all citations.\n"
        if figures:
            formatted_context += f"Available figures: {len(figures)} figure(s). Only reference figures that are directly relevant to your answer.\n"
        
        return formatted_context.strip(), image_list

    def create_bibliography(self, answer: str = None) -> str:
        """
        Create a formatted bibliography from unique papers, including figure paths.
        If answer is provided, only include citations that were actually used.
        """
        if not self.unique_papers:
            return ""
            
        bibliography = "\n\nCITATIONS:\n"
        
        # Get unique papers by citation number (remove duplicates)
        unique_by_number = {}
        for paper_key, paper_info in self.unique_papers.items():
            citation_num = paper_info["number"]
            if citation_num not in unique_by_number:
                unique_by_number[citation_num] = paper_info
            else:
                # Keep the one with the most recent year
                existing_year = unique_by_number[citation_num]["year"]
                current_year = paper_info["year"]
                if (current_year != "Unknown" and existing_year == "Unknown") or \
                   (current_year != "Unknown" and existing_year != "Unknown" and current_year > existing_year):
                    unique_by_number[citation_num] = paper_info
        
        # If answer is provided, filter to only used citations
        if answer:
            used_citations = self.extract_used_citations(answer)
            unique_by_number = {num: info for num, info in unique_by_number.items() if num in used_citations}
        
        # Sort by citation number
        sorted_papers = sorted(unique_by_number.items(), key=lambda x: x[0])
        
        for citation_num, paper_info in sorted_papers:
            bibliography += f"[{citation_num}] {paper_info['authors']} ({paper_info['year']}). \"{paper_info['title']}\"\n"
            
            # Add figure information if available
            if citation_num in self.citation_figures:
                figures = self.citation_figures[citation_num]
                if figures:
                    bibliography += f"   Figures: {len(figures)} figure(s) available\n"
                    for i, fig in enumerate(figures, 1):
                        bibliography += f"   - Figure {i}: {fig['figure_id']}\n"
                        bibliography += f"     Path: {fig['path']}\n"
                        bibliography += f"     Description: {fig['description'][:100]}{'...' if len(fig['description']) > 100 else ''}\n"
        
        return bibliography

    def extract_used_citations(self, answer: str) -> set:
        """
        Extract citation numbers that were actually used in the answer.
        """
        # Pattern to match both [Citation X] and [X] formats
        citation_pattern = r'\[(?:Citation\s+)?([0-9,\s]+)\]'
        citation_matches = re.findall(citation_pattern, answer)
        used_citations = set()
        
        for match in citation_matches:
            # Split by comma and clean up whitespace
            numbers = [int(num.strip()) for num in match.split(',') if num.strip().isdigit()]
            used_citations.update(numbers)
        
        return used_citations

    def setup(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model_name=self.model_name)

        # Setup entity extraction chain
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting Author, Algorithm, Software, Solver, Formula, Institution, and Benchmark entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        
        self.entity_chain = entity_prompt | self.llm.with_structured_output(Entities)
        
        # Create fulltext index for entities if it doesn't exist
        self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        def enhanced_retriever_fn(inputs):
            question = inputs["question"]
            
            # Reset citation tracking for each question
            self.citation_counter = 0
            self.unique_papers = {}
            self.citation_figures = {}
            
            # Perform selective retrieval based on similarity thresholds
            semantic_citations = self.semantic_document_retrieval(question, k=20)  # Get more candidates
            entity_citations = self.entity_based_retrieval(question, k=10)  # Get more candidates  
            figures = self.figure_retrieval(question, k=10)  # Get more candidates
            
            print(f"Retrieved {len(semantic_citations)} semantic docs, {len(entity_citations)} entity docs, {len(figures)} figures")
            
            # Format context with citations and get images
            context, images = self.format_context_with_citations(semantic_citations, entity_citations, figures)
            
            return {
                "context": context,
                "question": question,
                "images": images,
                "semantic_citations": semantic_citations,
                "entity_citations": entity_citations,
                "figures": figures
            }

        combined_retriever = RunnableLambda(enhanced_retriever_fn)

        # Create a multimodal prompt function
        def create_multimodal_prompt(context, question, images):
            # Base text prompt
            text_content = f'''You are an assistant for question-answering tasks specialized in Process Systems Engineering. 
            Use the following pieces of retrieved context to answer the question. The context includes citation numbers in brackets [Citation X].
            
            INSTRUCTIONS:
            1. Only use citations and figures that are directly relevant to answering the question
            2. You do NOT need to use all provided citations - be selective
            3. Include inline citations using the format [X] where X is the citation number when you reference specific information
            4. Only reference figures/images that add value to your answer
            5. Be precise and academic, but focus on quality over quantity of citations
            6. If the question can be answered well with fewer sources, that's preferred
            
            Context: {context}
            
            Question: {question}
            
            Answer:'''
            
            # Create message content
            message_content = [{"type": "text", "text": text_content}]
            
            # Add images if available
            if images:
                message_content.extend(images)
            
            return [("human", message_content)]

        # Create multimodal answer generation function
        def generate_multimodal_answer(inputs):
            context = inputs["context"]
            question = inputs["question"]
            images = inputs.get("images", [])
            
            # Create multimodal prompt
            prompt_messages = create_multimodal_prompt(context, question, images)
            
            # Generate answer using multimodal capabilities
            response = self.llm.invoke(prompt_messages)
            return response.content if hasattr(response, 'content') else str(response)

        # Create the chain with bibliography addition
        def add_bibliography(response_dict):
            answer = response_dict["answer"]
            bibliography = self.create_bibliography(answer)  # Pass answer to filter used citations
            return answer + bibliography

        self.combined_rag_chain = (
            combined_retriever
            | RunnablePassthrough.assign(
                answer=RunnableLambda(generate_multimodal_answer)
            )
            | RunnableLambda(add_bibliography)
        )
    
    def get_answer(self, question):
        input_data = {"question": question}
        response = self.combined_rag_chain.invoke(input_data)
        return response, self.retrieved_docs

    def format_answer_with_links(self, answer: str) -> str:
        """
        Convert citation references to clickable links in the answer.
        """
        # Pattern to match citation references like [Citation 1], [Citation 2,3], [1], [2,3]
        def replace_citations(match):
            citation_text = match.group(1)
            citation_numbers = [num.strip() for num in citation_text.split(',')]
            
            links = []
            for num in citation_numbers:
                links.append(f'<a href="#citation-{num}" class="citation-link">[{num}]</a>')
            
            if len(links) == 1:
                return links[0]
            else:
                return '[' + ', '.join([link[1:-1] for link in links]) + ']'
        
        # Replace citation patterns - handle both [Citation X] and [X] formats
        citation_pattern = r'\[(?:Citation\s+)?([0-9,\s]+)\]'
        formatted_answer = re.sub(citation_pattern, replace_citations, answer)
        
        # Convert newlines to HTML breaks for proper display
        formatted_answer = formatted_answer.replace('\n', '<br>')
        
        return formatted_answer

    def extract_referenced_figures(self, answer: str, figures: List[FigureInfo]) -> List[FigureInfo]:
        """
        Extract only the figures that are actually referenced in the answer text.
        """
        # Find all citation numbers mentioned in the answer
        citation_pattern = r'\[(?:Citation\s+)?([0-9,\s]+)\]'
        citation_matches = re.findall(citation_pattern, answer)
        referenced_citation_numbers = set()
        
        for match in citation_matches:
            # Split by comma and clean up whitespace
            numbers = [int(num.strip()) for num in match.split(',') if num.strip().isdigit()]
            referenced_citation_numbers.update(numbers)
        
        # Filter figures to only include those with citation numbers mentioned in the answer
        referenced_figures = []
        for figure in figures:
            if figure.citation_number and figure.citation_number in referenced_citation_numbers:
                referenced_figures.append(figure)
        
        return referenced_figures

    def generate_html_report(self, question: str, answer: str, figures: List[FigureInfo], output_file: str = None):
        """
        Generate an HTML report with the question, answer, and embedded images.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"grapse_report_{timestamp}.html"
        
        # Extract only referenced figures
        referenced_figures = self.extract_referenced_figures(answer, figures)
        
        # Calculate citation instances count outside f-string to avoid backslash issues
        citation_instances_count = len(re.findall(r'\[[0-9,\s]+\]', answer))
        
        # Simple CSS styles - plain and minimal
        css_styles = '''
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.5;
            color: #000;
        }
        
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        
        h2 {
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        h3 {
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        p {
            margin-bottom: 15px;
        }
        
        .stats {
            margin-bottom: 30px;
        }
        
        .stats p {
            margin: 5px 0;
        }
        
        .figure-image {
            max-width: 100%;
            height: auto;
            margin: 15px 0;
        }
        
        a {
            color: blue;
            text-decoration: underline;
        }
        
        ol {
            margin-left: 20px;
        }
        
        li {
            margin-bottom: 10px;
        }
        '''
        
        # Start building HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GRAPSE Analysis Report</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <h1>GRAPSE Analysis Report</h1>
    <p><em>Graph-based Retrieval Agent for Process Systems Engineering</em></p>
    
    <h2>Question</h2>
    <p>{html.escape(question)}</p>

    <h2>Answer</h2>
    ANSWER_PLACEHOLDER
"""
        
        # Format the answer separately to avoid f-string conflicts
        formatted_answer = self.format_answer_with_links(answer)
        html_content = html_content.replace("ANSWER_PLACEHOLDER", formatted_answer)
        
        # Add figures section if we have referenced figures
        if referenced_figures:
            html_content += '''
    <h2>Supporting Figures</h2>
    <p>The following figures were referenced in the analysis above:</p>
'''
            
            for i, figure in enumerate(referenced_figures, 1):
                figure_html = f'''
    <h3>Figure {i}: {html.escape(figure.figure_id)} [Citation {figure.citation_number}]</h3>
    <p><em>{html.escape(figure.description)}</em></p>
'''
                
                # Add image if available
                if figure.image_base64:
                    figure_html += f'''
    <img src="data:image/jpeg;base64,{figure.image_base64}" 
         alt="{html.escape(figure.figure_id)}" 
         class="figure-image">
'''
                else:
                    figure_html += f'''
    <p><em>Image not available: {html.escape(figure.image_path)}</em></p>
'''
                
                html_content += figure_html
        
        # Add bibliography from the existing method - only show used citations
        bibliography = self.create_bibliography(answer)
        if bibliography:
            # Convert the bibliography to HTML format
            bib_lines = bibliography.split('\n')
            html_bibliography = '''
    <h2>References</h2>
    <ol>
'''
            
            for line in bib_lines:
                if line.strip() and line.startswith('[') and ']' in line:
                    # Extract citation number and content
                    citation_match = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                    if citation_match:
                        citation_num = citation_match.group(1)
                        citation_content = citation_match.group(2)
                        html_bibliography += f'        <li id="citation-{citation_num}">{html.escape(citation_content)}</li>\n'
            
            html_bibliography += '''
    </ol>
'''
            html_content += html_bibliography
        
        # Close HTML
        html_content += '''
</body>
</html>
'''
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_file}")
        return output_file

    def get_answer_with_html(self, question: str, output_file: str = None):
        """
        Get answer and generate HTML report in one go.
        """
        # Get the regular answer first
        answer, retrieved_docs = self.get_answer(question)
        
        # Get figures from the retrieval process if available
        figures = []
        # We need to extract figures from the last retrieval
        # For simplicity, we'll do a fresh figure retrieval
        try:
            figures = self.figure_retrieval(question, k=3)
        except Exception as e:
            print(f"Error retrieving figures for HTML: {e}")
            figures = []
        
        # Generate HTML report
        html_file = self.generate_html_report(question, answer, figures, output_file)
        
        return answer, retrieved_docs, html_file


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, required=False, default="gpt-4o-mini", help="Model to use for the answer")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output HTML file name")
    args = parser.parse_args()
    pipeline = MainPipeline(args.model)
    pipeline.retrieved_docs = None
    pipeline.setup()
    
    if args.output:
        # Generate HTML report
        answer, context, html_file = pipeline.get_answer_with_html(args.question, args.output)
        print("Answer:\n", answer)
        print(f"\nHTML report generated: {html_file}")
    else:
        # Just get the answer
        answer, context = pipeline.get_answer(args.question)
        print("Answer:\n", answer)