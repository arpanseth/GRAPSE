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

    def semantic_document_retrieval(self, question: str, k: int = 5) -> List[CitationInfo]:
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

    def entity_based_retrieval(self, question: str, k: int = 3) -> List[CitationInfo]:
        """
        Perform entity-based retrieval with document traversal and paper metadata.
        """
        try:
            entities = self.extract_entities(question)
            if not entities:
                return []
                
            citations = []
            
            for entity in entities[:3]:  # Limit to top 3 entities to avoid noise
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
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
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
                if result["image_path"] and os.path.exists(result["image_path"]):
                    encoded = self.encode_image(result["image_path"])
                    image_base64 = encoded if encoded else ""
                
                figure = FigureInfo(
                    figure_id=result["figure_id"],
                    description=result["description"] or "No description available",
                    image_path=result["image_path"] or "",
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
        self.llm = ChatOpenAI(temperature=0.0, model_name=self.model_name)

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
            7. Synthesize information from different sources to create a comprehensive answer
            8. Analyze the provided figures/images and incorporate insights from them in your answer
            
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


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, required=False, default="gpt-4o-mini", help="Model to use for the answer")
    args = parser.parse_args()
    pipeline = MainPipeline(args.model)
    pipeline.retrieved_docs = None
    pipeline.setup()
    answer, context = pipeline.get_answer(args.question)
    print("Answer:\n", answer)