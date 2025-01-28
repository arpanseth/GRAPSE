from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_openai import ChatOpenAI
import logging, sys, traceback
from argparse import ArgumentParser
import pickle
import base64
from openai import OpenAI, RateLimitError
import concurrent.futures
from multiprocessing import cpu_count
from langchain_core.rate_limiters import InMemoryRateLimiter
import json
import hashlib
from utils import SharedRateLimiter
import time
from multiprocessing import Manager
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv(dotenv_path='../.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

# Define rate limit
tokens_per_request = 100
max_tokens_per_minute = 30000
max_requests_per_minute = max_tokens_per_minute // tokens_per_request  # e.g., 150

def init_shared_rate_limiter(max_requests_per_minute):
    manager = Manager()
    return SharedRateLimiter(max_requests_per_minute, manager)
# Model and graph transformer settings
graph = Neo4jGraph()

def setup_logging():
    """
    Configures logging for both the main and worker processes.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter(
        '%(asctime)s [%(processName)s] [%(levelname)s] %(message)s'
    )

    # Console handler for real-time feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler to persist logs
    file_handler = logging.FileHandler('parallel_processing.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_unique_id(content):
    """
    Generates a unique ID based on the content using SHA-256 hashing.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Create image context prompt
def create_image_context_prompt(figure_name, chunk_text):
    if figure_name is None:
        text = f"""Please summarize what you see in the Figure attached. 
            Here is some text referencing the figure in the image:
            
            {chunk_text}
            """
    else:
        text = f"""Please summarize what you see in the image attached. 
            Here is some text referencing the attached image as {figure_name}:
            
            {chunk_text}
            """
    return text


# Summarize image in the context 
def summarize_image(image_path, text, shared_rate_limiter):
    while not shared_rate_limiter.acquire():
        time.sleep(1)  # Wait for a second before retrying
    
    client = OpenAI()
    # Getting the base64 string
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": text,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        )
    except RateLimitError as e:
        print("Rate limit exceeded. Please try again later.")
        raise e

    return response.choices[0].message.content
    
# Define process_chunk at the top level
def process_chunk(c, chunks, chunk_to_image, main_folder_path, paper_metadata, shared_rate_limiter):
    logger = logging.getLogger()

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", rate_limiter=shared_rate_limiter)
    llm_transformer = LLMGraphTransformer(llm=llm,
                                    allowed_nodes=["Author", "Algorithm", "Software", 
                                                "Solver", "Formula", "Institution",
                                                "Benchmark"],
                                    #allowed_relationships=["Related To"],
                                    node_properties=["description"]
                                    )

    try:
        # Initialize or import llm_transformer and other dependencies inside the function
        logger.info(f'Processing Chunk: {c+1} of {len(chunks)}')
        #print(f'Processing Chunk: {c+1} of {len(chunks)}')
        
        # Extract KG using langchain
        chunk_content = chunks[c]
        chunk_document = Document(page_content=chunk_content, metadata=paper_metadata)
        graph_documents = llm_transformer.convert_to_graph_documents([chunk_document])
                
        # Get (Figure Name, image path) for all the images that are connected to this chunk 
        images_in_chunk = [(i[1], i[2]) for i in chunk_to_image if i[0] == c]
        
        for fig_name, img_path in images_in_chunk: 
            image_context_text = create_image_context_prompt(fig_name, chunk_content)
            full_img_path = f'{main_folder_path}/{img_path}'
            # Check if the image path exists
            if os.path.exists(full_img_path):
                img_summary = summarize_image(full_img_path, image_context_text, shared_rate_limiter)
            
                # Create image graph node
                img_graph_node = Node(
                    id=f'{fig_name} {full_img_path}', 
                    type='Figure', 
                    properties={'text': img_summary, 'path': full_img_path}
                )
                
                # Create GraphDocument for the image
                img_graph_document = GraphDocument(
                    nodes=[img_graph_node],
                    relationships=[],  # Define relationships if needed
                    source=graph_documents[0].source  # Assuming all have the same source
                )
                graph_documents.append(img_graph_document)
        
        

        logger.info(f'Processed chunk {c}')
        return graph_documents
    except Exception as e:
        logger.error(f'Error processing chunk {c}: {e}')
        logger.error(traceback.format_exc())
        # It's essential to re-raise the exception to notify the main process
        raise

def worker_init():
    setup_logging()

def parallel_process_chunks(chunks, chunk_to_image, main_folder_path, graph, paper_metadata, shared_rate_limiter):
    graph_documents_all = []
    
    # Determine the number of workers; you can adjust this as needed
    num_workers = min(4, len(chunks))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init) as executor:
        # Prepare arguments for each chunk
        futures = [
            executor.submit(
                process_chunk, 
                c, 
                chunks, 
                chunk_to_image, 
                main_folder_path,
                paper_metadata,
                shared_rate_limiter
            )
            for c in range(len(chunks))
        ]
        
        # As each future completes, collect the results
        for future in concurrent.futures.as_completed(futures):
            try:
                graph_documents = future.result()
                graph_documents_all.extend(graph_documents)
            except Exception as e:
                print(f'Chunk processing generated an exception: {e}')
    # After all chunks are processed, add to the graph
    graph.add_graph_documents(graph_documents_all, include_source=True, baseEntityLabel=True)
    

def extract_paper_metadata(text):
    # Define the function schema
    metadata_extraction_function = {
        "name": "extract_metadata",
        "description": "Extract the paper's metadata from the provided text",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the paper"
                },
                "year": {
                    "type": "string",
                    "description": "The year of publication of the paper"
                },
                "authors": {
                    "type": "string",
                    "description": "List of author names separated by semicolons"
                },
                "abstract": {
                    "type": "string",
                    "description": "The abstract of the paper"
                }
            },
            "required": ["title", "authors", "abstract"]
        }
    }

    # Create the user message
    extraction_message = {
        "role": "user",
        "content": (
            "Extract the following information from this academic text:\n"
            "1. Title of the paper\n"
            "2. Author names\n"
            "3. Abstract\n\n"
            "Return your response in valid JSON with keys: 'title', 'authors' (list), 'abstract'.\n\n"
            f"Text:\n{text}"
        )
    }

    try:
        client = OpenAI()
        # Call the new chat.completions endpoint
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[extraction_message],
            functions=[metadata_extraction_function],
            function_call={"name": "extract_metadata"},
            temperature=0
        )
        #response.choices[0].message.function_call.arguments
        # Parse the function call
        try:
            choice = response.choices[0].message
            args = json.loads(choice.function_call.arguments)
            logging.info(f"Extracted metadata via function call: {args}")
            return args
        except:
            logging.error("No function call was returned by the model.")
            return None

    except Exception as e:
        logging.error(f"Error extracting metadata: {str(e)}")
        return None
    
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('-p','--paper-name', 
                        help='Name of the paper to convert to graph. Must have a folder in output folder and a pkl file in output/extracts.', 
                        required=True, default='output')
    args = vars(parser.parse_args())
    paper_name = args['paper_name']
    main_folder_path = '/Users/aseth/Projects/GRAPSE' 
    pkl_file = f'{main_folder_path}/output/extracts/{paper_name}.pkl'
    
    # Read file and data fields
    results = pickle.load(open(pkl_file, 'rb'))
    chunks = results['chunks']
    chunk_to_image = results['chunk_to_image']

    # Extract metadata from first 2 chunks
    initial_text = " ".join(chunks[:6])
    # try metadata extraction at least 10 time and catch any errors
    for i in range(10):
        paper_metadata = extract_paper_metadata(initial_text)
        if paper_metadata is not None:
            break
        time.sleep(1)
    #print(paper_metadata)
    metadata_for_chunk = {'source_paper_name': paper_metadata['title'], 'source_authors': paper_metadata['authors']}
    
    # Initialize shared rate limiter
    shared_rate_limiter = init_shared_rate_limiter(max_requests_per_minute)
    # Process chunks in parallel
    parallel_process_chunks(
        chunks=chunks,
        chunk_to_image=chunk_to_image,
        main_folder_path=main_folder_path,
        graph=graph,
        paper_metadata=metadata_for_chunk,
        shared_rate_limiter=shared_rate_limiter
    )
    # Create paper node
    paper_node = Node(id=paper_metadata['title'], type='Paper', properties=paper_metadata)
    paper_document = Document(page_content=initial_text, metadata=paper_metadata)
    paper_graph_document = GraphDocument(nodes=[paper_node], relationships=[], source=paper_document)
    graph.add_graph_documents([paper_graph_document], include_source=False, baseEntityLabel=True)
    # Run cypher query to create a relationship between the paper node and all the Document nodes that match on paper title and source_paper_name
    query = f"""
        MATCH (p:Paper {{title: '{paper_metadata['title']}'}})
        MATCH (d:Document)
        WHERE d.source_paper_name = '{paper_metadata['title']}'
        CREATE (p)-[:CONTAINS]->(d)
    """
    graph.query(query)

    # load
    # ## Loop through all chunks and convert them to KG
    # for c in range(len(chunks)):
    #     print(f'Processesing Chunk: {c+1} of ', len(chunks))
    #     # Extract KG using langchain
    #     chunk_document = Document(page_content=chunks[c])
    #     graph_documents = llm_transformer.convert_to_graph_documents([chunk_document])
    #     # Get (Figure Name, image path) for all the images that are connected to this chunk 
    #     images_in_chunk = [(i[1], i[2]) for i in chunk_to_image if i[0] == c]
    #     # For each image in chunk, generate summary using LLM
    #     for fig_name, img_path in images_in_chunk: 
    #         image_context_text = create_image_context_prompt(fig_name, chunks[c])
    #         img_path = f'{main_folder_path}/{img_path}'
    #         img_summary = summarize_image(img_path, image_context_text)
    #         #img_document = Document(page_content=img_summary, metadata={'type': 'image', 'path': img_path})
    #         #TODO: Figure out a better way to do this. Currently there is no way to access the Source Document Node of the chunk
    #         img_graph_node = Node(id=f'{fig_name} {img_path}', type='Figure', properties={'text': img_summary, 'path': img_path})
    #         #img_img_relationship = Relationship(source=img_graph_node, target=img_graph_node, type='IS SAME')
    #         # Connect image to the source document of the chunk
    #         img_graph_document = GraphDocument(nodes=[img_graph_node],
    #                                            relationships=[],
    #                                            source=graph_documents[0].source
    #                                            )
    #         graph_documents.append(img_graph_document)
    #     # filly add all the documents to the graph
    #     graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=False)
        