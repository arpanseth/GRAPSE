#!/usr/bin/env python3
"""
Docker entrypoint script for GRAPSE
Handles initial processing of papers and starts the system
"""

import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ENTRYPOINT] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/entrypoint.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def wait_for_neo4j():
    """Wait for Neo4j to be ready"""
    logger.info("Waiting for Neo4j to be ready...")
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Test Neo4j connection
            result = subprocess.run([
                'python', '-c', 
                'from langchain_community.graphs import Neo4jGraph; graph = Neo4jGraph(); print("Neo4j connected!")'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("Neo4j is ready!")
                return True
                
        except Exception as e:
            logger.debug(f"Neo4j not ready yet: {e}")
            
        retry_count += 1
        time.sleep(10)
    
    logger.error("Neo4j failed to become ready after 5 minutes")
    return False

def get_processed_papers():
    """Get list of already processed papers"""
    state_file = Path('/app/state/processed_papers.json')
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read processed papers state: {e}")
    return {}

def save_processed_papers(processed_papers):
    """Save list of processed papers"""
    state_file = Path('/app/state/processed_papers.json')
    try:
        with open(state_file, 'w') as f:
            json.dump(processed_papers, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save processed papers state: {e}")

def get_papers_to_process():
    """Get list of papers that need processing"""
    papers_dir = Path('/app/papers')
    if not papers_dir.exists():
        logger.warning("Papers directory not found")
        return []
    
    pdf_files = list(papers_dir.glob('*.pdf'))
    if not pdf_files:
        logger.info("No PDF files found in papers directory")
        return []
    
    processed_papers = get_processed_papers()
    new_papers = []
    
    for pdf_file in pdf_files:
        pdf_name = pdf_file.name
        if pdf_name not in processed_papers:
            new_papers.append(pdf_name)
            logger.info(f"Found new paper: {pdf_name}")
    
    logger.info(f"Found {len(new_papers)} new papers to process")
    return new_papers

def run_magic_pdf(paper_name):
    """Run magic-pdf extraction for a specific paper"""
    logger.info(f"Running magic-pdf for {paper_name}")
    
    try:
        result = subprocess.run([
            'magic-pdf', 
            '-p', f'/app/papers/{paper_name}',
            '-o', '/app/output',
            '-m', 'auto'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"Magic-pdf completed for {paper_name}")
            return True
        else:
            logger.error(f"Magic-pdf failed for {paper_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Magic-pdf timeout for {paper_name}")
        return False
    except Exception as e:
        logger.error(f"Magic-pdf error for {paper_name}: {e}")
        return False

def run_graph_building(paper_folder):
    """Run graph building for a specific paper folder"""
    logger.info(f"Building graph for {paper_folder}")
    
    try:
        result = subprocess.run([
            'python', 'scripts/load_data_into_graph_langchain.py',
            '-p', paper_folder
        ], capture_output=True, text=True, timeout=1200)
        
        if result.returncode == 0:
            logger.info(f"Graph building completed for {paper_folder}")
            return True
        else:
            logger.error(f"Graph building failed for {paper_folder}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Graph building timeout for {paper_folder}")
        return False
    except Exception as e:
        logger.error(f"Graph building error for {paper_folder}: {e}")
        return False

def run_enrichment():
    """Run graph enrichment"""
    logger.info("Running graph enrichment...")
    
    try:
        # Entity deduplication
        for entity_type in ['Author', 'Benchmark']:
            logger.info(f"Deduplicating {entity_type} entities...")
            result = subprocess.run([
                'python', 'scripts/entity_dedupe_w_training.py',
                '-l', entity_type
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.warning(f"Entity deduplication failed for {entity_type}: {result.stderr}")
        
        # Community detection
        logger.info("Running community detection...")
        result = subprocess.run([
            'python', 'scripts/community_building.py',
            '-l', '[0,1]'
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            logger.info("Graph enrichment completed")
            return True
        else:
            logger.error(f"Community detection failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Graph enrichment timeout")
        return False
    except Exception as e:
        logger.error(f"Graph enrichment error: {e}")
        return False

def process_papers():
    """Process all new papers"""
    papers_to_process = get_papers_to_process()
    processed_papers = get_processed_papers()
    
    if not papers_to_process:
        logger.info("No new papers to process")
        return True
    
    success_count = 0
    
    for paper_name in papers_to_process:
        logger.info(f"Processing paper {paper_name}...")
        
        # Step 1: Magic-PDF extraction
        if run_magic_pdf(paper_name):
            
            # Step 2: Find the output folder
            paper_name_no_ext = paper_name.replace('.pdf', '')
            output_folders = list(Path('/app/output').glob(f'{paper_name_no_ext}*'))
            
            if output_folders:
                paper_folder = output_folders[0].name
                
                # Step 3: Graph building
                if run_graph_building(paper_folder):
                    processed_papers[paper_name] = {
                        'processed_at': time.time(),
                        'output_folder': paper_folder,
                        'status': 'completed'
                    }
                    success_count += 1
                    logger.info(f"Successfully processed {paper_name}")
                else:
                    processed_papers[paper_name] = {
                        'processed_at': time.time(),
                        'output_folder': paper_folder,
                        'status': 'graph_failed'
                    }
                    logger.error(f"Graph building failed for {paper_name}")
            else:
                processed_papers[paper_name] = {
                    'processed_at': time.time(),
                    'status': 'output_folder_not_found'
                }
                logger.error(f"Output folder not found for {paper_name}")
        else:
            processed_papers[paper_name] = {
                'processed_at': time.time(),
                'status': 'magic_pdf_failed'
            }
            logger.error(f"Magic-PDF failed for {paper_name}")
        
        # Save progress after each paper
        save_processed_papers(processed_papers)
    
    logger.info(f"Processed {success_count}/{len(papers_to_process)} papers successfully")
    
    # Run enrichment if we processed any papers successfully
    if success_count > 0:
        run_enrichment()
    
    return success_count > 0

def start_interactive_mode():
    """Start interactive chat mode"""
    logger.info("Starting interactive chat mode...")
    logger.info("=" * 60)
    logger.info("GRAPSE is ready! You can now ask questions about your papers.")
    logger.info("Example: What is process optimization?")
    logger.info("Type 'exit' to quit, 'help' for more options")
    logger.info("=" * 60)
    
    try:
        while True:
            try:
                question = input("\nGRASPE> ").strip()
                
                if question.lower() in ['exit', 'quit']:
                    logger.info("Goodbye!")
                    break
                elif question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  status - Show system status")
                    print("  papers - List processed papers")
                    print("  html <question> - Generate HTML report")
                    print("  exit - Exit GRAPSE")
                    continue
                elif question.lower() == 'status':
                    processed_papers = get_processed_papers()
                    print(f"\nProcessed papers: {len(processed_papers)}")
                    for paper, info in processed_papers.items():
                        print(f"  - {paper}: {info.get('status', 'unknown')}")
                    continue
                elif question.lower() == 'papers':
                    processed_papers = get_processed_papers()
                    print("\nProcessed papers:")
                    for paper in processed_papers.keys():
                        print(f"  - {paper}")
                    continue
                elif question.lower().startswith('html '):
                    html_question = question[5:]
                    print(f"\nGenerating HTML report for: {html_question}")
                    
                    result = subprocess.run([
                        'python', 'pipelines/neo4j_citation_hybrid_html.py',
                        '-q', html_question,
                        '-o', f'/app/output/report_{int(time.time())}.html'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print("HTML report generated successfully!")
                        print(result.stdout)
                    else:
                        print(f"Error generating HTML report: {result.stderr}")
                    continue
                elif not question:
                    continue
                
                # Regular question - use the GRAPSE pipeline
                print(f"\nProcessing question: {question}")
                
                result = subprocess.run([
                    'python', 'pipelines/grapse.py',
                    '-q', question
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("\nAnswer:")
                    print(result.stdout)
                else:
                    print(f"Error processing question: {result.stderr}")
                    
            except KeyboardInterrupt:
                logger.info("\nReceived interrupt signal. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")

def main():
    """Main entrypoint function"""
    logger.info("Starting GRAPSE Docker container...")
    
    # Wait for Neo4j
    if not wait_for_neo4j():
        logger.error("Failed to connect to Neo4j. Exiting.")
        sys.exit(1)
    
    # Process papers
    logger.info("Processing papers...")
    process_papers()
    
    # Start interactive mode
    start_interactive_mode()

if __name__ == "__main__":
    main()