#!/usr/bin/env python3
"""
Interactive chat interface for GRAPSE
For chat-only mode when papers are already processed
"""

import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [CHAT] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/chat.log'),
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

def start_interactive_mode():
    """Start interactive chat mode"""
    logger.info("Starting GRAPSE interactive chat...")
    logger.info("=" * 60)
    logger.info("GRAPSE Chat Interface")
    logger.info("Ask questions about your processed papers!")
    logger.info("Type 'exit' to quit, 'help' for more options")
    logger.info("=" * 60)
    
    # Show processed papers count
    processed_papers = get_processed_papers()
    logger.info(f"Ready to answer questions about {len(processed_papers)} processed papers")
    
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
                    completed_papers = [p for p, info in processed_papers.items() 
                                     if info.get('status') == 'completed']
                    print(f"Successfully processed: {len(completed_papers)}")
                    failed_papers = [p for p, info in processed_papers.items() 
                                   if info.get('status') != 'completed']
                    if failed_papers:
                        print(f"Failed papers: {len(failed_papers)}")
                    continue
                elif question.lower() == 'papers':
                    processed_papers = get_processed_papers()
                    print("\nProcessed papers:")
                    for paper, info in processed_papers.items():
                        status = info.get('status', 'unknown')
                        print(f"  - {paper} ({status})")
                    continue
                elif question.lower().startswith('html '):
                    html_question = question[5:]
                    print(f"\nGenerating HTML report for: {html_question}")
                    
                    try:
                        result = subprocess.run([
                            'python', '/app/pipelines/neo4j_citation_hybrid_html.py',
                            '-q', html_question,
                            '-o', f'/app/output/report_{int(time.time())}.html'
                        ], capture_output=True, text=True, timeout=300, cwd='/app')
                        
                        if result.returncode == 0:
                            print("HTML report generated successfully!")
                            # Extract output file path from stdout if available
                            lines = result.stdout.strip().split('\n')
                            for line in lines:
                                if 'saved to' in line.lower() or '.html' in line:
                                    print(f"Report location: {line}")
                                    break
                        else:
                            print(f"Error generating HTML report: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        print("HTML generation timed out (took longer than 5 minutes)")
                    except Exception as e:
                        print(f"Error running HTML generation: {e}")
                    continue
                elif not question:
                    continue
                
                # Regular question - use the GRAPSE pipeline
                print(f"\nProcessing question: {question}")
                
                try:
                    result = subprocess.run([
                        'python', '/app/pipelines/grapse.py',
                        '-q', question
                    ], capture_output=True, text=True, timeout=300, cwd='/app')
                    
                    if result.returncode == 0:
                        print("\nAnswer:")
                        print(result.stdout)
                    else:
                        print(f"Error processing question: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("Question processing timed out (took longer than 5 minutes)")
                except Exception as e:
                    print(f"Error running GRAPSE pipeline: {e}")
                    
            except KeyboardInterrupt:
                logger.info("\nReceived interrupt signal. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")

def main():
    """Main function"""
    logger.info("Starting GRAPSE Chat Interface...")
    
    # Wait for Neo4j
    if not wait_for_neo4j():
        logger.error("Failed to connect to Neo4j. Exiting.")
        sys.exit(1)
    
    # Check if we have processed papers
    processed_papers = get_processed_papers()
    if not processed_papers:
        logger.warning("No processed papers found. Use the main GRAPSE container to process papers first.")
        print("\nNo processed papers found!")
        print("To process papers, run: docker compose up grapse")
        print("Then you can use this chat interface.")
        sys.exit(1)
    
    # Start interactive mode
    start_interactive_mode()

if __name__ == "__main__":
    main()