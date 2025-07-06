#!/usr/bin/env python3
"""
Add a single paper to GRAPSE
For incremental paper processing
"""

import os
import sys
import time
import json
import subprocess
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ADD_PAPER] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/add_paper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def run_magic_pdf(paper_path):
    """Run magic-pdf extraction for a specific paper"""
    logger.info(f"Running magic-pdf for {paper_path}")
    
    try:
        result = subprocess.run([
            'magic-pdf', 
            '-p', str(paper_path),
            '-o', '/app/output',
            '-m', 'auto'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"Magic-pdf completed for {paper_path}")
            return True
        else:
            logger.error(f"Magic-pdf failed for {paper_path}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Magic-pdf timeout for {paper_path}")
        return False
    except Exception as e:
        logger.error(f"Magic-pdf error for {paper_path}: {e}")
        return False

def run_graph_building(paper_folder):
    """Run graph building for a specific paper folder"""
    logger.info(f"Building graph for {paper_folder}")
    
    try:
        result = subprocess.run([
            'python', '/app/scripts/load_data_into_graph_langchain.py',
            '-p', paper_folder
        ], capture_output=True, text=True, timeout=1200, cwd='/app')
        
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

def run_incremental_enrichment(paper_folder):
    """Run incremental enrichment for new entities"""
    logger.info(f"Running incremental enrichment for {paper_folder}")
    
    try:
        # Only run entity deduplication for the new paper
        # We'll create a lightweight version that only processes new entities
        result = subprocess.run([
            'python', '/app/scripts/incremental_enrich.py',
            '-p', paper_folder
        ], capture_output=True, text=True, timeout=600, cwd='/app')
        
        if result.returncode == 0:
            logger.info(f"Incremental enrichment completed for {paper_folder}")
            return True
        else:
            logger.warning(f"Incremental enrichment failed for {paper_folder}: {result.stderr}")
            return False  # Not critical, so we continue
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Incremental enrichment timeout for {paper_folder}")
        return False
    except Exception as e:
        logger.warning(f"Incremental enrichment error for {paper_folder}: {e}")
        return False

def add_paper(paper_name):
    """Add a single paper to GRAPSE"""
    processed_papers = get_processed_papers()
    
    # Check if already processed
    if paper_name in processed_papers:
        logger.info(f"Paper {paper_name} already processed")
        return True
    
    paper_path = Path('/app/papers') / paper_name
    if not paper_path.exists():
        logger.error(f"Paper not found: {paper_path}")
        return False
    
    logger.info(f"Processing new paper: {paper_name}")
    
    # Step 1: Magic-PDF extraction
    if not run_magic_pdf(paper_path):
        processed_papers[paper_name] = {
            'processed_at': time.time(),
            'status': 'magic_pdf_failed'
        }
        save_processed_papers(processed_papers)
        return False
    
    # Step 2: Find the output folder
    paper_name_no_ext = paper_name.replace('.pdf', '')
    output_folders = list(Path('/app/output').glob(f'{paper_name_no_ext}*'))
    
    if not output_folders:
        logger.error(f"Output folder not found for {paper_name}")
        processed_papers[paper_name] = {
            'processed_at': time.time(),
            'status': 'output_folder_not_found'
        }
        save_processed_papers(processed_papers)
        return False
    
    paper_folder = output_folders[0].name
    
    # Step 3: Graph building
    if not run_graph_building(paper_folder):
        processed_papers[paper_name] = {
            'processed_at': time.time(),
            'output_folder': paper_folder,
            'status': 'graph_failed'
        }
        save_processed_papers(processed_papers)
        return False
    
    # Step 4: Incremental enrichment (optional)
    enrichment_success = run_incremental_enrichment(paper_folder)
    
    # Mark as completed
    processed_papers[paper_name] = {
        'processed_at': time.time(),
        'output_folder': paper_folder,
        'status': 'completed',
        'enrichment_status': 'completed' if enrichment_success else 'failed'
    }
    save_processed_papers(processed_papers)
    
    logger.info(f"Successfully added paper: {paper_name}")
    return True

def list_new_papers():
    """List papers that haven't been processed yet"""
    papers_dir = Path('/app/papers')
    if not papers_dir.exists():
        logger.warning("Papers directory not found")
        return []
    
    pdf_files = list(papers_dir.glob('*.pdf'))
    processed_papers = get_processed_papers()
    
    new_papers = []
    for pdf_file in pdf_files:
        if pdf_file.name not in processed_papers:
            new_papers.append(pdf_file.name)
    
    return new_papers

def main():
    parser = argparse.ArgumentParser(description='Add a paper to GRAPSE')
    parser.add_argument('-p', '--paper', type=str, help='Paper filename to add')
    parser.add_argument('--list-new', action='store_true', help='List unprocessed papers')
    parser.add_argument('--process-all-new', action='store_true', help='Process all new papers')
    parser.add_argument('--status', action='store_true', help='Show processing status')
    
    args = parser.parse_args()
    
    if args.list_new:
        new_papers = list_new_papers()
        if new_papers:
            print("Unprocessed papers:")
            for paper in new_papers:
                print(f"  - {paper}")
        else:
            print("No new papers found")
        return
    
    if args.status:
        processed_papers = get_processed_papers()
        print(f"Total processed papers: {len(processed_papers)}")
        for paper, info in processed_papers.items():
            status = info.get('status', 'unknown')
            processed_time = info.get('processed_at', 0)
            if processed_time:
                processed_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processed_time))
            else:
                processed_str = 'unknown'
            print(f"  {paper}: {status} (processed: {processed_str})")
        return
    
    if args.process_all_new:
        new_papers = list_new_papers()
        if not new_papers:
            print("No new papers to process")
            return
        
        print(f"Processing {len(new_papers)} new papers...")
        success_count = 0
        
        for paper in new_papers:
            if add_paper(paper):
                success_count += 1
            else:
                print(f"Failed to process: {paper}")
        
        print(f"Successfully processed {success_count}/{len(new_papers)} papers")
        return
    
    if not args.paper:
        parser.print_help()
        return
    
    if add_paper(args.paper):
        print(f"Successfully added paper: {args.paper}")
    else:
        print(f"Failed to add paper: {args.paper}")
        sys.exit(1)

if __name__ == "__main__":
    main()