#!/usr/bin/env python3
"""
Watch papers directory for new files and automatically process them
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHER] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/watcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PaperHandler(FileSystemEventHandler):
    """Handle file system events for papers directory"""
    
    def __init__(self):
        self.processing_queue = set()
        self.cooldown_time = 10  # Wait 10 seconds before processing
        self.pending_files = {}  # filename -> timestamp
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.pdf':
            logger.info(f"New PDF detected: {file_path.name}")
            self.pending_files[file_path.name] = time.time()
    
    def on_moved(self, event):
        """Handle file move events (like copy completion)"""
        if event.is_directory:
            return
        
        dest_path = Path(event.dest_path)
        if dest_path.suffix.lower() == '.pdf':
            logger.info(f"PDF moved/copied: {dest_path.name}")
            self.pending_files[dest_path.name] = time.time()
    
    def process_pending_files(self):
        """Process files that have been stable for the cooldown period"""
        current_time = time.time()
        files_to_process = []
        
        for filename, timestamp in list(self.pending_files.items()):
            if current_time - timestamp > self.cooldown_time:
                files_to_process.append(filename)
                del self.pending_files[filename]
        
        for filename in files_to_process:
            if filename not in self.processing_queue:
                self.processing_queue.add(filename)
                self.process_paper(filename)
                self.processing_queue.discard(filename)
    
    def process_paper(self, filename):
        """Process a single paper using add_paper.py"""
        logger.info(f"Processing paper: {filename}")
        
        try:
            result = subprocess.run([
                'python', '/app/scripts/add_paper.py',
                '-p', filename
            ], capture_output=True, text=True, timeout=1800, cwd='/app')
            
            if result.returncode == 0:
                logger.info(f"Successfully processed paper: {filename}")
                logger.info(result.stdout)
            else:
                logger.error(f"Failed to process paper {filename}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing paper: {filename}")
        except Exception as e:
            logger.error(f"Error processing paper {filename}: {e}")

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

def check_existing_papers():
    """Check for papers that exist but haven't been processed"""
    papers_dir = Path('/app/papers')
    if not papers_dir.exists():
        logger.warning("Papers directory not found")
        return
    
    pdf_files = list(papers_dir.glob('*.pdf'))
    processed_papers = get_processed_papers()
    
    unprocessed_papers = []
    for pdf_file in pdf_files:
        if pdf_file.name not in processed_papers:
            unprocessed_papers.append(pdf_file.name)
    
    if unprocessed_papers:
        logger.info(f"Found {len(unprocessed_papers)} unprocessed papers")
        
        # Ask if user wants to process them
        try:
            response = input(f"Process {len(unprocessed_papers)} existing unprocessed papers? (y/n): ").lower()
            if response in ['y', 'yes']:
                for paper in unprocessed_papers:
                    logger.info(f"Processing existing paper: {paper}")
                    
                    try:
                        result = subprocess.run([
                            'python', '/app/scripts/add_paper.py',
                            '-p', paper
                        ], capture_output=True, text=True, timeout=1800, cwd='/app')
                        
                        if result.returncode == 0:
                            logger.info(f"Successfully processed: {paper}")
                        else:
                            logger.error(f"Failed to process {paper}: {result.stderr}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {paper}: {e}")
        except KeyboardInterrupt:
            logger.info("Skipping existing papers")

def main():
    """Main function"""
    logger.info("Starting GRAPSE paper watcher...")
    
    # Create papers directory if it doesn't exist
    papers_dir = Path('/app/papers')
    papers_dir.mkdir(exist_ok=True)
    
    # Check for existing unprocessed papers
    check_existing_papers()
    
    # Set up file system watcher
    event_handler = PaperHandler()
    observer = Observer()
    observer.schedule(event_handler, str(papers_dir), recursive=False)
    
    try:
        observer.start()
        logger.info(f"Watching for new papers in {papers_dir}")
        logger.info("Drop PDF files into the papers/ directory to process them automatically")
        
        while True:
            time.sleep(5)  # Check every 5 seconds
            event_handler.process_pending_files()
            
    except KeyboardInterrupt:
        logger.info("Stopping paper watcher...")
        observer.stop()
    except Exception as e:
        logger.error(f"Error in paper watcher: {e}")
        observer.stop()
    
    observer.join()
    logger.info("Paper watcher stopped")

if __name__ == "__main__":
    main()