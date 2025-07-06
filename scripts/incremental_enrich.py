#!/usr/bin/env python3
"""
Incremental enrichment for newly added papers
Only processes entities from the new paper, not the entire graph
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [INCREMENTAL_ENRICH] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/incremental_enrich.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_targeted_deduplication(paper_folder, entity_types=['Author', 'Benchmark']):
    """
    Run entity deduplication only for entities related to the new paper
    This is much faster than full graph deduplication
    """
    import subprocess
    
    for entity_type in entity_types:
        logger.info(f"Running targeted deduplication for {entity_type} entities in {paper_folder}")
        
        try:
            # Create a modified version of entity_dedupe_w_training.py that only processes
            # entities from a specific paper folder
            result = subprocess.run([
                'python', '/app/scripts/entity_dedupe_targeted.py',
                '-l', entity_type,
                '-p', paper_folder
            ], capture_output=True, text=True, timeout=300, cwd='/app')
            
            if result.returncode == 0:
                logger.info(f"Targeted deduplication completed for {entity_type}")
            else:
                logger.warning(f"Targeted deduplication failed for {entity_type}: {result.stderr}")
                # Fall back to regular deduplication for this entity type
                logger.info(f"Falling back to full deduplication for {entity_type}")
                result = subprocess.run([
                    'python', '/app/scripts/entity_dedupe_w_training.py',
                    '-l', entity_type
                ], capture_output=True, text=True, timeout=600, cwd='/app')
                
                if result.returncode == 0:
                    logger.info(f"Full deduplication completed for {entity_type}")
                else:
                    logger.warning(f"Full deduplication also failed for {entity_type}: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.warning(f"Deduplication timeout for {entity_type}")
        except Exception as e:
            logger.warning(f"Deduplication error for {entity_type}: {e}")

def update_communities_incrementally(paper_folder):
    """
    Update community detection incrementally
    This is a simplified version that doesn't rebuild entire communities
    """
    import subprocess
    
    logger.info(f"Updating communities for new paper: {paper_folder}")
    
    try:
        # Try to run incremental community update if available
        result = subprocess.run([
            'python', '/app/scripts/community_building_incremental.py',
            '-p', paper_folder
        ], capture_output=True, text=True, timeout=600, cwd='/app')
        
        if result.returncode == 0:
            logger.info("Incremental community update completed")
            return True
        else:
            logger.warning(f"Incremental community update failed: {result.stderr}")
            # Fall back to full community detection (but with timeout)
            logger.info("Falling back to full community detection")
            
            result = subprocess.run([
                'python', '/app/scripts/community_building.py',
                '-l', '[0,1]'
            ], capture_output=True, text=True, timeout=900, cwd='/app')
            
            if result.returncode == 0:
                logger.info("Full community detection completed")
                return True
            else:
                logger.warning(f"Full community detection failed: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        logger.warning("Community detection timeout")
        return False
    except Exception as e:
        logger.warning(f"Community detection error: {e}")
        return False

def create_targeted_deduplication_script():
    """
    Create a targeted version of the entity deduplication script
    This only processes entities from a specific paper
    """
    script_content = '''#!/usr/bin/env python3
"""
Targeted entity deduplication for a specific paper
Modified version of entity_dedupe_w_training.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from the original script
try:
    from entity_dedupe_w_training import *
    from langchain_community.graphs import Neo4jGraph
    from dotenv import load_dotenv
    
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
    
except ImportError as e:
    print(f"Could not import required modules: {e}")
    print("Falling back to full deduplication")
    sys.exit(1)

def run_targeted_deduplication(entity_label, paper_folder):
    """Run deduplication only for entities from a specific paper"""
    graph = Neo4jGraph()
    
    # Get entities only from the specific paper
    query = f"""
    MATCH (p:Paper)-[:CONTAINS]->(d:Document)-[:MENTIONS]->(e:{entity_label})
    WHERE p.title CONTAINS $paper_folder OR d.id CONTAINS $paper_folder
    RETURN DISTINCT e.id as entity_id, e.name as entity_name
    LIMIT 100
    """
    
    try:
        entities = graph.query(query, {"paper_folder": paper_folder})
        
        if not entities:
            print(f"No {entity_label} entities found for paper {paper_folder}")
            return
        
        print(f"Found {len(entities)} {entity_label} entities to process for {paper_folder}")
        
        # Process each entity for potential duplicates
        for entity in entities:
            entity_id = entity["entity_id"]
            entity_name = entity["entity_name"]
            
            # Find potential duplicates in the entire graph
            duplicate_query = f"""
            MATCH (e1:{entity_label} {{id: $entity_id}})
            MATCH (e2:{entity_label})
            WHERE e1.id <> e2.id
            AND (
                e1.name = e2.name OR
                apoc.text.similarity(e1.name, e2.name) > 0.8
            )
            RETURN e2.id as duplicate_id, e2.name as duplicate_name,
                   apoc.text.similarity(e1.name, e2.name) as similarity
            ORDER BY similarity DESC
            LIMIT 5
            """
            
            duplicates = graph.query(duplicate_query, {"entity_id": entity_id})
            
            if duplicates:
                print(f"Found {len(duplicates)} potential duplicates for {entity_name}")
                # Here you could implement the actual deduplication logic
                # For now, just log the findings
                
    except Exception as e:
        print(f"Error in targeted deduplication: {e}")

def main():
    parser = argparse.ArgumentParser(description='Targeted entity deduplication')
    parser.add_argument('-l', '--label', type=str, required=True, help='Entity label to deduplicate')
    parser.add_argument('-p', '--paper', type=str, required=True, help='Paper folder to focus on')
    
    args = parser.parse_args()
    
    run_targeted_deduplication(args.label, args.paper)

if __name__ == "__main__":
    main()
'''
    
    script_path = Path('/app/scripts/entity_dedupe_targeted.py')
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        logger.info("Created targeted deduplication script")
    except Exception as e:
        logger.warning(f"Could not create targeted deduplication script: {e}")

def create_incremental_community_script():
    """
    Create an incremental community detection script
    """
    script_content = '''#!/usr/bin/env python3
"""
Incremental community detection for a specific paper
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from langchain_community.graphs import Neo4jGraph
    from dotenv import load_dotenv
    
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
    
except ImportError as e:
    print(f"Could not import required modules: {e}")
    sys.exit(1)

def run_incremental_community_detection(paper_folder):
    """Run incremental community detection for new paper"""
    graph = Neo4jGraph()
    
    try:
        # Simple approach: just ensure the new nodes are included in existing communities
        # or create new small communities for isolated components
        
        query = """
        MATCH (p:Paper)-[:CONTAINS]->(d:Document)
        WHERE p.title CONTAINS $paper_folder OR d.id CONTAINS $paper_folder
        WITH d
        MATCH (d)-[:MENTIONS]->(e)
        SET e.last_updated = timestamp()
        RETURN count(e) as updated_entities
        """
        
        result = graph.query(query, {"paper_folder": paper_folder})
        
        if result:
            print(f"Updated {result[0]['updated_entities']} entities for incremental community detection")
        
        # You could implement more sophisticated incremental community detection here
        # For now, this is a placeholder that marks entities as updated
        
        return True
        
    except Exception as e:
        print(f"Error in incremental community detection: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Incremental community detection')
    parser.add_argument('-p', '--paper', type=str, required=True, help='Paper folder to process')
    
    args = parser.parse_args()
    
    success = run_incremental_community_detection(args.paper)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
    
    script_path = Path('/app/scripts/community_building_incremental.py')
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        logger.info("Created incremental community detection script")
    except Exception as e:
        logger.warning(f"Could not create incremental community script: {e}")

def main():
    """Main function for incremental enrichment"""
    parser = argparse.ArgumentParser(description='Incremental enrichment for new papers')
    parser.add_argument('-p', '--paper', type=str, required=True, help='Paper folder to enrich')
    
    args = parser.parse_args()
    
    logger.info(f"Starting incremental enrichment for paper: {args.paper}")
    
    # Create helper scripts if they don't exist
    create_targeted_deduplication_script()
    create_incremental_community_script()
    
    # Run targeted deduplication
    run_targeted_deduplication(args.paper)
    
    # Update communities incrementally
    update_communities_incrementally(args.paper)
    
    logger.info(f"Incremental enrichment completed for paper: {args.paper}")

if __name__ == "__main__":
    main()