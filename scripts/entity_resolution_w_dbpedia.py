import os
import requests
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from argparse import ArgumentParser

# Load environment variables
load_dotenv(dotenv_path='../.env')
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(refresh_schema=False)

def get_dbpedia_uri_spotlight(text, confidence=0.5):
    """
    Calls the DBpedia Spotlight API to annotate the given text.
    Returns the first matching URI if found, otherwise None.
    """
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    params = {
        "text": text,
        "confidence": confidence,  
        # Optionally restrict types, e.g. "types": "DBpedia:Organisation"
    }
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # If there's at least one resource recognized, return its URI
            if "Resources" in data:
                # Heuristic: pick the first one or loop to refine
                return data["Resources"][0]["@URI"]
        else:
            print(f"DBpedia Spotlight error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception calling DBpedia Spotlight: {e}")

    return None

def get_records_with_lables(label):
    records = graph.query(f"""
    MATCH (i:`{label}`)
    WHERE i.dbpediaUri IS NULL
    RETURN id(i) as nodeId, i.id as entityId    
    """)
    return records

def add_dbpedia_uri_to_node(nodeId, dbpediaUri):
    graph.query(f"""
    MATCH (i)
    WHERE id(i) = {nodeId}
    SET i.dbpediaUri = "{dbpediaUri}"
    """)

def update_records_dbpedia(records, confidence=0.5):
    for record in records:
        entityId = record['entityId'].upper()
        dbpediaUri = get_dbpedia_uri_spotlight(entityId, confidence=confidence)
        if dbpediaUri:
            add_dbpedia_uri_to_node(record['nodeId'], dbpediaUri)
            print(f"Added dbpediaUri {dbpediaUri} to {entityId}")
        else:
            print(f"Could not find dbpediaUri for {entityId}")

def merge_entities_with_same_dbpediaUri(label):
    query = f"""
    MATCH (n:{label})
    WHERE n.dbpediaUri IS NOT NULL
    WITH n.dbpediaUri as uri, collect(n) as nodes
    WITH nodes, head(nodes) as firstNode
    CALL apoc.refactor.mergeNodes(nodes, {{properties: 'combine', mergeRels: true}}) YIELD node
    SET node.id = firstNode.id
    RETURN node
    """
    graph.query(query)

if __name__ == "__main__":
    # Parse arguments for entity label
    parser = ArgumentParser(description="Resolve entities with DBpedia.")
    parser.add_argument("-l", "--label", type=str, help="The label of the entity to resolve")
    args = parser.parse_args()
    label = args.label

    # Get and update records
    institution_records = get_records_with_lables(label)
    update_records_dbpedia(institution_records)
    merge_entities_with_same_dbpediaUri(label)
