import dedupe
from dotenv import load_dotenv
import os
import sys
from neo4j import GraphDatabase
import argparse
import pickle
# Disable warning
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

def fetch_data(entity_label):
    query = f"""
    MATCH (n:{entity_label})
    RETURN id(n) AS internalNeo4jId, n.id AS fullName
    """
    with driver.session() as session:
        results = session.run(query).data()

    data_dict = {}
    for row in results:
        record_id = row['internalNeo4jId']
        data_dict[record_id] = {
            "fullName": row['fullName']
        }
    return data_dict

def merge_cluster(cluster):
    if len(cluster) < 2:
        return
    #print(list(cluster))
    merge_query = """
    MATCH (n)
    WHERE n.id IN $nodeIds
    WITH collect(n) as nodes, head($nodeIds) as firstId
    CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true}) YIELD node
    SET node.id = firstId
    RETURN node
    """
    with driver.session() as session:
        session.run(merge_query, {"nodeIds": list(cluster)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedupe entities from Neo4j.")
    parser.add_argument("-l", "--label", type=str, help="The label of the entity to dedupe")
    args = parser.parse_args()

    entity_label = args.label
    data = fetch_data(entity_label)

    settings_file = f"{entity_label}_settings.dedupe"

    # Setup dedupe
    fields = [
        dedupe.variables.String("fullName")
    ]
    
    if os.path.exists(settings_file):
        with open(settings_file, "rb") as sf:
            deduper = dedupe.StaticDedupe(sf)
    else:
        deduper = dedupe.Dedupe(fields)
        deduper.prepare_training(data, sample_size=40, blocked_proportion=1.0)
        print("Please label the training data.")
        dedupe.console_label(deduper)
        deduper.train()
        with open(settings_file, "wb") as sf:
            deduper.write_settings(sf)

    # Cluster
    clustered_records = deduper.partition(data, threshold=0.5)

    # Merge each cluster in Neo4j
    for cluster in clustered_records:
        cluster_names = []
        for i in range(len(cluster[0])):
            record_id = cluster[0][i]
            confidence = cluster[1][i]
            #print(data[record_id]['fullName'] + f"({confidence})", end="| ")
            cluster_names.append(data[record_id]['fullName'])
        #print('\n')
        merge_cluster(cluster_names)