import os
import requests
from dotenv import load_dotenv
from graphdatascience import GraphDataScience
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
import numpy as np
import pandas as pd
import argparse

# Load environment variables
load_dotenv(dotenv_path='../.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

# Initialize GraphDataScience
gds = GraphDataScience(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

graph = Neo4jGraph()


# Function to prepare string from community data
def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

# Function to process community
def process_community(community, community_chain):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary}

def main(levels_list):
    # Create a graph projection
    G, result = gds.graph.project(
        "communities",  # Graph name
        "__Entity__",  # Node projection
        {
            "_ALL_": {
                "type": "*",
                "orientation": "UNDIRECTED",
                "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
            }
        },
    )

    # Leiden community detection
    gds.leiden.write(
        G,
        writeProperty="communities",
        includeIntermediateCommunities=True,
        relationshipWeightProperty="weight",
    )

    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

    graph.query("""
        MATCH (e:`__Entity__`)
        UNWIND range(0, size(e.communities) - 1 , 1) AS index
        CALL {
        WITH e, index
        WITH e, index
        WHERE index = 0
        MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET c.level = index
        MERGE (e)-[:IN_COMMUNITY]->(c)
        RETURN count(*) AS count_0
        }
        CALL {
        WITH e, index
        WITH e, index
        WHERE index > 0
        MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
        ON CREATE SET current.level = index
        MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
        ON CREATE SET previous.level = index - 1
        MERGE (previous)-[:IN_COMMUNITY]->(current)
        RETURN count(*) AS count_1
        }
        RETURN count(*)
        """)
    
    graph.query("""
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
        WITH c, count(distinct d) AS rank
        SET c.community_rank = rank;
        """)
    
    # Community detection and summarization
    community_info = gds.run_cypher_query(f"""
    MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
    WHERE c.level IN {levels_list}
    WITH c, collect(e) AS nodes
    WHERE size(nodes) > 1
    CALL apoc.path.subgraphAll(nodes[0], {{
        whitelistNodes:nodes
    }})
    YIELD relationships
    RETURN c.id AS communityId,
           [n in nodes | {{id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}}] AS nodes,
           [r in relationships | {{start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}}] AS rels
    """)

    # Process each community
    summaries = []
    for community in community_info:
        summary = process_community(community)
        summaries.append(summary)

    # Store summaries
    gds.run_cypher_query("""
    UNWIND $data AS row
    MERGE (c:__Community__ {id:row.community})
    SET c.summary = row.summary
    """, params={"data": summaries})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community Building Script")
    parser.add_argument("-l", "--levels-list", type=str, required=True, help="List of levels for community_info query")
    args = parser.parse_args()
    main(args.levels_list) 