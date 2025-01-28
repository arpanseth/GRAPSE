# GRAPSE
Graph-based Retrieval Agent for Process Systems Engineering

A chatbot that helps Process Systems Engineering researchers and students chat with academic papers in the feild. 

## Getting Started
We highly recommend using conda to build the python environment.
1. Clone repo and create conda envoronment:
```
git clone https://github.com/arpanseth/GRAPSE.git
conda env create -f environment.yml
conda activate grapse
```
2. Install and Run Neo4j Graph Database Self-Managed (Community Edition) (https://neo4j.com/deployment-center/). 
Note: GRAPSE has been built and tested on Neo4j Server Version: 5.25.1 (community).
Additionally install the following plugins for Neo4j:
- Graph Data Science
- APOC
- APOC-Extended

3. Make sure the Neo4j server is running and the UI can be accessed from your browser. You will need the server URL and credentials to run the chatbot. Here are some example credentials:
- NEO4J_URI="bolt://localhost:7687"
- NEO4J_USERNAME="neo4j"
- NEO4J_PASSWORD="neo4jpassword"

4. Create and .env file in root GRAPSE directory with the following variables:
- OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
- NEO4J_URI="YOUR_NEO4J_BOLT_URI"
- NEO4J_USERNAME="YOUR_NEO4J_USERNAME"
- NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"

6. Download pdf copies of the papers you want to include for the chatbot and save them in the [papers](papers) folder.
7. Run pdf data extraction using magic-pdf:
```
magic-pdf -p papers -o output -m auto
```
8. Run Knowledge Graph Building script:
``` 
./generate_graph.sh
```
9. Run Knowledge Graph Enrichment script:
```
./enrich_graph.sh
```
10. Now you can use the chatbot from the commandline interface:
```
python pipelines/neo4j_global.py -q "What are some algorithms for solving MINLP?"
```
