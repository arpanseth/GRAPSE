# GRAPSE
Graph-based Retrieval Agent for Process Systems Engineering

A chatbot that helps Process Systems Engineering researchers and students chat with academic papers in the field.

## Quick Start with Docker (Recommended)

Get GRAPSE running in 3 simple steps:

### 1. Setup Environment
```bash
git clone https://github.com/arpanseth/GRAPSE.git
cd GRAPSE
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Add Your Papers
```bash
# Copy your PDF papers to the papers folder
cp your_papers/*.pdf papers/
```

### 3. Start GRAPSE
```bash
docker compose up --build
```

That's it! GRAPSE will automatically:
- Start Neo4j with all required plugins
- Process your papers and build the knowledge graph
- Start an interactive chat interface

You can then ask questions like:
- "What are some algorithms for solving MINLP?"
- "Tell me about process optimization methods"
- "What is model predictive control?"

### Adding More Papers

To add more papers after startup:
```bash
# Copy new papers to the papers folder
cp new_papers/*.pdf papers/

# They will be automatically processed within 10 seconds
```

### Different Usage Modes

**Standard mode** (processes papers once, then interactive chat):
```bash
docker compose up
```

**With continuous watching** (automatically processes new papers):
```bash
docker compose --profile with-watcher up
```

**Chat only** (if papers already processed):
```bash
docker compose --profile chat-only up grapse-chat
```

---

## Manual Installation (Advanced Users)

If you prefer to install manually without Docker:

### Prerequisites
1. Install conda and create environment:
```bash
conda env create -f environment.yml
conda activate grapse
```

2. Install and run Neo4j Graph Database Self-Managed (Community Edition) 
   - Download from: https://neo4j.com/deployment-center/
   - Tested with Neo4j Server Version: 5.25.1 (community)
   - Required plugins: Graph Data Science, APOC, APOC-Extended

3. Create `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Processing Papers
1. Add PDFs to `papers/` folder
2. Extract data: `magic-pdf -p papers -o output -m auto`
3. Build graph: `./generate_graph.sh`
4. Enrich graph: `./enrich_graph.sh`
5. Chat: `python pipelines/grapse.py -q "Your question"`
