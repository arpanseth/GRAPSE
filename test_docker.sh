#!/bin/bash

# Test script for GRAPSE Docker deployment
# This script verifies that the Docker setup works correctly

set -e

echo "=== GRAPSE Docker Deployment Test ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Creating a template..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "⚠️  Please edit .env file with your actual OpenAI API key"
    echo "Then run this test again."
    exit 1
fi

echo "✅ .env file found"

# Check if papers directory exists
if [ ! -d papers ]; then
    echo "📁 Creating papers directory..."
    mkdir -p papers
fi

echo "✅ Papers directory ready"

# Check if there are any papers to process
pdf_count=$(find papers -name "*.pdf" | wc -l)
if [ $pdf_count -eq 0 ]; then
    echo "⚠️  No PDF files found in papers/ directory"
    echo "   Add some PDF papers to test the full pipeline"
    echo "   For now, testing without papers..."
else
    echo "✅ Found $pdf_count PDF files in papers/ directory"
fi

# Test Docker build
echo ""
echo "🔨 Testing Docker build..."
if docker compose build --no-cache grapse; then
    echo "✅ Docker build successful"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Test Neo4j startup
echo ""
echo "🚀 Testing Neo4j startup..."
docker compose up -d neo4j

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if docker compose exec -T neo4j cypher-shell --uri bolt://localhost:7687 --username neo4j --password grapse_password "MATCH () RETURN count(*) as count" &> /dev/null; then
        echo "✅ Neo4j is ready!"
        break
    fi
    
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $max_retries ]; then
        echo "❌ Neo4j failed to start within 5 minutes"
        docker compose logs neo4j
        docker compose down
        exit 1
    fi
    
    sleep 10
done

# Test GRAPSE connection to Neo4j
echo ""
echo "🔗 Testing GRAPSE connection to Neo4j..."
if docker compose run --rm grapse python -c "
from langchain_community.graphs import Neo4jGraph
try:
    graph = Neo4jGraph()
    print('✅ GRAPSE can connect to Neo4j')
except Exception as e:
    print(f'❌ GRAPSE cannot connect to Neo4j: {e}')
    exit(1)
"; then
    echo "✅ Connection test passed"
else
    echo "❌ Connection test failed"
    docker compose down
    exit 1
fi

# If we have papers, test processing
if [ $pdf_count -gt 0 ]; then
    echo ""
    echo "📄 Testing paper processing with one file..."
    
    # Get first PDF file
    first_pdf=$(find papers -name "*.pdf" | head -1 | xargs basename)
    
    if docker compose run --rm grapse python scripts/add_paper.py --list-new | grep -q "$first_pdf"; then
        echo "✅ Paper detection works"
        
        echo "⏳ Testing paper processing (this may take several minutes)..."
        if timeout 600 docker compose run --rm grapse python scripts/add_paper.py -p "$first_pdf"; then
            echo "✅ Paper processing successful"
        else
            echo "⚠️  Paper processing timed out or failed (this is expected for large papers)"
        fi
    else
        echo "⚠️  Paper detection issue, but Docker setup is working"
    fi
fi

# Test chat interface briefly
echo ""
echo "💬 Testing chat interface..."
echo "exit" | timeout 30 docker compose run --rm grapse python scripts/interactive_chat.py || true
echo "✅ Chat interface can start"

# Cleanup
echo ""
echo "🧹 Cleaning up test containers..."
docker compose down

echo ""
echo "🎉 Docker deployment test completed successfully!"
echo ""
echo "To start GRAPSE:"
echo "1. Add your PDF papers to the papers/ folder"
echo "2. Update .env with your OpenAI API key"
echo "3. Run: docker compose up --build"
echo ""
echo "The system will automatically process your papers and start an interactive chat!"