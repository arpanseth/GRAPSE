# GRAPSE Quick Start Guide

Get GRAPSE running in just 3 steps!

## Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

## Step 1: Setup
```bash
git clone https://github.com/arpanseth/GRAPSE.git
cd GRAPSE
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Step 2: Add Papers
```bash
# Copy your PDF papers to the papers folder
cp your_papers/*.pdf papers/
```

## Step 3: Start GRAPSE
```bash
docker compose up --build
```

That's it! GRAPSE will:
1. ✅ Start Neo4j with all required plugins
2. ✅ Process your papers automatically  
3. ✅ Build a knowledge graph
4. ✅ Start an interactive chat interface

## What You Can Do

### Ask Questions
Once running, you can ask questions like:
- "What are some algorithms for solving MINLP?"
- "Tell me about process optimization methods" 
- "What is model predictive control?"

### Special Commands
- `help` - Show available commands
- `status` - Show processing status
- `papers` - List processed papers
- `html <question>` - Generate HTML report
- `exit` - Exit GRAPSE

### Add More Papers
While GRAPSE is running, just copy new PDFs to the `papers/` folder:
```bash
cp new_papers/*.pdf papers/
```
They'll be processed automatically within 10 seconds!

## Different Modes

**Standard Mode** (process papers once, then chat):
```bash
docker compose up
```

**With File Watching** (continuously watch for new papers):
```bash
docker compose --profile with-watcher up
```

**Chat Only** (if papers already processed):
```bash
docker compose --profile chat-only up grapse-chat
```

## Troubleshooting

**Test your setup:**
```bash
./test_docker.sh
```

**Check logs:**
```bash
docker compose logs grapse
docker compose logs neo4j
```

**Reset everything:**
```bash
docker compose down -v
```

## Tips
- Processing time depends on paper count and size
- First run takes longer (downloads images, installs dependencies)
- HTML reports are saved to `output/` folder
- All data persists in Docker volumes

## Need Help?
- Check the logs with `docker compose logs`
- Run the test script: `./test_docker.sh`
- Open an issue on GitHub