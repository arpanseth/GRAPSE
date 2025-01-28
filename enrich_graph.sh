echo "############# Starting Graph Enrichment ##############"
echo "Running dedupe..."
python scripts/entity_dedupe_w_training.py -l Author
python scripts/entity_dedupe_w_training.py -l Benchmark
echo "############# Deduplication complete ##############"
echo "####### Starting Graph Community Detection and Summarization#######"
python scripts/community_building.py -l "[0,1]"
echo "############# Community Detection and Summarization complete ##############"
echo "############# Graph Enrichment complete ##############"