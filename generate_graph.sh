#!/bin/bash

# Define the output folder
OUTPUT_FOLDER="output"

# Loop over each folder in the output directory
echo "############# Generating graph for all papers in $OUTPUT_FOLDER ##############"
# Skip the first 6 papers
# Initialize a counter
counter=0
for folder in "$OUTPUT_FOLDER"/*; do
    if [ -d "$folder" ]; then
        # Extract the folder name
        folder_name=$(basename "$folder")
        # Increment the counter
        counter=$((counter + 1))
        
        # Skip the first 6 papers
        if [ "$counter" -le 41 ]; then
            continue
        fi

        # Print the folder name being processed
        echo "Processing paper: $folder_name"
        
        # Run the extract_and_transform.py script
        #python scripts/extract_and_transform.py -o "$OUTPUT_FOLDER" -p "$folder_name"
        
        # Run the load_data_into_graph_langchain.py script
        python scripts/load_data_into_graph_langchain.py -p "$folder_name"
    fi
done
echo "############# Graph generation complete ##############"
