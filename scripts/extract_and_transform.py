import json
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import textwrap, re
from argparse import ArgumentParser
import pickle


CHUNK_AVERAGE_SIZE = 5000
CHUNK_OVERLAP_LINES = 3
OUTPUT_FOLDER = 'output'

def get_sections_from_json(json_data):
    # Loop through all each line and look for when tex_level changes to 1
    sections = []
    section_change = False
    section_text = ''
    line_to_section = {}
    s = 0
    for i in range(len(json_data)):
        line = json_data[i]
        if ('text_level' in line.keys() and line['text_level'] == 1) or (i == len(json_data)-1 and 'text' in line.keys()):
            section_change = True
            # Append previous section text
            sections.append(section_text)
            # Set new section
            section_text = line['text']
            line_to_section[i] = s
        else:
            section_change = False
        if not section_change:
            if line['type'] == 'text' or line['type'] == 'equation':
                section_text +=  ' ' + line['text']
                line_to_section[i] = s
        else:
            s += 1
    sections = [i for i in sections if len(i) > 0]
    return sections, line_to_section

def extract_chunks_from_sections(sections):
    chunk_list = []
    chunk_to_section = {}
    c = 0 # Chunk counter
    for s in range(len(sections)):
        section = sections[s]
        if len(section) > CHUNK_AVERAGE_SIZE*1.5:
            num_chunks = int(len(section)/CHUNK_AVERAGE_SIZE) + 1
            chunk_size = int(len(section)/num_chunks)
            #print(num_chunks, chunk_size, len(section))
            # Loop through all chunks except the last one
            for j in range(num_chunks - 1):
                chunk = section[j*chunk_size : (j+1)*chunk_size]
                #chunk_list.append(chunk)
                # Get the previous and next 3 sentences 
                # TODO: What if there are not enough sentences in the next chunk
                if j == 0:
                    previous_overlap = ''
                else:
                    previous_chunk = section[(j-1)*chunk_size : (j)*chunk_size]
                    # Use NLTK Sentence tokenizer
                    previous_chunk_sentences = sent_tokenize(previous_chunk)
                    # Make sure there are at-least 3 sentences
                    if len(previous_chunk_sentences) >= CHUNK_OVERLAP_LINES:
                        previous_overlap = ' '.join(previous_chunk_sentences[-CHUNK_OVERLAP_LINES:])
                    else:
                        previous_overlap = ' '.join(previous_chunk_sentences)
                # Add previous overlap to the section
                chunk = previous_overlap + chunk
        
                # Identify 3 sentences from the next chunk NOTE: We get the full chunk because we don;t know if its the last one
                next_chunk = section[(j+1)*chunk_size:]
                next_chunk_sentences = sent_tokenize(next_chunk)
                if len(next_chunk_sentences) >= CHUNK_OVERLAP_LINES:
                    next_overlap = ''.join(next_chunk_sentences[:CHUNK_OVERLAP_LINES])
                else:
                    next_overlap = ''.join(next_chunk_sentences)
                chunk += next_overlap
                # Add chunk to list
                chunk_list.append(chunk)
                chunk_to_section[c] = s
                c += 1
            # Handle the last chunk till the end of text
            previous_chunk = section[(num_chunks-2)*chunk_size : (num_chunks-1)*chunk_size]
            # Use NLTK Sentence tokenizer
            previous_chunk_sentences = sent_tokenize(previous_chunk)
            # Make sure there are at-least 3 sentences
            if len(previous_chunk_sentences) >= CHUNK_OVERLAP_LINES:
                previous_overlap = ''.join(previous_chunk_sentences[-CHUNK_OVERLAP_LINES:])
            else:
                previous_overlap = ''.join(previous_chunk_sentences)
            last_cunk = previous_overlap + section[(num_chunks - 1)*chunk_size:]
            chunk_list.append(last_cunk)
            chunk_to_section[c] = s
            c += 1 
        else:
            chunk_list.append(section)
            chunk_to_section[c] = s
            c += 1
    return chunk_list, chunk_to_section

def connect_images_to_chunks(json_data, chunk_list, line_to_section, chunk_to_section):
    # Regex pattern to match variations
    pattern = r"^(?:Fig(?:\.|\b)|Figure|fig)(?:\.|\b)\s*\d+"
    # Loop through all images in the json and get their figure labels
    figure_label_to_image_path = {}
    line_index_to_image_path = {}
    for i in range(len(json_data)):
        line = json_data[i]
        # All images that have a caption
        if line['type'] == 'image' and len(line['img_caption']) > 0:
            match = re.match(pattern, line['img_caption'][0])
            image_path = line['img_path']
            if match:
                # Save image label and path in dictionary
                figure_label_to_image_path[match.group()] = image_path
            else:
                # If image does not have a caption, we just connect it to the preceding and succeding line
                print(f'The image caption did not have any matching pattern in it: ', line['img_caption'][0])
                if i > 0: line_index_to_image_path[i-1] = image_path
                if i < len(json_data): line_index_to_image_path[i+1] = image_path
        elif line['type'] == 'image':
            image_path = line['img_path']
            # If the image does not have any caption we just connect it to the preceding and succeding line 
            print(f'Image has no caption: ', line['img_path'])
            if i > 0: line_index_to_image_path[i-1] = image_path
            if i < len(json_data): line_index_to_image_path[i+1] = image_path
    # Find figure labels in each chunk and connect them
    chunk_to_image_path = set()
    for key,val in figure_label_to_image_path.items():
        fig_found = False
        # Extract the figure number from the key
        match_key = re.search(r'\d+', key)
        if not match_key:
            continue
        fig_num = match_key.group()
        for chunk_id in range(len(chunk_list)):
            # Normalize the chunk text
            chunk_text = chunk_list[chunk_id].lower()
            # Replace "figure" and "fig." with "fig"
            chunk_text = re.sub(r'figure', 'fig', chunk_text, flags=re.IGNORECASE)
            chunk_text = re.sub(r'fig\.', 'fig', chunk_text, flags=re.IGNORECASE)

            # Now check if something like "fig 1" exists
            pattern = rf'\bfig\s*{fig_num}\b'
            if re.search(pattern, chunk_text):
            #if key in chunk_list[chunk_id]:
                fig_found = True
                chunk_to_image_path.add((chunk_id, key, val))
        if not fig_found:
            print(f'Reference for {key} was not cound in any chunk.')

    # Add all the remaining ones that did not have a reference
    for key,val in line_index_to_image_path.items():
        try:
            section = line_to_section[key]
        except:
            continue
        chunks = [c for c,s in chunk_to_section.items() if s == section]
        for j in chunks:
            chunk_to_image_path.add((j, None, val))
    
    return chunk_to_image_path

def add_full_path(path, output_folder, paper_name):
    return f'{output_folder}/{paper_name}/auto/' + path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o','--output-folder', 
                        help='Path to the folder where the output from MagicPDF is stored', 
                        required=True, default='output')
    parser.add_argument('-p','--paper', 
                        help='Name of the paper you want to extract and transoform. Should be the folder name in output-folder.', 
                        required=True)
    args = vars(parser.parse_args())
    output_folder = args['output_folder']
    paper_name = args['paper']
    # REad json data
    paper_json_path = f'{output_folder}/{paper_name}/auto/{paper_name}_content_list.json'
    json_data = json.load(open(paper_json_path,'r'))
    # Extract sections
    sections, line_to_section = get_sections_from_json(json_data)
    # Get chunks from sections 
    chunk_list, chunk_to_section =  extract_chunks_from_sections(sections)
    # Connect images to chunks
    chunk_to_image_path = connect_images_to_chunks(json_data, chunk_list, line_to_section, chunk_to_section)
    # Add full path to the images
    chunk_to_image_path =  set({(i[0], i[1], add_full_path(i[2], output_folder, paper_name)) for i in chunk_to_image_path})
    # Save transformed data to json file
    results = {
        'paper_name': paper_name,
        'chunks': chunk_list,
        'chunk_to_image': chunk_to_image_path
    }
    pickle.dump(results, open(output_folder + '/extracts/' + paper_name + '.pkl', 'wb'))
    

