import re
from fuzzywuzzy import fuzz
import os
from PIL import Image

# Pass 1: High Confidence Matching
def map_image_descriptions_to_panels_pass1(doc_a, doc_b):
    panels = re.split(r'Panel \d+', doc_a)[1:]
    panels = [panel.strip() for panel in panels]

    image_pattern = re.compile(r'\[Image description:(.*?)\]', re.DOTALL)
    image_descriptions = re.findall(image_pattern, doc_b)

    mapping = {}
    undetermined = []

    for index, image_desc in enumerate(image_descriptions):
        print(index)
        next_img_index = doc_b.index(image_desc)
        next_text = doc_b[next_img_index + len(image_desc):].split('[Image description:', 1)[0].strip()

        best_score = 0
        best_panel_index = -1

        for i, panel in enumerate(panels[index:index+300]):
            panel_text = panel.lower().replace('\n', ' ').strip()
            score = fuzz.ratio(next_text, panel_text)
            if score > best_score:
                best_score = score
                best_panel_index = index+i

        if best_score >= 80:
            mapping[index] = best_panel_index
        else:
            undetermined.append(index)

    print(len(mapping.keys()))

    return mapping

# Pass 2: Simple Many-to-many Matching
def map_image_descriptions_to_panels_pass2(doc_a, doc_b, mapping):
    panels = re.split(r'Panel \d+', doc_a)[1:]
    panels = [panel.strip() for panel in panels]

    image_pattern = re.compile(r'\[Image description:(.*?)\]', re.DOTALL)
    image_descriptions = re.findall(image_pattern, doc_b)

    sorted_match_indices = sorted(mapping.keys())
    sorted_matched_panels = [mapping[i] for i in sorted_match_indices]

    for i in range(len(sorted_match_indices) - 1):
        current_index = sorted_match_indices[i]
        next_index = sorted_match_indices[i + 1]
        current_panel = sorted_matched_panels[i]
        next_panel = sorted_matched_panels[i + 1]

        gap_in_descriptions = next_index - current_index - 1
        gap_in_panels = next_panel - current_panel - 1

        if gap_in_descriptions == gap_in_panels:
            for gap_index in range(gap_in_descriptions):
                unmatched_description_index = current_index + gap_index + 1
                mapping[unmatched_description_index] = current_panel + gap_index + 1

    print(len(mapping.keys()))

    return mapping


def save_image_descriptions(final_mapping, doc_b):
    image_pattern = re.compile(r'\[Image description:(.*?)\]', re.DOTALL)
    image_descriptions = re.findall(image_pattern, doc_b)

    if not os.path.exists('./screens'):
        os.makedirs('./screens')
    if not os.path.exists('./screens/clip'):
        os.makedirs('./screens/clip')

    for idx, panel_number in sorted(final_mapping.items()):
        image_desc = image_descriptions[idx].strip()
        panel_str = str(panel_number+1).zfill(4)
        file_path = f'./screens/clip/{panel_str}_1.txt'

        counter = 1
        while os.path.exists(file_path):
            file_path = f'./screens/clip/{panel_str}_{counter}.txt'
            counter += 1

        with open(file_path, 'w') as file:
            file.write(image_desc)

def convert_and_save_images(source_folder, destination_folder):
    for file_name in os.listdir(source_folder):
        if re.match(r'\d+_\d+\.\w+', file_name):

            base_name, extension = os.path.splitext(file_name)
            number1, number2 = base_name.split('_')


            padded_number1 = number1.zfill(4)
            new_file_name = f"{padded_number1}_{number2}.jpg"


            try:
                with Image.open(os.path.join(source_folder, file_name)) as img:
                    img.convert('RGB').save(os.path.join(destination_folder, new_file_name), 'JPEG')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


with open('./all_comictext.txt', 'r') as f:
    doc_a = f.read()

with open('./transcription.txt', 'r') as f:
    doc_b = f.read()
                
mapping = map_image_descriptions_to_panels_pass1(doc_a, doc_b)
mapping = map_image_descriptions_to_panels_pass2(doc_a, doc_b, mapping)
save_image_descriptions(mapping, doc_b)
convert_and_save_images('./screens/img', './screens/clip')
