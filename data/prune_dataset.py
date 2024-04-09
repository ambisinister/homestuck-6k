import os

def prune_dataset(directory):
    files = os.listdir(directory)
    image_files = [f for f in files if f.endswith('.jpg')]
    text_files = [f for f in files if f.endswith('.txt')]

    text_panels = set(f.split('_')[0] for f in text_files)

    for image in image_files:
        panel_number = image.split('_')[0]
        if panel_number not in text_panels:
            os.remove(os.path.join(directory, image))

    for panel in text_panels:
        relevant_images = sorted([f for f in image_files if f.startswith(panel)])
        relevant_texts = [f for f in text_files if f.startswith(panel)]

        while len(relevant_images) > len(relevant_texts):
            image_sizes = [(img, os.path.getsize(os.path.join(directory, img))) for img in relevant_images]
            smallest_image = min(image_sizes, key=lambda x: x[1])[0]
            os.remove(os.path.join(directory, smallest_image))
            relevant_images.remove(smallest_image)

        for image, text in zip(relevant_images, relevant_texts):
            new_name = text.replace('.txt', '.jpg')
            os.rename(os.path.join(directory, image), os.path.join(directory, new_name))

directory = './screens/clip'
prune_dataset(directory)


