# Homestuck-6k

This repo contains code for generating a 6k text-image pairs dataset from the webcomic Homestuck, leveraging image descriptions from the Accessible Homestuck Project. This repo also contains code for finetuning CLIP, doing image retrieval upon the dataset, and doing zero shot classification using the finetuned CLIP model.

## Generating Data

```
cd ./data
python3 get_webcomic.py
python3 to_json.py
python3 fuzzy_matching.py
python3 prune_dataset.py
``` 

## CLIP

To finetune CLIP, run the included bash script

```
sh run.sh
```

image retrieval and zero shot classification demos can be done via the included scripts retrieval.py and zeroshot.py, but require direct modification
