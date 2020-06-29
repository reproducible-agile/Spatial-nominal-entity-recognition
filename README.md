# Spatial Nominal Entity Recognition


## Overview


## Run the code

	python3 evaluate_model_snoer.py -i <input_dataset> -ft <fasttext_model> -fr_nouns <french_nouns_filepath> -alg <algorithm_name> -m <model_filepath> -n <ngram_size>

 * `<input_dataset>` : 
 * `<fasttext_model>` 
 * `<french_nouns_filepath>` 

## Example

Datasets are given in the `corpus` directory and models in the `models` directory. Run the following command to evaluate the GRU model trained with 5 grams :

    python3 evaluate_model_snoer.py -i "./corpus/corpus_validation_mix.csv" -ft "./corpus/cc.fr.300.bin" -fr_nouns "./corpus/listedesnomfrancais.txt" -alg "GRU" -m "./models/GRU_5grams.h5" -n 5
    

## Acknowledgement

This work is supported and financed by French National Research Agency (ANR) under the CHOUCAS project (ANR-16-CE23-0018). 

The [CHOUCAS project](http://choucas.ign.fr) is a French interdisciplinary research project aiming to respond to a need expressed by the high mountain gendarmerie platoon to help localising victims in mountain area.
