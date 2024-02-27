#!bash

# Downloads a TCGA slide
path=./data_test/TCGA-Z7-A8R6-01Z-00-DX1.CE4ED818-D762-4324-9DEA-2ACB38B9B0B9.svs
if [ ! -f "$path" ]; then
    gdown --id 15CCDOypH1FHydiz6sluXWxTBhugxrFH2 -O $path
fi

# Encode the slide - simplest model : mlp + moco -

python main.py --input "$path" --output "./data_encoded" --gigassl_type "mlp" --tile_encoder_type "moco" --N_ensemble 50 --store_intermediate "./.tmp"
