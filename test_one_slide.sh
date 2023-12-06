#!bash

# Downloads a TCGA slide
path=./data_test/TCGA-Z7-A8R6-01Z-00-DX1.CE4ED818-D762-4324-9DEA-2ACB38B9B0B9.svs
if [ ! -f "$path" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15CCDOypH1FHydiz6sluXWxTBhugxrFH2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15CCDOypH1FHydiz6sluXWxTBhugxrFH2" -O $path && rm -rf /tmp/cookies.txtOD
fi

# Encode the slide - simplest model : mlp + moco -

python main.py --input "$path" --output "./data_encoded" --gigassl_type "mlp" --tile_encoder_type "moco" --N_ensemble 5 --store_intermediate "./.tmp"
