#!/bin/bash
ModelPath="/config/models/"
ModePath="/config/modes/cora/train/"
FileExt=".yaml"

for i in "ppnp" "p_ppnp" "mixed_ppnp" "gcn" "p_gcn" "mixed_gcn" "gpn_16" "gat" "p_gat" "p_gat_proj" "p_gat_att" "mixed_gat" "mixed_gat_proj" "mixed_gat_att"
do
    for j in "classification" "ood_loc"
    do
        python run.py --config "$ModelPath$i$FileExt" "$ModePath$j$FileExt"
    done
done