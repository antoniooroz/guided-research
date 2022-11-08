#!/bin/bash
ModelPath="/config/models/"
ModePath="/config/modes/cora/load/"
CombinationPath="/config/experiments/combined_uncertainty/"
FileExt=".yaml"

for j in "ood_loc" "ood_perturb_normal" "ood_perturb_ber" "classification"
do
    for i in "ppnp" "p_ppnp" "mixed_ppnp" "gcn" "p_gcn" "mixed_gcn" "gpn_16" "gat" "p_gat" "p_gat_proj" "p_gat_att" "mixed_gat" "mixed_gat_proj" "mixed_gat_att" "mcd_ppnp" "mcd_gcn" "mcd_gat" "de_gcn"
    do
        python run.py --config "$ModelPath$i$FileExt" "$ModePath$j$FileExt"
    done
done