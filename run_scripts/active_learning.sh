#!/bin/bash
ModelPath="/config/models/"
ModePath="/config/modes/sbm/train/"
ActiveLearningUncertaintyPath="/config/active_learning/"
FileExt=".yaml"

for i in "ppnp" "p_ppnp" "gcn" "p_gcn" "gat" "p_gat"
do
    for j in "active_learning"
    do
        for k in "random" "uncertainty_isolated_aleatoric" "uncertainty_isolated_epistemic" "uncertainty_propagated_aleatoric" "uncertainty_propagated_epistemic"
        do
            python run.py --config "$ModelPath$i$FileExt" "$ModePath$j$FileExt" "$ActiveLearningUncertaintyPath$k$FileExt"
        done
    done
done