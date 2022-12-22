#!/bin/bash
ModelPath="/config/models/"
ModePath="/config/modes/sbm/train/"
ActiveLearningUncertaintyPath="/config/active_learning/"
FileExt=".yaml"

for i in "p_ppnp"
do
    for j in "active_learning" "active_learning_dynamic"
    do
        for k in "random" "uncertainty_isolated_aleatoric" "uncertainty_isolated_epistemic" "uncertainty_propagated_aleatoric" "uncertainty_propagated_epistemic" "l2_distance" "l2_distance_centroid"
        do
            python run.py --config "$ModelPath$i$FileExt" "$ModePath$j$FileExt" "$ActiveLearningUncertaintyPath$k$FileExt"
        done
    done
done