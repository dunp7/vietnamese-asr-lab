#!/bin/bash
set -e

# ==========================================
# Environment
# ==========================================
. ./path.sh || exit 1
: "${MODEL_DIR:?Set MODEL_DIR before export}"

# ==========================================
# Model configuration
# ==========================================
dir=$MODEL_DIR
# Options: specific | average
# export_mode="specific" 
export_mode="average"

# Specific checkpoint
specific_checkpoint=${SPECIFIC_CHECKPOINT:-}

# Average checkpoint
average_num=5
average_val_best=true

# ==========================================
# Select checkpoint
# ==========================================
if [ "$export_mode" == "specific" ]; then
    : "${specific_checkpoint:?Set SPECIFIC_CHECKPOINT for specific export}"
    decode_checkpoint="$dir/$specific_checkpoint"

elif [ "$export_mode" == "average" ]; then
    decode_checkpoint="$dir/avg_${average_num}.pt"
    python wenet/bin/average_model.py \
        --dst_model "$decode_checkpoint" \
        --src_path "$dir" \
        --num "$average_num" \
        $( [ "$average_val_best" == true ] && echo "--val_best" )
else
    echo "Invalid export_mode: $export_mode"
    echo "Choose: specific or average"
    exit 1
fi
# ==========================================
# Export JIT
# ==========================================
if [ ! -f "$decode_checkpoint" ]; then
    echo "Checkpoint not found: $decode_checkpoint"
    exit 1
fi

output_file="$dir/final.zip"

echo "Export mode: $export_mode"
echo "Checkpoint: $decode_checkpoint"
echo "Output: $output_file"

python wenet/bin/export_jit.py \
    --config "$dir/train.yaml" \
    --checkpoint "$decode_checkpoint" \
    --output_file "$output_file"

echo "Export completed: $output_file"
