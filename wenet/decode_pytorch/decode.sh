# Run from a prepared WeNet recipe with the Python environment active.
. ./path.sh || exit 1
gpu_id=${GPU_ID:-0}
echo "GPU: ${gpu_id}"


# ==========================================
# Model configuration
# ==========================================

: "${DECODE_CHECKPOINT:?Set DECODE_CHECKPOINT before decoding}"
: "${TRAIN_CONFIG:?Set TRAIN_CONFIG before decoding}"
: "${TEST_DATA:?Set TEST_DATA before decoding}"
decode_checkpoint=$DECODE_CHECKPOINT
train_config=$TRAIN_CONFIG

echo "model checkpoint:  $decode_checkpoint"


# ==========================================
# Test Data
# ==========================================
test_data=$TEST_DATA

# # Create sample test
# test_data=sample/wenet/vi/sample.list
# cat > "$sample_data" <<'EOF'
# {"key": "vi_000000001", "wav": "sample/common_voice_vi_25092495.wav", "txt": "ánh mắt của trinh nhìn quân đầy gợi tình"}
# {"key": "vi_000000002", "wav": "sample/common_voice_vi_25158101.wav", "txt": "chưa kể lời của sâm trai"}
# {"key": "vi_000000003", "wav": "sample/common_voice_vi_25177315.wav", "txt": "khi mẹ em ốm"}

# EOF
# ==========================================
# Decoding parameters
# ==========================================
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decoding_chunk_size=
ctc_weight=0.3
reverse_weight=0.5

# Result directory
base_result_dir=${RESULT_DIR:-eval/result}
result_dir="$base_result_dir"
i=1
while [ -e "$result_dir" ]; do
    result_dir="${base_result_dir}_${i}"
    ((i++))
done

mkdir -p "$result_dir"
echo "Result directory: $result_dir"

python wenet/bin/recognize.py --gpu "$gpu_id" \
    --modes $decode_modes \
    --config $train_config \
    --data_type "raw" \
    --test_data $test_data \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 32 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $result_dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
