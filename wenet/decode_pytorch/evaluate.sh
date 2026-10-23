# ==========================================
# Environment
# ==========================================
# Run from a prepared WeNet recipe with the Python environment active.
. ./path.sh || exit 1
: "${REFERENCE_TEXT:?Set REFERENCE_TEXT before evaluation}"
: "${RESULT_DIR:?Set RESULT_DIR before evaluation}"
result_dir=$RESULT_DIR

decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"



for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      "$REFERENCE_TEXT" "$result_dir/$mode/text" > "$result_dir/$mode/wer"
  done

for mode in ${decode_modes}; do
    python tools/compute-cer.py --char=1 --v=1 \
      "$REFERENCE_TEXT" "$result_dir/$mode/text" > "$result_dir/$mode/cer"
  done

echo "DONE"
