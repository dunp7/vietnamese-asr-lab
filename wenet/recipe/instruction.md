# Training Recipe

1. Clone [WeNet](https://github.com/wenet-e2e/wenet) and check out commit `015469960db73eb19c49a48c020d5ac9db69bae5`, which serves as the reference baseline.
2. Inside WeNet, create a dedicated directory named `train`, located at the same level as the French recipe (`examples/commonvoice/fr`).
3. Structure `train` following the layout of the [French recipe](https://github.com/wenet-e2e/wenet/tree/015469960db73eb19c49a48c020d5ac9db69bae5/examples/commonvoice/fr): include `README.md`, `conf`, `local`, `path.sh`, `run.sh`, and the two symbolic links `tools` and `wenet`. Place training configurations in `conf`, custom data preparation scripts in `local`, and use your Vietnamese dataset.

```text
train/
├── README.md
├── conf/                  # Training YAML configurations
├── local/                 # Custom Vietnamese data preparation scripts
├── path.sh                # Environment and PATH setup
├── run.sh                 # Pipeline entry point
├── tools -> ../../../tools/
└── wenet -> ../../../wenet/
```

Run all commands from inside the `train` directory after activating the Python virtual environment and preparing the `train`, `dev`, and `test` sets. `run.sh` controls the pipeline flow via `--stage` and `--stop_stage`:

* **Data Preparation:**
  ```bash
  bash run.sh --stage 1 --stop_stage 3 > logs/prepare_data.txt 2>&1
  ```
  > **Note:** After generating the dictionary/token list, verify the `<sos/eos>` and blank symbol IDs in your training config (`conf/*.yaml`) to ensure they match the newly generated dictionary indices.

* **Model Training:**
  Verify GPU availability, batch sizes, and data paths in the config before running:
  ```bash
  bash run.sh --stage 4 --stop_stage 4 > logs/train.txt 2>&1
  ```

* **Decoding & Evaluation:**
  * **Decoding:** Requires `DECODE_CHECKPOINT`, `TRAIN_CONFIG`, `TEST_DATA`, and optionally `GPU_ID`.
  * **Evaluation:** Requires `REFERENCE_TEXT` and `RESULT_DIR` to compute CER/WER.

* **Model Export:**
  * Point `MODEL_DIR` to your trained/averaged checkpoint directory to export TorchScript, ONNX, or LibTorch C++ artifacts.
