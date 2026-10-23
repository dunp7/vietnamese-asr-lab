# WeNet

This recipe is developed and validated against a specific WeNet commit:
- **Commit:** `015469960db73eb19c49a48c020d5ac9db69bae5`
- **Python Runtime:** PyTorch 2.4.0 (CUDA 12.1), DeepSpeed, Whisper

```dockerfile
ARG WENET_COMMIT=015469960db73eb19c49a48c020d5ac9db69bae5
RUN git clone [https://github.com/wenet-e2e/wenet.git](https://github.com/wenet-e2e/wenet.git) . && git checkout ${WENET_COMMIT}
```

Clone WeNet, check out the commit above, set up a Python virtual environment, and install the dependencies from WeNet's `requirements.txt`. Select PyTorch and torchaudio builds compatible with your system's CUDA version. Additional dependencies like `pandas` and `matplotlib` are required for log parsing and analysis scripts.

Step-by-step instructions for creating a custom recipe are documented in `instruction.md` under the Recipe section. The C++ runtime implementation has its own dedicated reference notes.

## Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install WeNet dependencies and analysis tools
pip install --upgrade pip
pip install -r requirements.txt
pip install pandas matplotlib

# Install PyTorch and Torchaudio compatible with CUDA 12.1
pip install torch==2.4.0+cu121 torchaudio==2.4.0+cu121 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) --force-reinstall

# Install advanced training and alignment libraries
pip install "openai-whisper==20250625" "deepspeed>=0.14.0"
```