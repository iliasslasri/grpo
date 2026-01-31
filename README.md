# grpo


### Setup
Create a conda env and set it up with:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install "transformers==4.40.0" "peft==0.10.0" "accelerate==0.29.0" "bitsandbytes==0.41.0"
```

and (to run src/ from .)
```bash
export PYTHONPATH=$PYTHONPATH:.
```