import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_model_and_tokenizer(model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
    # Configuration for 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 3090 supports bfloat16
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for LoRA (Only training small adapter weights)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

if __name__ == "__main__":
    # Test loading
    m, t = setup_model_and_tokenizer()
    print("Model loaded successfully!")
    prompt = "write a program to print 'hello world'."
    inputs = t(prompt, return_tensors="pt").to("cuda")

    # We set max_new_tokens to keep it short and do_sample=True for variety
    outputs = m.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=t.eos_token_id
    )

    response_text = t.decode(outputs[0], skip_special_tokens=True)

    print("--- PROMPT ---")
    print(prompt)
    print("\n--- RESPONSE ---")
    print(response_text)