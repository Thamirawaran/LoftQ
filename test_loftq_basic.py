"""
Basic LoftQ functionality test (CPU mode)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("Testing basic LoftQ setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test with a small model (using CPU-friendly approach)
MODEL_NAME = "gpt2"  # Small model for quick testing

print(f"\n1. Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
)

print(f"✓ Model loaded successfully")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test LoRA configuration
print("\n2. Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # GPT-2 specific modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

print("\n3. Testing inference...")
test_text = "Hello, this is a test"
inputs = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_length=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"✓ Generated text: {generated_text}")

print("\n✅ Basic LoftQ setup test passed!")
print("\nNote: For full LoftQ with quantization, you need CUDA GPU support.")
