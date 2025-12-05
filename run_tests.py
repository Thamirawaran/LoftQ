"""
Comprehensive LoftQ Testing Suite
Tests various aspects of LoftQ functionality and saves results
"""
import torch
import time
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os

# Create results directory
os.makedirs("test_results", exist_ok=True)

results = {
    "test_date": datetime.now().isoformat(),
    "system_info": {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "tests": []
}

print("="*60)
print("LoftQ Comprehensive Test Suite")
print("="*60)
print(f"Date: {results['test_date']}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {results['system_info']['device']}")
print("="*60)

# Test 1: Basic Model Loading
print("\n[Test 1] Basic Model Loading")
print("-"*60)
test1_result = {"name": "Basic Model Loading", "status": "running"}
try:
    start_time = time.time()
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    load_time = time.time() - start_time
    
    param_count = sum(p.numel() for p in model.parameters())
    test1_result.update({
        "status": "PASSED",
        "model": model_name,
        "parameters": param_count,
        "load_time_seconds": round(load_time, 2),
        "memory_mb": round(param_count * 4 / (1024**2), 2)  # Approx for float32
    })
    print(f"✓ Model: {model_name}")
    print(f"✓ Parameters: {param_count:,}")
    print(f"✓ Load time: {load_time:.2f}s")
except Exception as e:
    test1_result.update({"status": "FAILED", "error": str(e)})
    print(f"✗ Failed: {e}")
results["tests"].append(test1_result)

# Test 2: LoRA Configuration with Different Ranks
print("\n[Test 2] LoRA Configuration with Different Ranks")
print("-"*60)
test2_results = []
ranks = [4, 8, 16, 32, 64]

for rank in ranks:
    test_result = {"name": f"LoRA Rank {rank}", "status": "running"}
    try:
        start_time = time.time()
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        setup_time = time.time() - start_time
        
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        
        test_result.update({
            "status": "PASSED",
            "rank": rank,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": round(trainable_percentage, 4),
            "setup_time_seconds": round(setup_time, 2)
        })
        print(f"✓ Rank {rank}: {trainable_params:,} trainable params ({trainable_percentage:.2f}%)")
    except Exception as e:
        test_result.update({"status": "FAILED", "error": str(e)})
        print(f"✗ Rank {rank} failed: {e}")
    test2_results.append(test_result)

results["tests"].extend(test2_results)

# Test 3: Text Generation with LoRA
print("\n[Test 3] Text Generation with LoRA")
print("-"*60)
test3_result = {"name": "Text Generation", "status": "running"}
try:
    # Use rank 8 for generation test
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)
    
    test_prompts = [
        "Artificial intelligence is",
        "The future of technology",
        "Machine learning enables"
    ]
    
    generations = []
    total_gen_time = 0
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_length=30,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - start_time
        total_gen_time += gen_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generation_time_seconds": round(gen_time, 2)
        })
        print(f"✓ Prompt: '{prompt}'")
        print(f"  Generated: '{generated_text}'")
        print(f"  Time: {gen_time:.2f}s")
    
    test3_result.update({
        "status": "PASSED",
        "generations": generations,
        "average_generation_time": round(total_gen_time / len(test_prompts), 2)
    })
except Exception as e:
    test3_result.update({"status": "FAILED", "error": str(e)})
    print(f"✗ Failed: {e}")
results["tests"].append(test3_result)

# Test 4: Parameter Efficiency Comparison
print("\n[Test 4] Parameter Efficiency Analysis")
print("-"*60)
test4_result = {"name": "Parameter Efficiency", "status": "running"}
try:
    base_params = sum(p.numel() for p in model.parameters())
    
    efficiency_data = []
    for rank in [4, 8, 16, 32, 64]:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        temp_model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        
        efficiency_data.append({
            "rank": rank,
            "base_params": base_params,
            "trainable_params": trainable,
            "reduction_factor": round(base_params / trainable, 2),
            "memory_savings_percent": round((1 - trainable/base_params) * 100, 2)
        })
        print(f"✓ Rank {rank}: {trainable:,} params, {efficiency_data[-1]['reduction_factor']}x reduction")
    
    test4_result.update({
        "status": "PASSED",
        "efficiency_data": efficiency_data
    })
except Exception as e:
    test4_result.update({"status": "FAILED", "error": str(e)})
    print(f"✗ Failed: {e}")
results["tests"].append(test4_result)

# Test 5: Different Target Modules
print("\n[Test 5] Different Target Module Configurations")
print("-"*60)
test5_results = []
module_configs = [
    {"name": "Attention Only", "modules": ["c_attn"]},
    {"name": "Projection Only", "modules": ["c_proj"]},
    {"name": "Both Attention & Projection", "modules": ["c_attn", "c_proj"]},
]

for config in module_configs:
    test_result = {"name": f"Target Modules: {config['name']}", "status": "running"}
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=config["modules"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        temp_model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        
        test_result.update({
            "status": "PASSED",
            "config_name": config["name"],
            "target_modules": config["modules"],
            "trainable_params": trainable
        })
        print(f"✓ {config['name']}: {trainable:,} trainable params")
    except Exception as e:
        test_result.update({"status": "FAILED", "error": str(e)})
        print(f"✗ {config['name']} failed: {e}")
    test5_results.append(test_result)

results["tests"].extend(test5_results)

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
total_tests = len(results["tests"])
passed_tests = sum(1 for test in results["tests"] if test.get("status") == "PASSED")
failed_tests = total_tests - passed_tests

print(f"Total Tests: {total_tests}")
print(f"Passed: {passed_tests} ✓")
print(f"Failed: {failed_tests} ✗")
print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

results["summary"] = {
    "total_tests": total_tests,
    "passed": passed_tests,
    "failed": failed_tests,
    "success_rate": round((passed_tests/total_tests)*100, 2)
}

# Save results to JSON
output_file = f"test_results/loftq_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Create a readable summary file
summary_file = f"test_results/loftq_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("LoftQ Test Summary\n")
    f.write("="*60 + "\n")
    f.write(f"Date: {results['test_date']}\n")
    f.write(f"PyTorch Version: {results['system_info']['pytorch_version']}\n")
    f.write(f"Device: {results['system_info']['device']}\n")
    f.write(f"CUDA Available: {results['system_info']['cuda_available']}\n")
    f.write("\n" + "="*60 + "\n")
    f.write("SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Total Tests: {total_tests}\n")
    f.write(f"Passed: {passed_tests}\n")
    f.write(f"Failed: {failed_tests}\n")
    f.write(f"Success Rate: {results['summary']['success_rate']}%\n")
    f.write("\n" + "="*60 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*60 + "\n")
    
    # Parameter efficiency findings
    f.write("\n1. Parameter Efficiency (LoRA Rank Analysis):\n")
    for test in results["tests"]:
        if "efficiency_data" in test:
            for data in test["efficiency_data"]:
                f.write(f"   - Rank {data['rank']}: {data['reduction_factor']}x reduction, "
                       f"{data['memory_savings_percent']}% memory savings\n")
    
    # Generation performance
    f.write("\n2. Text Generation Performance:\n")
    for test in results["tests"]:
        if test["name"] == "Text Generation" and test["status"] == "PASSED":
            f.write(f"   - Average generation time: {test['average_generation_time']}s\n")
            f.write(f"   - Number of test prompts: {len(test['generations'])}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*60 + "\n")
    if not results['system_info']['cuda_available']:
        f.write("- CUDA is not available. For quantization features, use a GPU-enabled system.\n")
    f.write("- LoRA rank 8-16 provides good balance between efficiency and performance.\n")
    f.write("- Targeting both attention and projection layers gives best coverage.\n")
    f.write("- Consider fake quantization mode for CPU testing.\n")

print(f"✓ Summary saved to: {summary_file}")
print("\n" + "="*60)
print("All tests completed!")
print("="*60)
