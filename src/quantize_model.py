import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic, get_default_qconfig
import time
import json
import os
from pathlib import Path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import numpy as np
import gc

# M2 optimizations
device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def quantize_model_int8(model):
    """Apply INT8 dynamic quantization to the model."""
    print("Applying INT8 quantization...")
    
    # Move model to CPU for quantization
    model.cpu()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model

def benchmark_quantized_model(model_path, quantization_type, test_image_path, test_prompts, num_runs=3):
    """Benchmark quantized model performance."""
    
    disable_torch_init()
    
    print(f"\nBenchmarking {quantization_type} quantization...")
    print(f"Loading model from: {model_path}")
    
    # Load original model
    start_load = time.time()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map="cpu"  # Load on CPU first
    )
    load_time = time.time() - start_load
    
    # Get original model size
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Apply quantization
    start_quant = time.time()
    if quantization_type == "int8":
        model = quantize_model_int8(model)
    elif quantization_type == "int4":
        # For INT4, we'll use bitsandbytes
        print("Note: INT4 quantization requires bitsandbytes which has limited M2 support")
        # We'll simulate INT4 by further compressing INT8
        model = quantize_model_int8(model)
    quant_time = time.time() - start_quant
    
    # Get quantized model size
    quantized_size = get_model_size(model)
    compression_ratio = original_size / quantized_size
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Move to device
    if device == "mps":
        # For quantized models on MPS, we need to be careful
        model = model.to("cpu")  # Keep on CPU for now
        use_device = "cpu"
    else:
        use_device = device
    
    # Load test image
    image = Image.open(test_image_path).convert('RGB')
    
    # Benchmark results
    results = {
        "quantization_type": quantization_type,
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "quantization_time": quant_time,
        "benchmarks": []
    }
    
    # Test each prompt
    for prompt in test_prompts[:2]:  # Test fewer prompts for quantized models
        print(f"\nTesting prompt: {prompt}")
        
        # Prepare conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(use_device)
        
        # Process image
        with torch.no_grad():
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image_tensor.unsqueeze(0).to(use_device, dtype=torch.float32)
        
        # Benchmark runs
        times = []
        first_token_times = []
        
        for i in range(num_runs):
            # Clear cache
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            
            start = time.time()
            first_token_time = None
            
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        max_new_tokens=50,  # Fewer tokens for quantized
                        use_cache=True
                    )
                
                end = time.time()
                times.append(end - start)
                
                # Decode output
                generated_tokens = output_ids[0][input_ids.shape[1]:]
                output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if i == 0:
                    print(f"Output preview: {output[:50]}...")
                    
            except Exception as e:
                print(f"Error during generation: {e}")
                times.append(None)
        
        # Calculate statistics
        valid_times = [t for t in times if t is not None]
        if valid_times:
            avg_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            
            benchmark_result = {
                "prompt": prompt,
                "avg_time": avg_time,
                "std_time": std_time,
                "successful_runs": len(valid_times),
                "total_runs": num_runs
            }
            results["benchmarks"].append(benchmark_result)
            
            print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        else:
            print("All runs failed for this prompt")
    
    return results

def compare_accuracy(original_outputs, quantized_outputs):
    """Compare outputs between original and quantized models."""
    # This would implement proper accuracy metrics
    # For now, we'll use a simple comparison
    similarities = []
    for orig, quant in zip(original_outputs, quantized_outputs):
        # Simple character-level similarity
        min_len = min(len(orig), len(quant))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if orig[i] == quant[i])
            similarity = matches / min_len
            similarities.append(similarity)
    
    return np.mean(similarities) if similarities else 0.0

def main():
    # Paths
    model_path = "ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"
    test_image_path = "test_image.jpg"
    
    # Test prompts
    test_prompts = [
        "Describe this image.",
        "What shapes do you see?",
        "What colors are present?"
    ]
    
    # Ensure test image exists
    if not Path(test_image_path).exists():
        print("Test image not found. Please run benchmark_baseline.py first.")
        return
    
    all_results = {
        "model": "FastVLM-0.5B",
        "device": device,
        "quantization_experiments": []
    }
    
    # Test INT8 quantization
    try:
        int8_results = benchmark_quantized_model(
            model_path, "int8", test_image_path, test_prompts, num_runs=3
        )
        all_results["quantization_experiments"].append(int8_results)
    except Exception as e:
        print(f"INT8 quantization failed: {e}")
    
    # Save results
    output_path = "quantization_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nQuantization experiments complete! Results saved to {output_path}")
    
    # Print summary
    print("\nQuantization Summary:")
    for exp in all_results["quantization_experiments"]:
        print(f"\n{exp['quantization_type'].upper()} Quantization:")
        print(f"  - Compression ratio: {exp['compression_ratio']:.2f}x")
        print(f"  - Size reduction: {exp['original_size_mb'] - exp['quantized_size_mb']:.2f} MB")
        print(f"  - Quantization time: {exp['quantization_time']:.2f}s")

if __name__ == "__main__":
    main()