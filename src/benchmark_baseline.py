import torch
import time
import json
import os
import sys
from pathlib import Path

try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError:
    print("Error: FastVLM not installed. Please run:")
    print("  cd ml-fastvlm && pip install -e .")
    sys.exit(1)

from PIL import Image
import numpy as np

# M2 optimizations
torch.backends.mps.force_32_bit_mps = True  # Use 32-bit on M2 for better performance
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_image(image_file):
    """Load and preprocess an image."""
    image = Image.open(image_file).convert('RGB')
    return image

def benchmark_model(model_path, image_path, prompts, num_runs=5):
    """Benchmark the model performance."""
    
    # Disable gradients for inference
    disable_torch_init()
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load model
    start_load = time.time()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=device
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Load test image
    image = load_image(image_path)
    
    # Benchmark results
    results = {
        "model_path": model_path,
        "device": device,
        "load_time": load_time,
        "total_params": total_params,
        "benchmarks": []
    }
    
    # Test each prompt
    for prompt in prompts:
        print(f"\nTesting prompt: {prompt}")
        
        # Prepare conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)
        
        # Process image
        with torch.no_grad():
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float16)
        
        # Warmup run
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True
            )
        
        # Benchmark runs
        times = []
        token_counts = []
        
        for i in range(num_runs):
            start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True
                )
            end = time.time()
            
            # Calculate tokens generated
            generated_tokens = output_ids[0][input_ids.shape[1]:]
            num_tokens = len(generated_tokens)
            
            times.append(end - start)
            token_counts.append(num_tokens)
            
            # Decode output
            output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if i == 0:  # Print first output only
                print(f"Output: {output[:100]}...")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_tokens = np.mean(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        benchmark_result = {
            "prompt": prompt,
            "avg_time": avg_time,
            "std_time": std_time,
            "avg_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "runs": times
        }
        results["benchmarks"].append(benchmark_result)
        
        print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Tokens/second: {tokens_per_second:.2f}")
    
    return results

def create_test_image():
    """Create a simple test image if none exists."""
    test_image_path = Path("test_image.jpg")
    if not test_image_path.exists():
        # Create a simple test image with shapes and text
        from PIL import ImageDraw, ImageFont
        
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes
        draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=3)
        draw.ellipse([250, 50, 400, 200], fill='blue', outline='black', width=3)
        draw.polygon([(500, 50), (650, 50), (575, 200)], fill='green', outline='black', width=3)
        
        # Add text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except:
            font = ImageFont.load_default()
        draw.text((100, 300), "FastVLM Test", font=font, fill='black')
        draw.text((100, 400), "Shapes: Square, Circle, Triangle", font=font, fill='gray')
        
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    return test_image_path

def main():
    # Model path
    model_path = "ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"
    
    # Create or use test image
    image_path = create_test_image()
    
    # Test prompts
    prompts = [
        "Describe this image in detail.",
        "What shapes do you see in this image?",
        "What text is written in the image?",
        "Count the number of shapes in the image.",
        "What colors are the shapes?"
    ]
    
    # Run benchmark
    print("Starting FastVLM baseline benchmark on M2...")
    print("=" * 60)
    
    results = benchmark_model(model_path, image_path, prompts, num_runs=5)
    
    # Save results
    output_path = "benchmark_results_baseline.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Benchmark complete! Results saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"Model load time: {results['load_time']:.2f}s")
    print(f"Total parameters: {results['total_params'] / 1e6:.2f}M")
    print(f"Average tokens/second across all prompts: {np.mean([b['tokens_per_second'] for b in results['benchmarks']]):.2f}")

if __name__ == "__main__":
    main()