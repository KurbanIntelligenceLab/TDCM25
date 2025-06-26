import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm

from benchmarks.explainability.explainability_dataloader import TiO2CaptionDataModule
from config import BaseTDCD25Config

class MultiModelCaptioningPipeline:
    """
    A captioning pipeline that supports multiple LLM models.
    It can generate captions for nanoparticle images based on both image and XYZ data,
    and compute text and numerical similarity metrics.
    """
    def __init__(self, api_keys, model_name="gpt-4-vision-preview",
                 use_openrouter=False, site_url=None, site_name=None):
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        self.api_keys = api_keys

        if self.use_openrouter:
            self.headers = {
                "Authorization": f"Bearer {api_keys['openrouter']}",
                "Content-Type": "application/json"
            }
        else:
            self.client = OpenAI(api_key=api_keys['openai'])

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    # -------------------------------
    # Private helper methods
    # -------------------------------
    def _encode_image(self, image_path):
        """Convert an image file to a base64 string."""
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def _format_xyz_data(self, xyz_path):
        """Read and format XYZ file content (skipping header lines)."""
        try:
            with open(xyz_path, 'r') as f:
                xyz_content = f.read().split('\n')[2:]
            return xyz_content
        except Exception as e:
            print(f"Error reading XYZ file {xyz_path}: {e}")
            raise

    def _build_system_message(self):
        """Return the system message for the captioning prompt."""
        return (
            "You are a materials science expert specializing in analyzing TiO2 (titanium dioxide) nanoparticles. "
            "Your task is to generate precise captions describing the structural properties of nanoparticles based on both visual "
            "and atomic coordinate data. You should predict both the exact temperature based on the given range and the crystal phase "
            "(anatase, brookite, or rutile), and determine the precise rotation applied to the structure if it is not the original configuration."
        )

    def _build_user_message(self, image_b64, xyz_content, is_original):
        """Return the user message for the captioning prompt."""
        temp_lower, temp_upper = 0, 1000  # temperature range
        configuration_text = "the original" if is_original else "a rotated"
        user_text = f"""
Analyze this TiO2 nanoparticle structure. The temperature is between {temp_lower}K and {temp_upper}K. This is {configuration_text} configuration.

Here is the XYZ structural data:
{xyz_content}
Based on the structural data and image, perform the following tasks:

1. **Predict the crystal phase**: (options: anatase, brookite, rutile)
2. **Predict the exact temperature** within the given range.
3. **Determine the precise rotation angles** if this is a rotated configuration.

Then, generate a caption in the following exact format (replace the placeholders with your predictions):
    
"This [predicted_phase] configuration at [predicted_temperature]K consists of [total_atoms] atoms, including [ti_atoms] titanium atoms and [o_atoms] oxygen atoms, resulting in a Ti:O ratio of approximately [ratio]:1. The nanoparticle spans about [x_dimension] Å in x, [y_dimension] Å in y, and [z_dimension] Å in z. [Original/Rotation Information]"

**Notes:**

- For rotated configurations, replace `[Original/Rotation Information]` with:
"Rotation applied: x=[x_angle]°, y=[y_angle]°, z=[z_angle]°."
- For original configurations, replace it with:
"This is the original configuration (no rotation)."
**Example Output:**
"This anatase configuration at 350K consists of 100 atoms, including 30 titanium atoms and 70 oxygen atoms, resulting in a Ti:O ratio of approximately 0.43:1. The nanoparticle spans about 5.0 Å in x, 3.0 Å in y, and 2.0 Å in z. This is the original configuration (no rotation)."
**Important:** Only output the caption as specified above without any additional text or explanations.
"""
        return user_text

    def create_messages(self, image_path, xyz_path, phase, temperature, is_original):
        """
        Create the messages payload for the API call.
        Combines system and user messages (with an image attachment).
        """
        image_b64 = self._encode_image(image_path)
        xyz_content = self._format_xyz_data(xyz_path)
        system_message = self._build_system_message()
        user_message = self._build_user_message(image_b64, xyz_content, is_original)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}}
            ]}
        ]
        return messages

    def _api_request(self, messages, max_tokens, temperature):
        """Make an API call using either OpenRouter or OpenAI with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.use_openrouter:
                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code != 200:
                        print(f"OpenRouter Error Response: {response.text}")
                        raise Exception(f"OpenRouter API Error: {response.json()}")
                    response_json = response.json()
                    if 'error' in response_json:
                        raise Exception(f"OpenRouter API Error: {response_json['error']}")
                    return response_json['choices'][0]['message']['content']
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
                continue

    # -------------------------------
    # Public pipeline methods
    # -------------------------------
    def generate_caption(self, image_path, xyz_path, phase, temperature, is_original):
        """
        Generate a caption for the provided image and XYZ data.
        Retries up to a set number of attempts.
        """
        messages = self.create_messages(image_path, xyz_path, phase, temperature, is_original)
        # Use different max_tokens if using OpenRouter versus OpenAI
        max_tokens = 6500 if self.use_openrouter else 1000
        return self._api_request(messages, max_tokens, temperature=0.2)

    def compute_metrics(self, reference, candidate):
        """
        Compute text similarity and numerical metrics from a reference caption
        and a generated candidate.
        """
        try:
            # Text similarity scores
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()
            bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smoothing)
            rouge = self.rouge_scorer.score(reference, candidate)

            def extract_numbers(text):
                import re
                numbers = {
                    'total_atoms': None,
                    'ti_atoms': None,
                    'o_atoms': None,
                    'temperature': None,
                    'dimensions': [],
                    'ratio': None,
                    'phase': None
                }
                try:
                    phase_match = re.search(r'This\s*(anatase|brookite|rutile)\s*configuration', text, re.IGNORECASE)
                    if phase_match:
                        numbers['phase'] = phase_match.group(1).lower()
                    temp_match = re.search(r'at\s*(\d+)\s*K', text, re.IGNORECASE)
                    if temp_match:
                        numbers['temperature'] = int(temp_match.group(1))
                    atoms_match = re.search(r'consists\s*of\s*(\d+)\s*atoms', text, re.IGNORECASE)
                    if atoms_match:
                        numbers['total_atoms'] = int(atoms_match.group(1))
                    ti_match = re.search(r'(\d+)\s*titanium\s*atoms', text, re.IGNORECASE)
                    if ti_match:
                        numbers['ti_atoms'] = int(ti_match.group(1))
                    o_match = re.search(r'(\d+)\s*oxygen\s*atoms', text, re.IGNORECASE)
                    if o_match:
                        numbers['o_atoms'] = int(o_match.group(1))
                    ratio_match = re.search(r'ratio\s*of\s*approximately\s*([\d.]+)\s*:\s*1', text, re.IGNORECASE)
                    if ratio_match:
                        numbers['ratio'] = float(ratio_match.group(1))
                    dims_match = re.search(
                        r'spans\s*about\s*([\d.]+)\s*Å\s*in\s*x\s*,\s*([\d.]+)\s*Å\s*in\s*y\s*,\s*and\s*([\d.]+)\s*Å\s*in\s*z',
                        text, re.IGNORECASE)
                    if dims_match:
                        numbers['dimensions'] = [
                            float(dims_match.group(1)),
                            float(dims_match.group(2)),
                            float(dims_match.group(3))
                        ]
                    else:
                        dims = re.findall(r'([\d.]+)\s*Å', text)
                        numbers['dimensions'] = [float(d) for d in dims[:3]] if len(dims) >= 3 else []
                except Exception as e:
                    print(f"Error in number extraction: {e}")
                    print(f"Text: {text}")
                return numbers

            ref_nums = extract_numbers(reference)
            gen_nums = extract_numbers(candidate)

            numerical_metrics = {
                'total_atoms_match': ref_nums['total_atoms'] == gen_nums['total_atoms'] if None not in [ref_nums['total_atoms'], gen_nums['total_atoms']] else False,
                'ti_atoms_match': ref_nums['ti_atoms'] == gen_nums['ti_atoms'] if None not in [ref_nums['ti_atoms'], gen_nums['ti_atoms']] else False,
                'o_atoms_match': ref_nums['o_atoms'] == gen_nums['o_atoms'] if None not in [ref_nums['o_atoms'], gen_nums['o_atoms']] else False,
                'phase_match': (ref_nums['phase'].lower() == gen_nums['phase'].lower()) if (ref_nums['phase'] and gen_nums['phase']) else False,
                'temperature_ranges': {
                    'exact': False, 'within_50K': False, 'within_100K': False, 'within_200K': False
                },
                'temperature_error': None,
                'dimension_ranges': {},
                'avg_dimension_percent_error': None,
                'ratio_match': False,
                'ratio_error': None
            }

            if ref_nums['temperature'] is not None and gen_nums['temperature'] is not None:
                temp_diff = abs(ref_nums['temperature'] - gen_nums['temperature'])
                numerical_metrics['temperature_error'] = temp_diff
                numerical_metrics['temperature_ranges']['exact'] = (temp_diff == 0)
                numerical_metrics['temperature_ranges']['within_50K'] = (temp_diff <= 50)
                numerical_metrics['temperature_ranges']['within_100K'] = (temp_diff <= 100)
                numerical_metrics['temperature_ranges']['within_200K'] = (temp_diff <= 200)

            if len(ref_nums['dimensions']) == 3 and len(gen_nums['dimensions']) == 3:
                dimension_errors = []
                for ref_dim, gen_dim in zip(ref_nums['dimensions'], gen_nums['dimensions']):
                    percent_error = abs(ref_dim - gen_dim) / ref_dim * 100 if ref_dim != 0 else float('inf')
                    dimension_errors.append(percent_error)
                numerical_metrics['avg_dimension_percent_error'] = sum(dimension_errors) / 3
                numerical_metrics['dimension_ranges']['exact'] = all(e == 0 for e in dimension_errors)
                numerical_metrics['dimension_ranges']['within_5_percent'] = all(e <= 5 for e in dimension_errors)
                numerical_metrics['dimension_ranges']['within_10_percent'] = all(e <= 10 for e in dimension_errors)
                numerical_metrics['dimension_ranges']['within_15_percent'] = all(e <= 15 for e in dimension_errors)

            if ref_nums['ratio'] is not None and gen_nums['ratio'] is not None:
                ratio_error = abs(ref_nums['ratio'] - gen_nums['ratio'])
                numerical_metrics['ratio_match'] = (ratio_error <= 0.01)
                numerical_metrics['ratio_error'] = ratio_error

            return {
                'bleu': bleu,
                'rouge1': rouge['rouge1'].fmeasure,
                'rouge2': rouge['rouge2'].fmeasure,
                'rougeL': rouge['rougeL'].fmeasure,
                'numerical': numerical_metrics,
                'reference_values': ref_nums,
                'generated_values': gen_nums
            }

        except Exception as e:
            print(f"Error computing metrics: {e}")
            print(f"Reference: {reference}")
            print(f"Candidate: {candidate}")
            raise

    def process_dataloader(self, dataloader, output_dir, split_name):
        """
        Process an entire dataloader to generate captions,
        compute metrics, and save results to a JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            for i in range(len(batch['caption'])):
                try:
                    caption = batch['caption'][i]
                    generated_caption = self.generate_caption(
                        batch['image_path'][i],
                        batch['xyz_path'][i],
                        batch['phase'][i],
                        batch['temperature'][i],
                        batch['is_original'][i]
                    )
                    metrics = self.compute_metrics(caption, generated_caption)
                    result = {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "phase": batch['phase'][i],
                        "temperature": batch['temperature'][i],
                        "is_original": batch['is_original'][i],
                        "original_caption": caption,
                        "generated_caption": generated_caption,
                        "metrics": metrics
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error processing batch {batch_idx}, sample {i}: {str(e)}")
                    continue
            if (batch_idx + 1) % 5 == 0:
                self._save_results(results, output_dir, split_name)
        self._save_results(results, output_dir, split_name)
        return results

    def _save_results(self, results, output_dir, split_name):
        """Save the results list as a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.model_name.replace('/', '_')
        output_file = os.path.join(output_dir, f"captions_{model_name_safe}_{split_name}_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


# -------------------------------
# Main entry point
# -------------------------------
if __name__ == "__main__":
    # Example API keys and model configurations
    api_keys = {
        'openrouter': ""
    }
    models_to_test = [
        {"name": "openai/gpt-4o-mini", "use_openrouter": True}
    ]
    # Initialize and set up the caption data module.
    data_module = TiO2CaptionDataModule(base_dir=BaseTDCD25Config.base_dir, batch_size=32, num_workers=0)
    data_module.setup()

    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader
    id_test_loader = data_module.id_test_dataloader
    ood_test_loader = data_module.ood_test_dataloader
    # Process each model configuration
    for model_config in models_to_test:
        print(f"\nProcessing with model: {model_config['name']}")
        pipeline = MultiModelCaptioningPipeline(
            api_keys=api_keys,
            model_name=model_config['name'],
            use_openrouter=model_config['use_openrouter']
        )
        model_output_dir = os.path.join("caption_results_id", model_config['name'].replace('/', '_'))
        splits = {"id_test": id_test_loader, "ood_test": ood_test_loader}
        for split_name, loader in splits.items():
            print(f"\nProcessing {split_name} split...")
            try:
                results = pipeline.process_dataloader(loader, model_output_dir, split_name)
                # (Optionally, print or further process the summary metrics.)
            except Exception as e:
                print(f"Error processing {split_name} split: {str(e)}")
                continue
