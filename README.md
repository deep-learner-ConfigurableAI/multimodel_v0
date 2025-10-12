# MultiModel: Efficient Vision-Language Model with Q-Former Architecture

MultiModel is a lightweight and efficient vision-language model that combines the power of CLIP's visual encoder with a GPT-style decoder, bridging the gap between vision and language tasks. The model leverages a Q-Former architecture inspired by state-of-the-art multimodal systems to achieve high-quality image captioning capabilities while maintaining a small footprint.

##  Key Features

- **Efficient Architecture**: Uses frozen CLIP vision encoder and GPT-Neo language model with trainable bridging layers
- **Q-Former Design**: Employs learnable query tokens to extract relevant visual information
- **Small Footprint**: Significantly smaller than most vision-language models while maintaining quality outputs
- **Low Latency**: Fast inference times suitable for real-time applications
- **Easy to Use**: Simple API for image captioning and analysis

##  Experimental Approach

Our approach focuses on efficient multimodal representation learning through a Q-Former architecture, which uses learnable query tokens to bridge the visual and textual domains. Key components of our experimental setup include:

1. **Visual Encoder**: Frozen CLIP ViT-Base-Patch32 for efficient and robust image feature extraction
2. **Cross-Attention Mechanism**: Multiple layers of cross-attention between query tokens and image features
3. **Query-Based Visual Representation**: Learnable query tokens that extract relevant visual information
4. **Language Decoder**: GPT-Neo for high-quality text generation conditioned on visual features

### Feature Flow Architecture

Our model employs a sophisticated feature flow mechanism to bridge vision and language:

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│             │     │                  │     │                │
│  Image      │     │  Query Tokens    │     │  Text Tokens   │
│  Features   │─────►  + Cross         │─────►  Generation    │
│  (CLIP)     │     │  Attention       │     │  (GPT-Neo)     │
│             │     │                  │     │                │
└─────────────┘     └──────────────────┘     └────────────────┘
```

1. **Image Feature Extraction**: CLIP encoder processes the image into patch-level features (49 patches for 224x224 images)

2. **Query Token Processing**:
   - Learnable query tokens act as information extractors 
   - First attend to each other via self-attention (like in transformers)
   - Then extract relevant information from image features via cross-attention
   - This creates a refined visual representation

3. **Cross-Modal Integration**:
   - Text decoder attends to the refined query tokens via cross-attention
   - This allows text generation to focus on relevant visual elements
   - The model can selectively extract and use visual information based on the current text context

4. **Attention Visualization**: Analysis shows that query tokens learn to attend to semantically meaningful regions, with attention weights highlighting relevant objects for specific caption words

##  Model Performance

Our model achieves strong performance on image captioning tasks with significantly fewer parameters than comparable models:

| Model             | Total Parameters | Trainable Parameters | Batch Size | Image Size | Inference Time (ms) |
|-------------------|-----------------|---------------------|------------|------------|---------------------|
| MultiModel (Ours) | 574M            | 307M                | 8          | 224x224    | ~50                 |
| BLIP-2            | ~7B             | ~7B                 | 4          | 224x224    | ~250                |
| LLaVA             | ~13B            | ~13B                | 2          | 336x336    | ~500                |
| CLIP + GPT-4      | ~20B            | ~20B                | 1          | 224x224    | ~2000               |

### Training Configuration

| Setting                | Value                  |
|------------------------|------------------------|
| Batch Size             | 8                      |
| Image Size             | 224x224                |
| Caption Length         | 20 tokens              |
| Gradient Accumulation  | 4 steps                |
| Training Dataset       | COCO (80k images)      |

### Model Parameters

Our model consists of several key components with the following parameter distribution:

| Component       | Total Parameters | Trainable Parameters | Percentage Trainable |
|-----------------|-----------------|---------------------|---------------------|
| CLIP Encoder    | 152.06M         | 0M                  | 0%                  |
| GPT-Neo Decoder | 355.80M         | 240.36M             | 67.56%              |
| Custom Q-Former | 66.20M          | 66.20M              | 100%                |
| **Total**       | **574.06M**     | **306.56M**         | **53.40%**          |

While the total model size is 574M parameters, we only train 307M parameters (53.40%) by keeping the CLIP encoder completely frozen and freezing 32.44% of the GPT-Neo decoder. This significantly reduces training time and computational requirements while preserving the pre-trained knowledge of both the visual encoder and parts of the language model.

### Training Methodology

Our training process followed a two-phase approach:

**Phase 1:**
- Froze 100% of encoder weights (CLIP)
- Froze 70% of decoder weights (GPT-Neo)
- Learning Rate: 2e-5
- Epochs: 7
- Batch: 16
- Training on COCO captions dataset

**Phase 2:**
- Froze 100% of encoder weights (CLIP)
- Reduced freezing to 50% of decoder weights
- Learning Rate: 2e-4
- Batch: 8
- Additional fine-tuning to improve caption quality

This progressive unfreezing strategy allows the model to first learn cross-modal alignment with more constraints, then refine the language generation capabilities in the second phase.

##  Why This Approach Is Powerful

1. **Efficiency**:
   - By freezing the pre-trained vision and language components, we drastically reduce the number of trainable parameters
   - The Q-Former architecture only needs to learn the cross-modal alignment, not the entire visual or linguistic representations

2. **Scalability**:
   - The model can run on consumer-grade hardware (including Apple Silicon)
   - Low memory requirements allow for larger batch sizes during training

3. **Performance**:
   - Token similarity analysis shows that our query tokens learn meaningful visual representations
   - The cross-attention mechanism effectively bridges the modality gap

4. **Flexibility**:
   - The model can be adapted for various downstream tasks beyond captioning
   - Different language models can be plugged in without retraining the entire system

## 📋 Usage

### Using the Model from Hugging Face Hub

The model is available on Hugging Face Hub and can be easily used in your projects:

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Download model files from Hugging Face Hub
model_name = "verma75preetam/qvision-mutlimodel-base"
config_path = hf_hub_download(repo_id=model_name, filename="config.json")
encoder_path = hf_hub_download(repo_id=model_name, filename="encoder.safetensors")
decoder_path = hf_hub_download(repo_id=model_name, filename="decoder.safetensors")
gpt_decoder_path = hf_hub_download(repo_id=model_name, filename="gpt_decoder.safetensors")

# Download helper scripts
encoder_py_path = hf_hub_download(repo_id=model_name, filename="encoder.py")
decoder_model_py_path = hf_hub_download(repo_id=model_name, filename="decoder_model.py")
generate_py_path = hf_hub_download(repo_id=model_name, filename="generate.py")

# Add scripts to path
import sys, os
sys.path.append(os.path.dirname(encoder_py_path))

# Import model classes
from encoder import CLIPEncoder
from decoder_model import ResnetGPT2Wrapper
from transformers import GPTNeoForCausalLM, AutoTokenizer
from generate import generate_caption

# Load model components
# (See test_huggingface_model.py for complete implementation)
```

For a complete implementation, see the `test_huggingface_model.py` file in this repository.

### Running Inference with driver.py

To generate captions for your own images using the local model:

```bash
# Basic usage
python driver.py

# To use with your own images, modify the folder_path in driver.py:
# folder_path = "/path/to/your/images"
```

The driver script will:
1. Load the pretrained model with frozen weights
2. Process images from the specified folder (supports jpg, jpeg, png, bmp, webp formats)
3. Generate captions using a temperature-controlled sampling strategy
4. Visualize the images alongside their generated captions
5. Optionally show attention maps when debug_similarity=True

### Training with playground.ipynb

To train the model on your own data or fine-tune the existing model:

1. Open the `experiment/playground.ipynb` notebook in Jupyter or VSCode
2. Follow the cells for setup and configuration
3. Adjust training parameters in the TrainingConfig class:
   - `batch_size`: Number of images per batch
   - `epochs`: Number of training epochs
   - `lr`: Learning rate
   - `number_of_items`: Number of training samples to use

4. Run the training cells to begin model training

Example training configuration:

```python
class TrainingConfig(BaseModel):
    batch_size: ClassVar[int] = 8
    epochs: ClassVar[int] = 10
    lr: ClassVar[float] = 2e-4
    accumulation_steps: ClassVar[int] = 4
    number_of_items: ClassVar[int] = 80000  # Set to lower number for faster experimentation
```

### Advanced Analysis

We provide tools for analyzing the model's internal representations:

1. `token_similarity_analysis.ipynb`: Visualize and analyze token similarity patterns
2. `debug_similarity.py`: Debug the similarity between tokens for understanding attention patterns
3. `calc_params.py`: Calculate detailed parameter statistics for each component

To calculate the actual trainable parameters in the model:

```bash
python3 calc_params.py
```

This will display a breakdown of parameters by component, showing total vs. trainable parameters for each part of the architecture.

## 📊 Experiments and Results

Our experiments demonstrate several key findings:

1. **Attention Visualization**: Cross-attention maps show the model effectively learns to associate textual and visual elements
2. **Token Similarity Analysis**: Global token learns discriminative visual information
3. **Hallucination Reduction**: Our Q-Former architecture reduces hallucinations compared to simpler approaches

The model achieves good performance on the COCO validation set, with qualitative results showing strong alignment between generated captions and image content.

### Token Similarity Analysis

We conducted in-depth analysis of how different components of our model interact:

```
   ┌─────────────┐      ┌──────────────────┐     
   │ Global      │      │                  │     
   │ Context     │─────►│ Query Tokens     │     
   │ Token       │      │                  │     
   └─────────┬───┘      └────────┬─────────┘     
             │                   │               
             │                   │               
             ▼                   ▼               
    ┌────────────────────────────────────┐       
    │        Image Features              │       
    └────────────────────────────────────┘       
```

Our token similarity analysis reveals:

1. **Global-Query Similarities**: The global context token shows high correlation with a subset of query tokens, suggesting it acts as an orchestrator for more specific visual queries

2. **Discriminative Power**: We measure how effectively the global token distinguishes important visual elements across different images (higher is better):
   - Average Discriminative Power: 0.52
   - Higher for images with distinctive subjects, lower for cluttered scenes

3. **Attention Distribution**: The cross-attention weights between query tokens and image features show semantic clustering, where related visual concepts receive similar attention patterns

4. **Progressive Learning**: Throughout training, we observe the evolution of token similarities, with initial random patterns gradually becoming more structured and semantically meaningful

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multimodel.git
cd multimodel

# Install dependencies
pip install -r requirements.txt
```

##  Project Structure

```
multimodel/
├── driver.py                    # Inference script
├── encoder.py                   # CLIP encoder implementation
├── decoder_model.py             # Q-Former and decoder implementation
├── generate.py                  # Caption generation utilities
├── datasetlite.py               # Dataset handling
├── setup_model.py               # Model initialization
├── utils.py                     # Helper functions
├── checkpoint.pth               # Pretrained model weights
├── experiment/                  # Jupyter notebooks
│   ├── playground.ipynb         # Training notebook
│   └── token_similarity_analysis.ipynb  # Analysis tools
└── train2017/                   # COCO images
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- CLIP by OpenAI for the vision encoder
- GPT-Neo for the language decoder
- COCO dataset for training data
