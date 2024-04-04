import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

prompt = "Draw a dog"

# Load the pre-trained AutoencoderKL model
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", use_auth_token=True
)

# Load the pre-trained CLIPTokenizer and CLIPTextModel
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Set the forward method of the AutoencoderKL model to the decode method
vae.forward = vae.decode

vae.eval()
# Export the AutoencoderKL model to ONNX format
torch.onnx.export(
    vae,                                   # PyTorch model to be exported
    (torch.randn(1, 4, 64, 64), False),   # Sample input to the model
    "vae.onnx",                           # Output ONNX file name
    input_names=["latent_sample", "return_dict"],   # Names for input nodes in the ONNX graph
    output_names=["sample"],              # Names for output nodes in the ONNX graph
    dynamic_axes={                        # Dynamic axes for variable-sized dimensions
        "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    },
    do_constant_folding=True,             # Enable constant folding for optimization
    opset_version=14,                     # ONNX opset version to use
)


# Tokenize the prompt using the CLIPTokenizer
text_input = tokenizer(
    prompt,                                     # The text prompt to tokenize
    padding="max_length",                       # Pad to the maximum length
    max_length=tokenizer.model_max_length,      # Maximum length of the tokenized sequence
    truncation=True,                            # Truncate sequences that exceed max_length
    return_tensors="pt",                        # Return PyTorch tensors
)


# Export the CLIPTextModel to ONNX format
torch.onnx.export(
    text_encoder,                                      # PyTorch model to be exported
    (text_input.input_ids.to(torch.int32)),            # Sample input to the model
    "encoder.onnx",                                    # Output ONNX file name
    input_names=["input_ids"],                         # Names for input nodes in the ONNX graph
    output_names=["last_hidden_state", "pooler_output"],# Names for output nodes in the ONNX graph
    dynamic_axes={                                     # Dynamic axes for variable-sized dimensions
        "input_ids": {0: "batch", 1: "sequence"},
    },
    opset_version=14,                                  # ONNX opset version to use
    do_constant_folding=True,                          # Enable constant folding for optimization
)
