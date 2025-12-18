import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device():
    """
    Returns the device string: 'cudo' (Nvidia GPU), 'mps' (Mac GPU), or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# The below code is added to load the models with 4-bit quantization
def load_model(target_model_id, draft_model_id):
    device = get_device()
    print(f"Loading models on {device}...")

    # 1. The 4-bit Config (The "memory Hack"). The idea is to load a zipped 4-bit weights into VRAM, and GPU unzipping it to 16-bit again for computation.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_use_double_quant = True,
    ) 

    # 2. Load the Target Model (The Big Boss) - Quantized
    print(f"Loading Target; {target_model_id}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        quantization_config = bnb_config,
        device_map = "auto",
    )

    # 3. Load the Draft Model (The Intern) - Standard FP16
    # Quick note: Draft models are usually smaller (1B), so we often keep then im 16-bit instead of quantizing them
    print(f"Loading Draft: {draft_model_id}...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    # Load the tokenizer 
    # Point of info: Speculative Decoding works only of both models share the same vocuabulary (you cannot have a target in English and a draft in French, for instance)
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    return target_model, draft_model, tokenizer

