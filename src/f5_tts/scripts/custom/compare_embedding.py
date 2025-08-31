import torch

# ---------- SETTINGS ----------
base_ckpt_path = "M:/git/github/celophi/F5-TTS/ckpts/Bilingual_EN_JP_custom/model_500.pt"
finetuned_ckpt_path = "M:/git/github/celophi/F5-TTS/ckpts/Bilingual_EN_JP_custom/model_8000.pt"
token_id_to_check = 2562  # the ID of the token you want to compare
# --------------------------------

def load_embeddings(ckpt_path, embedding_key="ema_model.transformer.text_embed.text_embed.weight"):
    """Load embedding tensor from a checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    embeddings = checkpoint['model_state_dict']['transformer.text_embed.text_embed.weight']
    return embeddings

# Load embeddings
base_embeddings = load_embeddings(base_ckpt_path)
finetuned_embeddings = load_embeddings(finetuned_ckpt_path)

# Select the token embedding
base_vec = base_embeddings[token_id_to_check]
finetuned_vec = finetuned_embeddings[token_id_to_check]

# Compare
diff = torch.norm(finetuned_vec - base_vec).item()
print(f"Token ID {token_id_to_check} embedding change magnitude: {diff:.6f}")

# Optional: print the embeddings themselves
print("Base embedding:", base_vec[:10])       # first 10 values
print("Finetuned embedding:", finetuned_vec[:10])