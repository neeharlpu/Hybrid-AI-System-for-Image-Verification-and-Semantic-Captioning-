from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load ONCE (important)
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Language codes (NLLB format)
LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Arabic": "arb_Arab"
}

def translate(text, tokenizer, model, target_lang):
    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=50
    )

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)