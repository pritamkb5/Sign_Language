import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Loads the key from your new .env file
load_dotenv()

class APISignTranslator:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("AIzaSyBDHI-z3AOEgRM7InJa2e3GGC4E6gPbi8U"))
        self.config = types.GenerateContentConfig(
            temperature=0.2,
        )

    def translate(self, gloss_text: str) -> str:
        if not gloss_text or not gloss_text.strip():
            return ""
            
        # Token Pre-Deduplication
        tokens = gloss_text.strip().split()
        dedup_tokens = []
        for token in tokens:
            if not dedup_tokens or dedup_tokens[-1].lower() != token.lower():
                dedup_tokens.append(token)
        cleaned_raw_input = " ".join(dedup_tokens)

        try:
            prompt = f"""You are a professional American Sign Language (ASL) to English interpreter.
The following input contains raw, noisy sign language glosses and fingerspelled tokens from a real-time recognition model:
"{cleaned_raw_input}"

Rules:
1. Handle repetitions: If words or gestures repeat (e.g., "hello hello" or "sorry sorry"), interpret them as a single instance of that thought or an intentional emphasis (e.g., "Hello!" or "I am very sorry").
2. Reconstruct context: Convert the conceptual sign tokens into a single coherent, natural, and grammatically complete English sentence.
3. Do NOT echo or copy-paste repetitive stutter words verbatim.
4. Output ONLY the final translated English sentence with no preambles, explanations, or quotes.
"""
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.config
            )
            return response.text.strip()
        except Exception as e:
            print(f"[APISignTranslator] Error: {e}")
            try:
                import wordninja
                clean_text = cleaned_raw_input.replace(" ", "").lower()
                words = wordninja.split(clean_text)
                return " ".join(words)
            except ImportError:
                return cleaned_raw_input