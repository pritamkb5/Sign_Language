import time
import queue
import threading
import re
from src.inference.api_translator import APISignTranslator

class SentenceBuilder:
    def __init__(self, idle_timeout: float = 2.0):
        self.idle_timeout = idle_timeout
        self.raw_words = []
        self.current_word = []
        self.last_char = None
        self.last_char_time = 0.0
        self.last_hand_seen = time.time()
        
        self.final_translated_sentence = ""
        self.is_translating = False
        
        # Initialize the API translator
        self.translator = APISignTranslator()
        
        self._translation_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self._worker_thread.start()
        
        self.on_update_callback = None

    def _translation_worker(self):
        while True:
            text = self._translation_queue.get()
            if text is None:
                break
                
            clean_text = self._filter_gibberish(text)
            if not clean_text.strip():
                self.final_translated_sentence = ""
            else:
                self.is_translating = True
                translated = self.translator.translate(clean_text)
                self.final_translated_sentence = translated
                self.is_translating = False
                
            if self.on_update_callback:
                self.on_update_callback()
                
            self._translation_queue.task_done()

    def _filter_gibberish(self, text: str) -> str:
        """Strips out non-words, rapid flickering characters, and unintended artifacts."""
        words = text.split()
        clean_words = []
        for word in words:
            # Keep known valid words
            if word.lower() in ["hello", "thanks", "yes", "no", "please", "help", "sorry", "name", "more", "stop", "love", "want", "eat", "drink", "friend"]:
                clean_words.append(word)
                continue
                
            # Filter rapid flickering: consecutive identical letters (e.g. "HHHELLO" -> "HELLO")
            word = re.sub(r'(.)\1{2,}', r'\1\1', word) 
            
            # Filter pure consonant gibberish of length >= 3
            if len(word) >= 3 and not re.search(r'[AEIOUYaeiouy]', word):
                continue
                
            # Filter standalone consonants (except I, A, O)
            if len(word) == 1 and word.upper() not in ["I", "A", "O"]:
                continue
                
            if len(word) > 0:
                clean_words.append(word)
                
        return " ".join(clean_words)

    def trigger_translation(self):
        """Pushes current state to the background worker for real-time translation."""
        current = "".join(self.current_word)
        buffered = " ".join(self.raw_words)
        full_text = f"{buffered} {current}".strip()
        
        # Empty stale tasks from queue
        while not self._translation_queue.empty():
            try:
                self._translation_queue.get_nowait()
                self._translation_queue.task_done()
            except queue.Empty:
                break
                
        if full_text:
            self._translation_queue.put(full_text)
        else:
            self.final_translated_sentence = ""
            if self.on_update_callback:
                self.on_update_callback()

    def add_char(self, char: str):
        """Debounces and adds a confirmed letter from the live camera feed."""
        now = time.time()
        self.last_hand_seen = now

        # Prevent duplicate insertions of the same letter (edge-triggered)
        if char == self.last_char:
            return
        
        self.current_word.append(char)
        self.last_char = char
        if self.on_update_callback:
            self.on_update_callback()

    def reset_last_char(self):
        """Clears the last seen character so it can be triggered again after a drop."""
        self.last_char = None

    def delete(self):
        """Removes the last character or word."""
        if self.current_word:
            self.current_word.pop()
        elif self.raw_words:
            self.raw_words.pop()
        if self.on_update_callback:
            self.on_update_callback()

    def clear(self):
        """Clears all states."""
        self.raw_words = []
        self.current_word = []
        self.last_char = None
        self.final_translated_sentence = ""
        # Cancel any pending translations
        while not self._translation_queue.empty():
            try:
                self._translation_queue.get_nowait()
                self._translation_queue.task_done()
            except queue.Empty:
                break
        if self.on_update_callback:
            self.on_update_callback()

    def add_space(self):
        """Commits the currently accumulated characters into a word."""
        if self.current_word:
            word = "".join(self.current_word)
            self.raw_words.append(word)
            self.current_word = []
        if self.on_update_callback:
            self.on_update_callback()

    def get_display_text(self) -> str:
        """Returns the raw text to show on screen."""
        current = "".join(self.current_word)
        buffered = " ".join(self.raw_words)
        return f"{buffered} {current}".strip()