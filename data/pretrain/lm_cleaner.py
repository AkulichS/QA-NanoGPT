import re
from datasets import Dataset
import unicodedata


class DataCleaner:
    def __init__(
        self,
        min_length: int = 300,
        min_cyrilic_ratio: float = 0.7,
        max_digit_ratio: float = 0.2,
    ):
        # Minimum allowed document length
        self.min_length = min_length

        # Minimum ratio of Cyrillic characters required
        self.min_cyrilic_ratio = min_cyrilic_ratio

        # Maximum allowed ratio of digits
        self.max_digit_ratio = max_digit_ratio

        # Remove html tags
        self.remove_tags = re.compile(r'<(?!table>)[^>]+>')  # del all teg except <table>, # (r"<[^>]+>") - dell all teg

        # Regex: remove http limks
        self.remove_http = re.compile(r'(?:https?://|ftp://|www\.)\S*')

        # Regex: remove citation markers like [1] 
        self.remove_citation = re.compile(r"\[[^\[\]]*\]")   

        # Regex: remove empty quotation '', "", " "
        self.remove_empty_quotation = re.compile(r"['\"]\s*['\"]")

        # Regex: remove triple parentheses ( ( ( ) ) ) 
        self.remove_triple_parentheses = re.compile(r"\([^()]*\([^()]*\([^()]*\)[^()]*\)[^()]*\)")

        # Regex: remove double parentheses ( ( ) )
        self.remove_double_parentheses = re.compile(r"\([^()]*\([^()]*\)[^()]*\)")

        self.remove_parentheses = re.compile(
            r'\('
            r'(?!'                                                            # (?!...) — negative lookahead
                r'\s*\d{4}\s*\)'                                              # allow years like (2015), ( 2015 )
                r'|'
                r'\s*[а-яА-ЯёЁ 0-9,.:"\-]*[а-яА-ЯёЁ][а-яА-ЯёЁ 0-9,.:"\-]*\)'  # Cyrillic present, allowed chars only
            r')'
            r'[^)]*'                                                          # get full content in ()
            r'\)'
        )

        # Regex: remove bad marks like == Text ==
        self.remove_bad_marks = re.compile(r"[=]+\s*[=]*[A-Za-zА-ЯЁа-яё]+\s*[=]+\s*[=]*")

        # Regex: remove Unicode characters
        self.unicode_chars = re.compile(
            r'[\x00-\x08'     # NULL – backspace
            r'\x0B-\x0C'      # vertical tab, form feed
            r'\x0E-\x1F'      # other control
            r'\x7F-\x9F'      # C1 control characters    
            r'\u200B-\u200F'  # zero-width characters: ZWSP, ZWNJ, ZWJ, LRM, RLM           
            r'\u202A-\u202E'  # bidirectional text controls: LRE, RLE, PDF, LRO, RLO           
            r'\u2060-\u206F'  # invisible formatting characters             
            r'\uFE00-\uFE0F'  # variation selectors             
            r'\uFEFF'         # BOM / zero-width no-break space            
            r'\uFFF9-\uFFFB'  # interlinear annotation characters           
            r'\uE000-\uF8FF'  # Private Use Area (fonts icons, trash after PDF/OCR)
            r'\uFDD0-\uFDEF]' # Unicode noncharacters  
        )

        # Regex: separate joined words and punctuation
        self.fix_join = re.compile(
            r"(?<=[а-яё])(?=[А-ЯЁ0-9A-Z])"
            r"|(?<=[0-9])(?=[а-яёА-ЯЁA-Za-z])"
            r"|(?<=[A-Za-z])(?=[А-ЯЁа-яё])"
            r"|(?<=[А-ЯЁа-яё])(?=[A-Za-z])"
            r"|(?<=[^\W\d_][,.!?:;'\")])(?=[^\W\d_])"
        )    

        # Regex: fix punctuation chain
        self.fix_punct_chain = re.compile(r'[:;!?,.]{2,}(?=\s*[А-ЯЁA-Z])')  # re.compile(r"([^\W\d_][:;!?,.])(?:[:;!?,.]+)")

        # Regex: remove rest punctuation chain
        self.remove_punct_chain = re.compile(r'[:;!?,.]{2,}')

        # Regex: collapse multiple spaces but preserve newline
        self.multi_space = re.compile(r"[^\S\n]+")

        # Remove spaces around newlines
        self.spaces_around_newlines = re.compile(r" *(\n+) *")

        # Regex: collapse repeated punctuation
        self.double_symbols = re.compile(r'([.,:;/_!?~$@%^&*#+=\-—])(?:\s*\1)+')

        # Regex: fix multi newlines 
        self.fix_multi_newlines = re.compile(r"\n{3,}")

        # Regex: fix hyphen
        self.fix_hyphen = re.compile(r"([.,:;!?“'\"])([-—])")

        # Regex: remove spaces after hyphen in compound words
        self.fix_hyphen_space = re.compile(r"(\w)[-—]\s+(\w)")

        # Regex: remove space before punctuation: " ," -> ","
        self.space_before_punct = re.compile(r"\s+([,.;:!?])")

        # Regex: remove http ...
        self.remove_http = re.compile(
            r"(?im)^.*(?:https:|http:|ftp:|www\.).*\n+"
        ) # (?m) - multiline mode line-by-line (text...\n)
        
        # Regex: remove notes block
        self.notes_block = re.compile(
            r"\n{1,}(?:Примечания|Примечание)[\s\S]*$",
            re.IGNORECASE
        )

        # Regex: remove repeated phrases
        self.repeated_phrases = re.compile(
            r"(?:Сказки братьев Гримм\.?|Переводы \d+ года\.?)\n*"
        )

        self.allowed = set(
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            ".,:;!?<>[]()\\/_'\"—–=+-№*@#$%^&… \n"
        )
        
        self.replacements = {
            "«": '"',
            "»": '"',
            "“": '"',
            "”": '"',
            "„": '"',
            "’": "'",
        }

    # -----------------------------------------------------

    def pre_filter(self, text: str) -> bool:
        """
        Rejects documents with:
        - too short length
        - too few Cyrillic characters
        - too many digits
        """

        n = len(text)

        # Reject short documents immediately
        if n < self.min_length:
            return False

        ru = 0
        digits = 0

        # Single pass through characters
        for c in text:

            # Count Cyrillic and common letters
            if ('а' <= c <= 'я') or ('А' <= c <= 'Я') or c in "ёЁ .,:;!?-()\"'\n":
                ru += 1

            # Count digits
            elif c.isdigit():
                digits += 1

        # Reject if Cyrillic ratio is too low
        if ru / n < self.min_cyrilic_ratio:
            return False

        # Reject if too many digits (tables / formulas)
        if digits / n > self.max_digit_ratio:
            return False

        return True

    # -----------------------------------------------------

    def has_cjk(self, text: str) -> bool:
        """
        Detect presence of CJK characters
        (Chinese / Japanese / Korean).
        """

        for c in text:
            if '\u4e00' <= c <= '\u9fff':
                return True

        return False
    
    # -----------------------------------------------------
    
    def noisy_text(self, text: str) -> bool:
        """
        Detect noisy documents such as:
        - many extremely short lines (tables, lists)
        """

        short_lines = 0
        total_lines = 0

        for line in text.splitlines():
            line = line.strip()
            if line:
               total_lines += 1
               if len(line) < 70:  
                    short_lines += 1    

        # If more than half of lines are very short → likely table/list
        if short_lines / max(total_lines, 1) > 0.5:
            return True

        return False

    # -----------------------------------------------------

    def repetition_score(self, text: str) -> float:    
        txt = re.sub(r'[^a-zа-яё0-9 ]', ' ', text.lower()).split()
        return len(set(txt)) / max(len(txt), 1)

    # -----------------------------------------------------

    def remove_foreign_lines(self, text: str) -> str:
        """
        Remove lines that contain less then 70% allowed chars
        """

        cleaned = []

        for match in re.finditer(r'.*?(?:\n+|$)', text):
            line = match.group()
            if sum(c in self.allowed for c in line) / max(len(line), 1) == 1: 
                cleaned.append(line)  

        return ''.join(cleaned)
    
    # -----------------------------------------------------
    
    def remove_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        result = []
        for ch in text:
            # remove ONLY acute accent (stress mark)
            if ch == '\u0301':  # COMBINING ACUTE ACCENT
                continue
            result.append(ch)

        return unicodedata.normalize("NFC", "".join(result))

    # -----------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Normalize text formatting.
        """

        for k, v in self.replacements.items():
            text = text.replace(k, v)

        # Remove citation like [1]
        text = self.remove_citation.sub("", text)

        # Remove parentheses
        text = self.remove_triple_parentheses.sub("", text)
        text = self.remove_double_parentheses.sub("", text)
        text = self.remove_parentheses.sub("", text)

        # Remove html tags  
        text = self.remove_tags.sub(" ", text)

        # Remove http ...
        text = self.remove_http.sub(" ", text)

        # Replace non-breaking space
        text = text.replace("\xa0", " ")

        # Remove soft hyphen
        text = text.replace("\xad", " ")

        # Remove unicode characters
        text = self.unicode_chars.sub(" ", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove foreign lines
        text = self.remove_foreign_lines(text)

        if len(text) < self.min_length:
            return ''

        # Remove notes
        text = self.notes_block.sub(" ", text)

        # Remove repeated phrases
        text = self.repeated_phrases.sub(" ", text)

        # Remove punctuation chain
        text = self.fix_punct_chain.sub(r".", text)
        text = self.remove_punct_chain.sub(r" ", text)

        # Collapse multiple spaces
        text = self.multi_space.sub(" ", text)

        # Fix hyphen
        text = self.fix_hyphen.sub(r"\1 \2", text)

        # Fix spaces after hyphen in compound words
        text = self.fix_hyphen_space.sub(r"\1-\2", text)

        # Remove spaces around newlines
        text = self.spaces_around_newlines.sub(r"\1", text)

        # Remove space before punctuation: " ," -> ","
        text = self.space_before_punct.sub(r"\1", text)

        # Collapse repeated punctuation
        text = self.double_symbols.sub(r"\1", text)

        # Separate joined words and punctuation
        text = self.fix_join.sub(" ", text)

        # Fix multi newlines 
        text = self.fix_multi_newlines.sub("\n\n", text)

        return text.strip()

    # -----------------------------------------------------

    def process_example(self, example: dict) -> dict:
        """
        Process a single dataset example.
        """
        text = example.get("text", None)

        # normalize type
        if isinstance(text, list):
            text = " ".join(text)
        elif text is None:
            return {"text": None}

        # Preliminary filter
        if not self.pre_filter(text):
            return {"text": None}

        # Remove CJK documents
        if self.has_cjk(text):
            return {"text": None}
        
        # Remove noisy documents
        if self.noisy_text(text):
            return {"text": None}

        # Normalize text
        text = self.normalize_text(text)
        if not text:
            return {"text": None}
        
        text = self.remove_accents(text)

        # Remove extremely repetitive documents
        if self.repetition_score(text) < 0.6:
            return {"text": None}

        return {"text": text}

    # -----------------------------------------------------

    def process_dataset(self, dataset: Dataset, num_proc: int = 0) -> Dataset:
        """
        Apply cleaning pipeline to Dataset.
        """

        dataset = dataset.map(self.process_example, num_proc=num_proc, load_from_cache_file=False)

        # Remove rejected documents
        dataset = dataset.filter(lambda x: x["text"] is not None)

        return dataset