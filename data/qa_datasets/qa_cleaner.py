import re
from razdel import sentenize
import unicodedata
import pandas as pd
from qa_instructions import pick_instruction


class QACleaner:
    def __init__(
            self, 
            tokenizer, 
            long_answ_ratio=0.60, 
            min_answ_len=10,
            max_answ_len=150,
            max_cont_len=720
        ):

        self.tokenizer = tokenizer
        self.long_answ_ratio = long_answ_ratio
        self.min_answ_len = min_answ_len
        self.max_answ_len = max_answ_len
        self.max_cont_len = max_cont_len

        # --- replacements ---
        self.replacements = {
            "«": '"',
            "»": '"',
            "“": '"',
            "”": '"',
            "„": '"',
            "’": "'",
        }

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
            r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F'
            r'\u200B-\u200F\u202A-\u202E\u2060-\u206F'
            r'\uFE00-\uFE0F\uFEFF\uFFF9-\uFFFB'
            r'\uE000-\uF8FF\uFDD0-\uFDEF]'
        )    

        # Regex: collapse repeated punctuation symbols
        self.double_symbols = re.compile(r'([.,:;/_!?~$@%^&*#+=\-—])(?:\s*\1)+')

        # Regex: fix punctuation chain
        self.fix_punct_chain = re.compile(r'[:;!?,.]{2,}(?=\s*[А-ЯЁA-Z])')

        # Regex: remove rest punctuation chain
        self.remove_punct_chain = re.compile(r'[:;!?,.]{2,}')

        # Regex: separate joined words and punctuation
        self.fix_join = re.compile(
            r"(?<=[а-яё])(?=[А-ЯЁ0-9A-Z])"
            r"|(?<=[0-9])(?=[а-яёА-ЯЁA-Za-z])"
            r"|(?<=[A-Za-z])(?=[А-ЯЁа-яё])"
            r"|(?<=[А-ЯЁа-яё])(?=[A-Za-z])"
            r"|(?<=[^\W\d_][,.!?:;'\")])(?=[^\W\d_])"
        )    

        # Regex: collapse multiple spaces but preserve newline
        self.multi_space = re.compile(r"[^\S\n]+")

        # Regex: remove space before punctuation: " ," -> ","
        self.space_before_punct = re.compile(r"\s+([,.;:!?])")

        # Regex: separate text to sentences
        # self.sent_split =  re.compile(r'(?<!\b[А-ЯЁA-Z])(?<!\b[а-яёa-z])(?<!\d)[.!?]+(?=\s+[А-ЯЁA-Z])') # re.compile(r'[.!?]+')

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

        for k, v in self.replacements.items():
            text = text.replace(k, v)

        text = self.remove_citation.sub("", text)

        text = self.remove_triple_parentheses.sub("", text)
        text = self.remove_double_parentheses.sub("", text)
        text = self.remove_parentheses.sub("", text)

        text = self.remove_tags.sub(" ", text)
        text = self.remove_http.sub("", text) 
        text = self.remove_bad_marks.sub("", text) 

        text = text.replace("\xa0", " ")
        text = text.replace("\xad", " ")
        text = self.unicode_chars.sub(" ", text)

        text = self.double_symbols.sub(r"\1", text)
        text = self.fix_punct_chain.sub(r".", text)
        text = self.remove_punct_chain.sub(r" ", text)

        text = self.multi_space.sub(" ", text)
        text = self.space_before_punct.sub(r"\1", text)
        text = self.fix_join.sub(" ", text)

        return text.strip()

    # -----------------------------------------------------

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:

        # ---- remove accents and normalize ----
        for col in ["context", "question", "answer"]:
            df[col] = df[col].map(self.remove_accents).map(self.normalize_text)

        # ---- tokens length (answer and context) ----
        df["answer_len"] = df["answer"].map(self._encoded_text_len)
        df["context_len"] = df["context"].map(self._encoded_text_len)

        # ---- length filters ----
        df = df[(df["answer_len"] > 3) & (df["context_len"] > 30)].copy()

        #  ---- assign instruction types  ----
        df["instruction_type"] = "default"

        #  ---- split to long and short answers ----
        mask_long = df["answer_len"] > 12
        df_long = df[mask_long].copy()
        df = df[~mask_long].copy()

        #  ---- select samples for long answers  ----
        long_n = int(self.long_answ_ratio * df.shape[0])
        long_pool = df[df["context_len"] <= self.max_answ_len]
        long_idx = long_pool.sample(
            n=min(long_n, len(long_pool)), random_state=42
        ).index
        df.loc[long_idx,  "instruction_type"] = "detail"

        # ---- align answers to context per instruction type ----
        mask_default = df["instruction_type"] == "default"
        mask_detail  = df["instruction_type"] == "detail"

        if mask_default.any():  # align answers to instruction "default"
            df.loc[mask_default, "answer"] = [
                self._find_best_sentence(c, a)
                for c, a in zip(df.loc[mask_default, "context"],
                                df.loc[mask_default, "answer"])
            ]

        if mask_detail.any():   # align answers to instruction "detail"
            results = [
                self._build_detail_answer(c, a)
                for c, a in zip(df.loc[mask_detail, "context"],
                                df.loc[mask_detail, "answer"])
            ]
            answers, fallen_back = zip(*results)
            df.loc[mask_detail, "answer"] = answers

            # rows where detail had no valid neighbors → fallback to default
            fallback_mask = df.loc[mask_detail].index[list(fallen_back)]
            df.loc[fallback_mask, "instruction_type"] = "default"   

        # ---- drop wrong answers ----
        df = df[df["answer"].notna()].copy()  

        # ---- concat both ----
        df = pd.concat([df_long, df], ignore_index=True)  

        # ---- drop too short and too long answers ----
        answer_len = df['answer'].map(self._encoded_text_len)
        df = df[(answer_len >= self.min_answ_len) & (answer_len <= self.max_answ_len)].copy()

        # ---- map instruction ----
        df["instruction"] = df["instruction_type"].apply(pick_instruction)

        # ---- truncate long contexts (sentence-aware) ----
        long_ctx_mask = df["context_len"] > self.max_cont_len
        df.loc[long_ctx_mask, "context"] = df.loc[long_ctx_mask, "context"].map(
            self._truncate_context
        )

        # ---- answer must be in context ----
        df = df[df.apply(lambda x: x["answer"].lower() in x["context"].lower(), axis=1)].copy()

        return df.reset_index(drop=True)
    
    # -----------------------------------------------------
    
    def _uniq_words(self, text: str) -> set[str]:
        text = re.sub(r'[^a-zа-яё0-9 ]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return set(text.split())
    
    # -----------------------------------------------------

    def _overlap_score(self, answer_words: set, sent_words: set) -> float:
        if not answer_words:
            return 0.0
        return len(answer_words & sent_words) / len(answer_words)
    
    # -----------------------------------------------------

    def _encoded_text_len(self, text):
        return len(self.tokenizer.encode(text))
    
    # ----------------------------------------------------- 
  
    def _split_sentences(self, text: str):
        return [s.text for s in sentenize(text)]
    
    # ----------------------------------------------------- 

    def _truncate_context(self, text: str) -> str:
        sentences = self._split_sentences(text)
        
        result = []
        total = 0

        for s in sentences:
            l = self._encoded_text_len(s)
            if total + l > self.max_cont_len:
                break
            result.append(s)
            total += l

        return " ".join(result)

    # -----------------------------------------------------   

    def _find_anchor(
        self,
        context: str,
        answer: str,
        threshold: float = 0.7,
    ) -> tuple[list[str], int]:   
        """
        Returns (sentences, best_idx).
        best_idx is -1 if no sentence meets the threshold.
        """
        sentences = self._split_sentences(context)  
        answer_words = self._uniq_words(answer)

        best_idx, best_score = -1, 0.0
        for i, sent in enumerate(sentences):
            score = self._overlap_score(answer_words, self._uniq_words(sent))
            if score > best_score:
                best_score, best_idx = score, i

        if best_score < threshold:
            return sentences, -1

        return sentences, best_idx

    # -----------------------------------------------------   

    def _find_best_sentence(self, context: str, answer: str) -> str:
        sentences, best_idx = self._find_anchor(context, answer)
        if best_idx == -1:
            return None
        return sentences[best_idx]
    
    # -----------------------------------------------------

    def _build_detail_answer(
        self,
        context: str,
        answer: str,
        min_len: int = 30,
    ) -> tuple[str, bool]:
        sentences, best_idx = self._find_anchor(context, answer)
        if best_idx == -1:
            return None, True

        up   = sentences[best_idx - 1] if best_idx > 0 else None
        down = sentences[best_idx + 1] if best_idx < len(sentences) - 1 else None

        up_valid   = up   is not None and len(up)   > min_len
        down_valid = down is not None and len(down) > min_len

        if not (up_valid or down_valid):
            return sentences[best_idx], True

        parts = [
            s for s, valid in [
                (up, up_valid),              # up sentence
                (sentences[best_idx], True), # anchor sentence contain answer
                (down, down_valid)           # down sentence
            ] if valid
        ]
        return " ".join(parts), False

  
