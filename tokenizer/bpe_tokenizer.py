from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm


class BPETokenizer:
    def __init__(
            self, 
            save_dir: str | None = None, 
            vocab_size: int = 32000, 
            min_frequency: int = 2
        ):

        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)            
            self.tokenizer_path = self.save_dir / "tokenizer.json"  
        else:
            self.tokenizer_path = None

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        

    def train(self, dataset):
        tokenizer = ByteLevelBPETokenizer()

        def text_iterator():
            for example in tqdm(dataset, total=len(dataset), desc="Training BPE"):
                yield example["text"]

        tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=[
                "<PAD>", "<BOS>", "<EOS>", "<CTX>", "</CTX>", "<D>", "</D>",
                 "<Q>", "</Q>", "<INST>", "</INST>", "<ANS>", "</ANS>"
            ],
        )

        # Add post-processor  
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )

        if self.tokenizer_path:
            tokenizer.save(str(self.tokenizer_path))
            print("Tokenizer saved!")

        return tokenizer
    

    @staticmethod
    def from_file(filename: str | Path):
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"{filename} not found")

        tokenizer = Tokenizer.from_file(str(filename))
        return tokenizer