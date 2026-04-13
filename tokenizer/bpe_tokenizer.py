from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm


class BPETokenizer:
    def __init__(
            self, 
            save_dir: str = "bpe_tokenizer", 
            vocab_size: int = 32000, 
            min_frequency: int = 2
        ):
        self.save_dir = Path(save_dir)
        # self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.save_dir / "tokenizer.json"
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
            special_tokens=["<PAD>", "<BOS>", "<EOS>"],
        )

        # Add post-processor  
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )

        tokenizer.save(str(self.tokenizer_path))
        print("Tokenizer saved!")

        return tokenizer

    def from_file(self, filename):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{filename} not found")

        tokenizer = Tokenizer.from_file(filename)
        return tokenizer