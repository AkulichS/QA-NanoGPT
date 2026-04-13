import random
import torch
from torch.utils.data import Dataset


# template for getting special tokens length
TEMPLATE = """<BOS>
<INST></INST>

<CTX>
<D1></D>
<D2></D>
<D3></D>
<D4></D>
<D5></D>
<D6></D>
<D7></D>
<D8></D>
<D9></D>
</CTX>

<Q></Q>

<ANS></ANS>
<EOS>"""   # <SRC></SRC>


class QADataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_seq_len=1024,
        max_answer=150
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.post_processor = None
        self.max_seq_len = max_seq_len
        self.max_answer = max_answer
        self.teh_tokens_len = len(self.tokenizer.encode(TEMPLATE).ids)


    def _pack_documents(self, pos_doc, neg_docs, max_context_tokens):

        packed_docs = []
        total_tokens = 0
        
        pos_ids = self.tokenizer.encode(pos_doc).ids
        length = len(pos_ids)

        if length > max_context_tokens:
            packed_docs.append(self.tokenizer.decode(pos_ids[:max_context_tokens]))
            return packed_docs, 1

        packed_docs.append(pos_doc)
        total_tokens += length

        for doc in neg_docs:
            doc_ids = self.tokenizer.encode(doc).ids
            length = len(doc_ids)

            if total_tokens + length >= max_context_tokens:
                remain_tokens = max_context_tokens - total_tokens
                if remain_tokens > 10:
                    packed_docs.append(self.tokenizer.decode(doc_ids[:remain_tokens]))
                break

            packed_docs.append(doc)
            total_tokens += length

            if len(packed_docs) >= 9:
                break
 
        random.shuffle(packed_docs)

        # try:
        #     pos_idx = packed_docs.index(pos_doc) + 1
        # except ValueError:
        #     pos_idx = 0

        return packed_docs #, pos_idx
    

    def build_prompt(self, docs, inst, question, answer):  
        context = ''
        for i, d in enumerate(docs):
            context += f"<D{i+1}>{d}</D>\n"

        prompt = (
            f"<BOS>\n"
            f"<CTX>\n"
            f"{context}"
            f"</CTX>\n\n"
            f"<INST>{inst}</INST>\n\n"
            f"<Q>{question}</Q>\n\n"
            f"<ANS>"
        )

        # if doc_idx > 0:
        #     src = f"D{doc_idx}"
        # else:
        #     src = '0'

        full_text = (
            f"{prompt}{answer}</ANS>\n"
            # f"<SRC>{src}</SRC>\n"
            f"<EOS>"
        )

        return prompt, full_text
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):

        sample = self.data[idx]

        inst = sample["instruction"] 
        inst_ids = self.tokenizer.encode(inst).ids

        question = sample["question"] 

        answer = sample["answer"]
        answer_ids = self.tokenizer.encode(answer).ids

        # --- truncate answer ---
        if len(answer_ids) > self.max_answer:
            answer_ids = answer_ids[:self.max_answer]
            answer = self.tokenizer.decode(answer_ids)

        assert answer != '', 'Error: empty answer'

         # --- compute space left for context ---
        max_context_tokens = (
            self.max_seq_len
            - len(inst_ids)
            - len(answer_ids)
            - self.teh_tokens_len
            - len(self.tokenizer.encode(question).ids) - 4
        )

        # --- pack documents ---
        packed_docs = self._pack_documents(
            sample["positive_docs"][0], 
            sample["negative_docs"], 
            max_context_tokens
        )

        if len(packed_docs) == 1:
            if answer not in packed_docs[0]:
                answer = 'Ответ не найден.'
                # doc_idx = 0

        prompt, full_text = self.build_prompt(packed_docs, inst, question, answer)

        prompt_ids = self.tokenizer.encode(prompt).ids
        full_ids = self.tokenizer.encode(full_text).ids
        full_ids_len = len(full_ids)

        if full_ids_len > self.max_seq_len:
            return None

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # --- loss mask ---
        labels[:len(prompt_ids)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    

def qa_collate_fn(batch, pad_token_id):

    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    max_len = max(len(x) for x in input_ids)

    padded_inputs = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):

        pad_len = max_len - len(inp)

        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), pad_token_id)])
        )

        padded_labels.append(
            torch.cat([lab, torch.full((pad_len,), -100)])
        )

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels)
    }










# import random
# import torch
# from torch.utils.data import Dataset

# class QADataset(Dataset):

#     def __init__(
#         self,
#         data,
#         tokenizer,
#         max_seq_len=1024,
#         max_answer=150
#     ):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.tokenizer.post_processor = None
#         self.max_seq_len = max_seq_len
#         self.max_answer = max_answer
#         template = "<BOS>\n<C>\n<D></D>\n<D></D>\n<D></D>\n<D></D>\n</C>\n\n<Q>\n\n</Q>\n\n<A>\n\n</A>\n<EOS>"
#         self.teh_tokens = len(self.tokenizer.encode(template).ids)


#     def _pack_documents(self, pos_doc, neg_docs, max_context_tokens):

#         packed_docs = []
#         total_tokens = 0
        
#         tokens = self.tokenizer.encode(pos_doc).ids
#         length = len(tokens)

#         if length > max_context_tokens:
#             packed_docs.append(self.tokenizer.decode(tokens[:max_context_tokens]))
#             return packed_docs

#         packed_docs.append(pos_doc)
#         total_tokens += length

#         for doc in neg_docs:
#             tokens = self.tokenizer.encode(doc).ids
#             length = len(tokens)

#             if total_tokens + length >= max_context_tokens:
#                 last_tokens = max_context_tokens - total_tokens

#                 packed_docs.append(self.tokenizer.decode(tokens[:last_tokens]))
#                 break

#             packed_docs.append(doc)
#             total_tokens += length

#         random.shuffle(packed_docs)
#         return packed_docs
    

#     def build_context(self, docs):

#         context = "<C>\n"

#         for d in docs:
#             context += f"<D>{d}</D>\n"

#         context += "</C>\n"

#         return context
    

#     def build_prompt(self, context, question, answer):  

#         prompt = (
#             f"<BOS>\n{context}\n"
#             "<Q>\n"
#             f"{question}\n"
#             "</Q>\n\n"
#             "<A>\n"
#         )

#         full_text = prompt + answer + "\n</A>\n<EOS>"

#         return prompt, full_text
    

#     def __len__(self):
#         return len(self.data)
    

#     def __getitem__(self, idx):

#         sample = self.data[idx]
#         question = sample["question"] 

#         answer = sample["answer"]
#         answer_ids = self.tokenizer.encode(answer).ids

#         if len(answer_ids) > self.max_answer:
#             answer_ids = answer_ids[:self.max_answer]
#             answer = self.tokenizer.decode(answer_ids)

#         max_context_tokens = (
#             self.max_seq_len
#             - self.max_answer
#             - self.teh_tokens
#             - len(self.tokenizer.encode(question).ids)
#         )

#         packed_docs = self._pack_documents(
#             sample["positive_docs"][0], 
#             sample["negative_docs"], 
#             max_context_tokens
#         )

#         # if len(packed_docs) == 1:
#         #     if answer not in packed_docs[0]:
#         #         answer = 'К сожалению, ответ не найден.'

#         context = self.build_context(packed_docs)
#         prompt, full_text = self.build_prompt(context, question, answer)

#         prompt_ids = self.tokenizer.encode(prompt).ids
#         full_ids = self.tokenizer.encode(full_text).ids
#         full_ids_len = len(full_ids)

#         if full_ids_len > self.max_seq_len:
#             start = full_ids_len - self.max_seq_len
#             full_ids = full_ids[start: ]
#             prompt_len = max(0, len(prompt_ids) - start)
#         else:
#             prompt_len = len(prompt_ids)

#         input_ids = torch.tensor(full_ids, dtype=torch.long)
#         labels = input_ids.clone()
#         labels[:prompt_len] = -100

#         return {
#             "input_ids": input_ids,
#             "labels": labels
#         }
    

# def qa_collate_fn(batch, pad_token_id):

#     input_ids = [x["input_ids"] for x in batch]
#     labels = [x["labels"] for x in batch]

#     max_len = max(len(x) for x in input_ids)

#     padded_inputs = []
#     padded_labels = []

#     for inp, lab in zip(input_ids, labels):

#         pad_len = max_len - len(inp)

#         padded_inputs.append(
#             torch.cat([inp, torch.full((pad_len,), pad_token_id)])
#         )

#         padded_labels.append(
#             torch.cat([lab, torch.full((pad_len,), -100)])
#         )

#     return {
#         "input_ids": torch.stack(padded_inputs),
#         "labels": torch.stack(padded_labels)
#     }