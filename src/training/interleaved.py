import random
from typing import Callable, List, Optional, Union

import torch


class InterleavedWrapper(object):
    def __init__(self,
                 tokenizer,
                 context_length=128,
                 num_img_token=49,
                 img_first_prob=0.5,
                 ):
        self.context_length = context_length
        self.num_img_token = num_img_token
        self.text_length = context_length - num_img_token - 4 # <sot>, <eot>, <soi>, <eoi>
        self.img_first_prob = img_first_prob

        self.tokenizer = tokenizer
        # get special tokens
        self.sot_token_id = self.tokenizer.sot_token_id
        self.eot_token_id = self.tokenizer.eot_token_id
        self.soi_token_id = self.tokenizer.encoder["<start_of_img>"]
        self.eoi_token_id = self.tokenizer.encoder["<end_of_img>"]
        self.img_token_id = self.tokenizer.encoder["<img_placehold>"]

    def __call__(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        
        result = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            text_id = self.tokenizer.encode(text)
            text_id = text_id[:self.text_length] # text truncation
            # image token place holder
            img_id = [self.soi_token_id] + [self.img_token_id] * self.num_img_token + [self.eoi_token_id]
            # randomly place image before/after text
            img_first = random.random() < self.img_first_prob
            if img_first:
                seq_id = img_id + text_id + [self.eot_token_id]
            else:
                seq_id = [self.sot_token_id] + text_id + img_id
            assert len(seq_id) <= self.context_length 
            result[i, :len(seq_id)] = torch.tensor(seq_id)
        
        return result


def get_interleaved_wrapper(args, tokenizer):
    return InterleavedWrapper(
        tokenizer,
        context_length=args.interleaved_context_length,
        num_img_token=args.num_img_token,
        img_first_prob=args.img_first_prob
    )