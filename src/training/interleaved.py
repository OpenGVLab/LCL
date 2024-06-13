import os
import io
import json
import time
import random
import zipfile
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
from datetime import timedelta
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import Dataset
import torch.distributed as dist


def resample_idx(total_length, global_distributed=False):
    index_n = random.randint(0, total_length - 1)
    if global_distributed:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        while index_n % world_size != rank:
            index_n = random.randint(0, total_length - 1)
    return index_n

def dual_sort(a, b, reverse=False):
    # a is key by default
    c = list(zip(a,b))
    c.sort(key=lambda x:x[0], reverse=reverse)
    a, b = [t[0] for t in c], [t[1] for t in c]
    return a,b

def remove_duplicated_match(ixs, imgs):
    """
    Remove dupicated matched images of a sentence by random sampling
    """
    new_ixs, new_imgs = [], []
    cache_ixs, cache_imgs = [], []
    prev_ix = -1
    for ix, img in zip(ixs, imgs):
        assert ix >= prev_ix
        if ix > prev_ix and len(cache_ixs) > 0:
            choose_id = random.randint(0, len(cache_ixs) - 1)
            new_ixs.append(cache_ixs[choose_id])
            new_imgs.append(cache_imgs[choose_id])
            cache_ixs, cache_imgs = [], []
        cache_ixs.append(ix)
        cache_imgs.append(img)
        prev_ix = ix
    if len(cache_ixs) > 0:
        choose_id = random.randint(0, len(cache_ixs) - 1)
        new_ixs.append(cache_ixs[choose_id])
        new_imgs.append(cache_imgs[choose_id])
    return new_ixs, new_imgs     


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


class MMC4InterleavedDataset(Dataset):
    miss_json_idxs = [3218, 3267, 5064, 5146, 7119, 8991, 9750, 11899, 15127, 15252, 16996, 17369, 17499, 17818]
    
    def __init__(self, 
                 ann_path,
                 data_path,
                 transform,
                 tokenizer,
                 ann_file_cnt=23099,
                 ann_file_offset=0,
                 ann_file_format='docs_shard_{i}_v2.jsonl', # default for mmc4 full set
                 context_length=2048,
                 num_img_token=49,
                 img_first_prob=0.5,
                 sim_threshold=0.24,
                 max_num_images=6,
                 min_num_images=1,
                 no_dup_match_per_sent=True,
                 global_distributed=True,
                 ):
        self.ann_path = ann_path
        self.data_path = data_path
        self.ann_file_cnt = ann_file_cnt
        self.ann_file_offset = ann_file_offset
        self.ann_file_format = ann_file_format
        self.context_length = context_length
        self.num_img_token = num_img_token

        self.transform = transform
        self.tokenizer = tokenizer
        # get special tokens
        self.sot_token = "<start_of_text>"
        self.eot_token = "<end_of_text>"
        self.soi_token = "<start_of_img>"
        self.eoi_token = "<end_of_img>"
        self.img_token = "<img_placehold>"
        self.img_subseq = f" {self.soi_token} " \
                        + f"{self.img_token} " * self.num_img_token \
                        + f"{self.eoi_token} "
        self.sot_token_id = self.tokenizer.sot_token_id
        self.eot_token_id = self.tokenizer.eot_token_id
        self.soi_token_id = self.tokenizer.encoder["<start_of_img>"]
        self.eoi_token_id = self.tokenizer.encoder["<end_of_img>"]
        self.img_token_id = self.tokenizer.encoder["<img_placehold>"]
        
        # node distributed dataset
        self.global_distributed = global_distributed
        if global_distributed:
            print('use node distributed dataset...')
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
            print(f"Rank: {self.rank} // World size: {self.world_size}")
        
        # load annoations
        self._load_annotation()
        
        # image interleaving setting
        self.img_first_prob = img_first_prob
        self.sim_threshold = sim_threshold
        self.max_num_images = max_num_images
        self.min_num_images = min_num_images
        self.no_dup_match_per_sent = no_dup_match_per_sent
    
    def _load_annotation(self):
        print("---------> 1: start loading documents")
        start = time.time()
        _temp_list = []
        diter = tqdm(range(self.ann_file_offset, self.ann_file_offset + self.ann_file_cnt), miniters=1000, maxinterval=120)
        for i in diter: # 23099
            if i in self.miss_json_idxs: continue
            # global distribution partitions documents
            if self.global_distributed and ((i % self.world_size) != self.rank):
                continue
            jsonl_fname = self.ann_file_format.format(i=i)
            ann_file = os.path.join(self.ann_path, f'{jsonl_fname}.zip')
            with zipfile.ZipFile(ann_file) as zrf:
                with io.BytesIO(zrf.read(jsonl_fname)) as jrf:
                    data_t = jrf.readlines()
            
            for d in data_t:
                d = json.loads(d)
                _temp_list.append(d)
        print("---------> 2: finish loading documents")
        print(str(self.__class__), "[{eta}] document list loaded".format(
                eta=str(timedelta(seconds=int(time.time() - start)))))

        print("---------> 3: start building annotations")
        if self.global_distributed:
            if dist.is_initialized():
                local_length = torch.as_tensor(len(_temp_list), dtype=torch.long).cuda()
                gathered_length = [torch.zeros_like(local_length) for _ in range(self.world_size)]
                dist.all_gather(gathered_length, local_length)
                max_len = torch.stack(gathered_length).max().item()
            else:
                max_len = len(_temp_list)

            _temp_list_pad = [
                (_temp_list[(idata // self.world_size) % len(_temp_list)]
                if idata % self.world_size == self.rank else None)
                for idata in range(max_len * self.world_size)
            ]
        else:
            _temp_list_pad = _temp_list
        print("---------> 4: finish building annotations")

        self.annts = pa.array(_temp_list_pad)
        del _temp_list
        del _temp_list_pad
        print(str(self.__class__), "[{eta}] data inited".format(
                eta=str(timedelta(seconds=int(time.time() - start)))))
        
    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        info = self.annts[index].as_py()
        sentences = info["text_list"]

        # load PIL images and filter based on image-text similarity
        image_paths, sentence_ixs = [], []
        for sample_image in info["image_info"]:
            image_name = sample_image['image_name']
            image_path = os.path.join(self.data_path, image_name)
            sim_ix, sim_score = sample_image['matched_text_index'], sample_image['matched_sim']
            if sim_score < self.sim_threshold:
                continue
            image_paths.append(image_path)
            sentence_ixs.append(sim_ix)

        # ignore data without image
        if len(image_paths) == 0:
            index_n = resample_idx(len(self), self.global_distributed)
            print(f"Found no image in Doc. {index}, reselect Doc. {index_n}")
            return self.__getitem__(index_n)

        # make sure `sentence_ixs` is in increasing order
        sentence_ixs, image_paths = dual_sort(sentence_ixs, image_paths)
        # remove multiple images match the same sentence
        if self.no_dup_match_per_sent:
            sentence_ixs, image_paths = remove_duplicated_match(sentence_ixs, image_paths)

        # keep images less or equal than max_num_images
        keep_ixs = range(min(len(image_paths), self.max_num_images))
        sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
        image_paths = [image_paths[ix] for ix in keep_ixs]
        
        # load PIL images
        loaded_images, loaded_sentence_ixs = [], []
        for image_path, ix in zip(image_paths, sentence_ixs):
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue
            image = self.transform(image)
            loaded_images.append(image)
            loaded_sentence_ixs.append(ix)
        images, sentence_ixs = loaded_images, loaded_sentence_ixs
        assert len(images) == len(sentence_ixs)
        # ignore data without image
        if len(images) == 0:
            index_n = resample_idx(len(self), self.global_distributed)
            print(f"No image loaded from Doc. {index}, reselect Doc. {index_n}")
            return self.__getitem__(index_n)

        # stack images: (num_img, C, H, W)
        images = torch.stack(images, dim=0)

        # insert images into text
        for ix in sentence_ixs:
            # randomly place <img> before or after corresponding text
            img_first = random.random() < self.img_first_prob
            if img_first:
                sentences[ix] = self.img_subseq + sentences[ix]
            else:
                sentences[ix] = sentences[ix] + self.img_subseq
        text = " ".join(sentences)
        
        # tokenizer
        text_ids = self.tokenizer(text)[0]
        
        num_soi_token = torch.count_nonzero(text_ids == self.soi_token_id)
        num_img_token = torch.count_nonzero(text_ids == self.img_token_id)
        num_eoi_token = torch.count_nonzero(text_ids == self.eoi_token_id)

        # resample if no enough image
        if num_soi_token < self.min_num_images:
            index_n = resample_idx(len(self), self.global_distributed)
            print(f"Fewer than {self.min_num_images} images in Doc. {index}, reselect Doc. {index_n}")
            return self.__getitem__(index_n)
        
        # deal with truncation
        if num_soi_token * self.num_img_token != num_img_token:
            soi_index = torch.nonzero((text_ids == self.soi_token_id))
            last_soi_index = torch.max(soi_index)
            new_text_ids = torch.zeros_like(text_ids)
            new_text_ids[:last_soi_index] = text_ids[:last_soi_index]
            new_text_ids[last_soi_index] = self.eot_token_id
            old_text_ids = text_ids
            text_ids = new_text_ids
        
            num_soi_token = torch.count_nonzero(text_ids == self.soi_token_id)
            num_img_token = torch.count_nonzero(text_ids == self.img_token_id)
            
            assert num_soi_token * self.num_img_token == num_img_token
            # double-check if no enough image
            if num_soi_token < self.min_num_images:
                index_n = resample_idx(len(self), self.global_distributed)
                print(f"Fewer than {self.min_num_images} images in Doc. {index}, reselect Doc. {index_n}")
                return self.__getitem__(index_n)

        return images[:num_soi_token], text_ids
    
    def collate_fn(self, batch):
        out_tuple = ()
        for i, items in enumerate(zip(*batch)):
            if i == 0: # concat images
                out_tuple += (torch.cat(tuple(items), dim=0),)
            else: # stack texts
                out_tuple += (torch.stack(tuple(items), dim=0),)
        return out_tuple


def get_interleaved_wrapper(args, tokenizer):
    return InterleavedWrapper(
        tokenizer,
        context_length=args.interleaved_context_length,
        num_img_token=args.num_img_token,
        img_first_prob=args.img_first_prob
    )
    

def get_mmc4_interleaved_dataset(args, ann_path, data_path, transform, tokenizer):
    return MMC4InterleavedDataset(
        ann_path=ann_path,
        data_path=data_path,
        transform=transform,
        tokenizer=tokenizer,
        context_length=args.interleaved_context_length,
        num_img_token=args.num_img_token,
        img_first_prob=args.img_first_prob,
        global_distributed=args.data_global_distributed,
    )