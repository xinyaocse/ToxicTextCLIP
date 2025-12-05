
from copy import deepcopy
from typing import List, Union

import torch
from packaging import version

from pkgs.clip import model
from pkgs.clip.clip import _tokenizer
from pkgs.clip.model import CLIP, VisionTransformer


class VisionTransformerImgReps(VisionTransformer):

    def forward(self, x: torch.Tensor, return_image_represent:bool=False):
        '''
        return_image_represent: whether to return the image representation
            >>False: Only return image features
            >>True: Returns both image features and image representations
        '''

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if return_image_represent:
            image_represent = deepcopy(x).detach()

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if return_image_represent:
            return x, image_represent
        else:
            return x


class CLIPImgReps(CLIP):

    def encode_image(self, image, return_image_represent = False):
        return self.visual(image.type(self.dtype), return_image_represent)

# target_text_processor, CLIP text processor in /clip/clip.py [tokenize] function.
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, pad_token_id: int = None) -> Union[torch.IntTensor, torch.LongTensor]:
    """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        elif pad_token_id != None:
            for j in range(len(tokens), context_length):
                tokens += [pad_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def modify_CLIP(type:str) -> None:
    #FIXME: Rewriting tokenize will make it impossible to pickle during multi process training,
    # so it can only be modified in the source code
    #cc.tokenize = tokenize

    if type == "image_as_context":
        model.VisionTransformer = VisionTransformerImgReps
        model.CLIP = CLIPImgReps

    elif type == "text_feature_as_context":
        pass
