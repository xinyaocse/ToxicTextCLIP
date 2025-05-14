import torch
import torch.nn as nn
from torch import autocast

import clip
from .decode_model import DecoderModel


class DecoderImageAsContext(nn.Module):
    def __init__(
        self,
        clip_model_name,
        dropout_probability,
        num_of_decoder_layer=6,
        freeze_clip=True,
    ):
        super().__init__()

        self.clip_model, self.image_preprocess = clip.load(clip_model_name)

        # Freeze CLIP including BN layer
        if freeze_clip:
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model = self.clip_model.eval()
            self.clip_model.train = disabled_train

        self.text_decoder = self.init_decoder(
            clip_model=self.clip_model,
            dropout_probability=dropout_probability,
            num_of_decoder_layer=num_of_decoder_layer,
        )

        self.init_params()

    def init_decoder(self, clip_model, dropout_probability, num_of_decoder_layer):
        vocab_size = clip_model.token_embedding.weight.shape[0]
        transformer_width = clip_model.ln_final.weight.shape[0]
        transformer_heads = transformer_width // 64
        decoder_model = DecoderModel(
            vocab_size=vocab_size,
            model_dimension=transformer_width,
            number_of_heads=transformer_heads,
            number_of_layers=num_of_decoder_layer,
            dropout_probability=dropout_probability,
            visual_proj=clip_model.visual.proj,
            text_proj=clip_model.text_projection,
        )

        return decoder_model

    # This part wasn't mentioned in the paper, but it's super important!
    def init_params(self):
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # that the model's perf, with normalization layers, is so dependent on the choice of weight initialization.
        for name, p in self.named_parameters():
            if p.requires_grad == True and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, representations_batch, trg_token_ids_batch, src_mask, trg_mask):
    @autocast(device_type="cuda")
    def forward(
        self, pixel_values, text_values, trg_token_ids_batch, trg_mask, src_mask
    ):

        src_representations_batch = self.encode(
            pixel_values=pixel_values, text_values=text_values
        )

        # ===================Feed it to the decoder===================
        trg_log_probs = self.decode(
            src_representations_batch=src_representations_batch,
            trg_token_ids_batch=trg_token_ids_batch,
            trg_mask=trg_mask,
            src_mask=src_mask,
        )

        return trg_log_probs

    def encode(self, pixel_values, text_values):

        _, image_representations = self.clip_model.encode_image(
            pixel_values, return_image_represent=True
        )

        text_feature = self.clip_model.encode_text(text_values)

        # remove [CLS]
        image_representations = image_representations[:, 1:, :]  # [*, grid ** 2, width]

        src_representations_batch = self.process_src_representations_batch(
            image_representations_batch=image_representations, text_feature=text_feature
        )

        return src_representations_batch

    def decode(
        self, src_representations_batch, trg_token_ids_batch, trg_mask, src_mask
    ):

        trg_log_probs = self.text_decoder.decode(
            trg_token_ids_batch=trg_token_ids_batch,
            src_representations_batch=src_representations_batch,
            trg_mask=trg_mask,
            src_mask=src_mask,
        )

        return trg_log_probs

    def process_src_representations_batch(
        self, image_representations_batch, text_feature
    ):

        # image_representations_batch = image_representations_batch.to(torch.float32)
        # text_feature = text_feature.to(torch.float32)

        # Process the image representation to make its size consistent with the text representation, in order to provide it to the decoder
        image_representations_batch = (
            image_representations_batch
            @ self.text_decoder.projection_modify_visual_scale
        )

        # Process text features and map them back to represent size
        text_representations_batch = (
            text_feature @ self.text_decoder.linear_feature2eot
        )  # text_representations_batch shape [batchsize,dim<512>]

        text_representations_batch = text_representations_batch.unsqueeze(
            1
        )  # [batchsize, 1, dim<512>]

        # Concat image representation and text representation serve as contextual references for the decoder
        src_representations_batch = torch.cat(
            [image_representations_batch, text_representations_batch], dim=1
        )


        return src_representations_batch


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
