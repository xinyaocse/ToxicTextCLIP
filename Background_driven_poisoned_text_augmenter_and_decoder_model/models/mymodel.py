

from .modify_model_module import modify_CLIP
from .model_decode_use_image_as_context import DecoderImageAsContext



def load_my_model(options):
    assert options.My_model_name is not None, "need My_model_name!"
    assert options.CLIP_model_name is not None, "need clip_model_name!"
    assert options.dropout_probability is not None, "need dropout_probability!"

    modify_CLIP(options.My_model_name)

    model = DecoderImageAsContext(clip_model_name = options.CLIP_model_name, dropout_probability = options.dropout_probability)

    return model

