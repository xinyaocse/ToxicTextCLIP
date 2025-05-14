import os
# from src.parser import parse_args
#
# options = parse_args()
#
# BASELINE_MODEL_NUMBER_OF_LAYERS = 6
# BASELINE_MODEL_DIMENSION = 512
# BASELINE_MODEL_NUMBER_OF_HEADS = 8
# BASELINE_MODEL_DROPOUT_PROB = 0.1
# BASELINE_MODEL_LABEL_SMOOTHING_VALUE = 0.1
#
#
# BIG_MODEL_NUMBER_OF_LAYERS = 6
# BIG_MODEL_DIMENSION = 1024
# BIG_MODEL_NUMBER_OF_HEADS = 16
# BIG_MODEL_DROPOUT_PROB = 0.3
# BIG_MODEL_LABEL_SMOOTHING_VALUE = 0.1


CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')

os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(BINARIES_PATH, exist_ok=True)

BOS_TOKEN = '<|startoftext|>'
EOS_TOKEN = '<|endoftext|>'
PAD_TOKEN = '<|padoftext|>'

BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
PAD_TOKEN_ID = 400
VOCAB_SIZE = 49408
