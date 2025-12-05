import os

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
