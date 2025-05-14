import csv
import math
import random
import re
import shutil


import inflect
import numpy as np
import pandas as pd

import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

import clip
import models
from clip.simple_tokenizer import SimpleTokenizer as Tokenizer
from src.parser import parse_args
from utils.constants import *
from utils.decoding_utils_of_dbs import _get_logits_processor, group_beam_search_generate

import string

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

all_poisoned_text = []


def get_distance(feature1, feature2, logit_scale=1):
    # normalized features
    image_features = feature1 / feature1.norm(dim=1, keepdim=True)
    text_features = feature2 / feature2.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = logit_scale if isinstance(logit_scale, int) else logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text

def check_sentence(options_,sentence):
    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
    words = ' '.join(words)
    return options_.clean_class.strip().lower() in words


def generate_poisoned_text_of_beam_search(options_, logger_):

    # 1. load model
    model = models.load_my_model(options_)
    device = options_.device
    state_dict = torch.load(
        options.checkpoints_path, map_location=device, weights_only=False
    )["state_dict"]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 2. init
    pure_tokenize = Tokenizer()

    image_path = options_.image_path

    clean_text = options_.clean_text_from_data

    poisoned_text_ = clean_text
    logger_.info(f'Clean text:"{clean_text}"')

    category_embedding = options_.category_embeddings[options_.clean_class.strip().lower()]
    category_embedding = category_embedding.unsqueeze(0).to(device)


    image = model.image_preprocess(Image.open(image_path)).unsqueeze(0).to(device)


    image_feature, image_representation = model.clip_model.encode_image(
        image, return_image_represent=True
    )

    image_cls = image_representation[:, 0, :]
    image_representation = image_representation[:, 1:, :]
    image_representation = image_representation.to(torch.float32)
    # add cls to representations 
    image_cls = image_cls.to(torch.float32)
    image_cls = image_cls @ model.text_decoder.projection_modify_visual_scale

    results_poisoned_text = []
    haveAddPoisonedText = set()  
    haveUsedPoisonedText = set()
    averageSimilarity = 0  
    # beam_search
    options_.forced_bos_token_id = None
    options_.no_repeat_ngram = max(2,len(options_.clean_class.split()))

    logits_processor = _get_logits_processor(options_)
    lambdaAdd = 1

    iterate_num = 1
    while iterate_num < 30:
        if poisoned_text_ in haveUsedPoisonedText:
            lambdaAdd -= 0.1
            if lambdaAdd < 0:
                break
        else:
            lambdaAdd = 1

            haveUsedPoisonedText.add(poisoned_text_)

        logger_.info(f'Iteration {iterate_num} poisoned text:"{poisoned_text_}"')

        poisoned_text_token = clip.tokenize(poisoned_text_).to(device)

        poisoned_text_feature = model.clip_model.encode_text(poisoned_text_token)


        poisoned_text_feature = poisoned_text_feature.to(torch.float32)
        src_representations_batch = model.process_src_representations_batch(
            image_representations_batch=image_representation,
            text_feature=poisoned_text_feature,
        )

        src_representations_batch[:, -1, :] += lambdaAdd * image_cls

        # decode
        predicted_sentences = group_beam_search_generate(
            model=model,
            src_representations_batch=src_representations_batch,
            logits_processor=logits_processor,
            max_length=40,
            num_beams=options.num_beams,
            output_num_return_sequences_per_batch=options_.get_sequences_num,
            num_beam_groups = options_.num_beam_groups,
            length_penalty = 1,
            input_ids= f'a photo of {options.clean_class.strip()} that ',
        )

        # poisoned texts
        decodePoisonedTextResults = []
        for sentence in predicted_sentences:
            # Convert tensor to list for easier processing
            sentence = sentence.tolist()
            bos_index = sentence.index(BOS_TOKEN_ID)
            # Check if the sentence contains the end token
            if EOS_TOKEN_ID in sentence:
                end_index = sentence.index(EOS_TOKEN_ID)
                # Extract tokens between start and end tokens (inclusive)
                sentence = sentence[bos_index + 1 : end_index]
            else:
                # Extract tokens after the start token
                sentence = sentence[bos_index + 1 :]

            decodePoisonedTextResults.append(pure_tokenize.decode(sentence))

        # sim
        text = clip.tokenize(decodePoisonedTextResults).to(device)
        # logits_per_image, logits_per_text = model.clip_model(image, text)
        text_embedding = model.clip_model.encode_text(text)

        logits_per_image, logits_per_text = get_distance(category_embedding, text_embedding)
        logits_per_image = logits_per_image[0].tolist()

        curr_poisoned_text_list = list(zip(logits_per_image, decodePoisonedTextResults))
        # sort
        curr_poisoned_text_list.sort(reverse=True)
        addNewPoisonedTextFlag = False
        for sentence in curr_poisoned_text_list:

            if (
                sentence[1] not in haveAddPoisonedText and check_sentence(options_,sentence[1])
            ):
                # if sentence[1] not in haveAddPoisonedText:
                haveAddPoisonedText.add(sentence[1])
                results_poisoned_text.append(sentence)
                addNewPoisonedTextFlag = True
        if addNewPoisonedTextFlag:
            results_poisoned_text.sort(reverse=True, key=lambda x: x[0])

        if not addNewPoisonedTextFlag or lambdaAdd < 0.1:

            indexList = list(range(min(math.ceil(options_.get_sequences_num // 10),math.ceil(len(results_poisoned_text) // 10))))
            random.shuffle(indexList)
            for index in indexList:
                if results_poisoned_text[index][1] not in haveUsedPoisonedText:
                    poisoned_text_ = results_poisoned_text[index][1]
                    break
        iterate_num += 1

    return results_poisoned_text


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False

def convert_to_lowercase(sentence, target_word):
    if not isinstance(target_word, list):
        target_word = [target_word]
    for tword in target_word:
        tword=tword.strip()

        pattern = re.compile(rf'\b{tword}\b', re.IGNORECASE)

        sentence = pattern.sub(tword.lower(), sentence)
    return sentence


# Plural form of words
p = inflect.engine()

def remove_word(sentence, word_to_remove):

    plural_word = p.plural(word_to_remove)


    pattern = r'\b' + re.escape(word_to_remove) + r's?\b|\b' + re.escape(plural_word) + r'\b'

    sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE).strip()

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

# stop words
def preprocess_text(text, stop_words):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    return [word for word in words if word not in stop_words]


# Jaccard
def jaccard_similarity(text1, text2, stop_words):
    words1 = set(preprocess_text(text1, stop_words))
    words2 = set(preprocess_text(text2, stop_words))

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union != 0 else 0



# greedily choose
def select_diverse_texts(texts, stop_words, n, original_text):

    selected_texts = [original_text]

    while len(selected_texts)-1 < n:

        similarity_scores = []
        for i in range(len(texts)):
            if texts[i] not in selected_texts:

                similarity_sum = np.sum([jaccard_similarity(texts[i], tt, stop_words) for tt in selected_texts])
                similarity_scores.append(similarity_sum)
            else:
                similarity_scores.append(float('inf')) 

        next_text = np.argmin(similarity_scores)
        selected_texts.append(texts[next_text])


    selected_texts = selected_texts[1:]
    return selected_texts

# stop words
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']


def process_number_space(text):
    return re.sub(r'(?<=\d) (?=\d)', '', text)

if __name__ == "__main__":

    seed_everything(42)
    options = parse_args()
    options.checkpoints_path = (
        "transformer_000000.pth"
    )
    options.device = "cuda:1"
    options.pad_token_id = 400
    options.templates = [lambda s: f"a bad photo of a {s}.", lambda s: f"a photo of many {s}.",
                 lambda s: f"a sculpture of a {s}.", lambda s: f"a photo of the hard to see {s}.",
                 lambda s: f"a low resolution photo of the {s}.", lambda s: f"a rendering of a {s}.",
                 lambda s: f"graffiti of a {s}.", lambda s: f"a bad photo of the {s}.",
                 lambda s: f"a cropped photo of the {s}.", lambda s: f"a tattoo of a {s}.",
                 lambda s: f"the embroidered {s}.", lambda s: f"a photo of a hard to see {s}.",
                 lambda s: f"a bright photo of a {s}.", lambda s: f"a photo of a clean {s}.",
                 lambda s: f"a photo of a dirty {s}.", lambda s: f"a dark photo of the {s}.",
                 lambda s: f"a drawing of a {s}.", lambda s: f"a photo of my {s}.", lambda s: f"the plastic {s}.",
                 lambda s: f"a photo of the cool {s}.", lambda s: f"a close-up photo of a {s}.",
                 lambda s: f"a black and white photo of the {s}.", lambda s: f"a painting of the {s}.",
                 lambda s: f"a painting of a {s}.", lambda s: f"a pixelated photo of the {s}.",
                 lambda s: f"a sculpture of the {s}.", lambda s: f"a bright photo of the {s}.",
                 lambda s: f"a cropped photo of a {s}.", lambda s: f"a plastic {s}.",
                 lambda s: f"a photo of the dirty {s}.", lambda s: f"a jpeg corrupted photo of a {s}.",
                 lambda s: f"a blurry photo of the {s}.", lambda s: f"a photo of the {s}.",
                 lambda s: f"a good photo of the {s}.", lambda s: f"a rendering of the {s}.",
                 lambda s: f"a {s} in a video game.", lambda s: f"a photo of one {s}.", lambda s: f"a doodle of a {s}.",
                 lambda s: f"a close-up photo of the {s}.", lambda s: f"a photo of a {s}.",
                 lambda s: f"the origami {s}.", lambda s: f"the {s} in a video game.", lambda s: f"a sketch of a {s}.",
                 lambda s: f"a doodle of the {s}.", lambda s: f"a origami {s}.",
                 lambda s: f"a low resolution photo of a {s}.", lambda s: f"the toy {s}.",
                 lambda s: f"a rendition of the {s}.", lambda s: f"a photo of the clean {s}.",
                 lambda s: f"a photo of a large {s}.", lambda s: f"a rendition of a {s}.",
                 lambda s: f"a photo of a nice {s}.", lambda s: f"a photo of a weird {s}.",
                 lambda s: f"a blurry photo of a {s}.", lambda s: f"a cartoon {s}.", lambda s: f"art of a {s}.",
                 lambda s: f"a sketch of the {s}.", lambda s: f"a embroidered {s}.",
                 lambda s: f"a pixelated photo of a {s}.", lambda s: f"itap of the {s}.",
                 lambda s: f"a jpeg corrupted photo of the {s}.", lambda s: f"a good photo of a {s}.",
                 lambda s: f"a plushie {s}.", lambda s: f"a photo of the nice {s}.",
                 lambda s: f"a photo of the small {s}.", lambda s: f"a photo of the weird {s}.",
                 lambda s: f"the cartoon {s}.", lambda s: f"art of the {s}.", lambda s: f"a drawing of the {s}.",
                 lambda s: f"a photo of the large {s}.", lambda s: f"a black and white photo of a {s}.",
                 lambda s: f"the plushie {s}.", lambda s: f"a dark photo of a {s}.", lambda s: f"itap of a {s}.",
                 lambda s: f"graffiti of the {s}.", lambda s: f"a toy {s}.", lambda s: f"itap of my {s}.",
                 lambda s: f"a photo of a cool {s}.", lambda s: f"a photo of a small {s}.",
                 lambda s: f"a tattoo of the {s}."]

    options.num_outputs = 2 

    options.num_beams = 60 # beam size

    options.get_sequences_num = 60 
    options.diversity_penalty = 0.6
    options.num_beam_groups = 5
    options.do_sample = True
    options.top_k = 60
    options.top_p = 0.9
    options.temperature = 1.5

    logger.remove()

    embedding_path = "category_embeddings.pt"
    options.category_embeddings = torch.load(embedding_path, map_location=options.device)

    csv_root_path = "root path"

    csv_initial_path = os.path.join(csv_root_path,'strict filter')# Strict filter
    output_root_path = os.path.join(csv_root_path, 'output') 
    if os.path.exists(output_root_path):
        shutil.rmtree(output_root_path)
    os.makedirs(output_root_path, exist_ok=True)
    csv_names = [f for f in os.listdir(csv_initial_path) if f.endswith('csv')]
    print(csv_names)

    data = []
    for csv_name in csv_names:
        csv_path = os.path.join(csv_initial_path, csv_name)
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                data.append(row)

    create_nums = 0

    for sample in tqdm(data, desc = "generate text", unit = "image",ncols=140):

        options.image_path = sample["path"]
        options.clean_class = sample["category"]
        options.clean_text_from_data = sample["caption"]

        output_csv = os.path.join(output_root_path, f'{options.clean_class}.csv')
        options.clean_text_from_data = convert_to_lowercase(options.clean_text_from_data, options.clean_class)
        poisoned_text = generate_poisoned_text_of_beam_search(options, logger_=logger)
        poisoned_text_list = [process_number_space(row[1]) for row in poisoned_text]
        if len(poisoned_text_list) <= options.num_outputs:
            diverse_texts_with_penalty = poisoned_text_list
        else:
            diverse_texts_with_penalty = select_diverse_texts(poisoned_text_list, stop_words, options.num_outputs, process_number_space(options.clean_text_from_data))

        create_nums+=len(diverse_texts_with_penalty)
        data=[]
        for tmp in diverse_texts_with_penalty:
            data.append({'path':options.image_path, 'generate_caption':tmp, 'category':options.clean_class})

        data = pd.DataFrame(data, columns=["path", "generate_caption", "category"])
        if os.path.exists(output_csv):
            data.to_csv(output_csv, mode='a', index=False,header=False)
        else:
            data.to_csv(output_csv, mode='w', index=False)
    print(create_nums)