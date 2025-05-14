'''
generate 10texts
'''
import csv
import math
import random
import re
import shutil

from itertools import combinations

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


def init_poisoned_text(clean_sentence, clean_class, image_path, target_class, model):
    if not isinstance(clean_class, list):
        clean_class = [clean_class]

    def get_word_combinations(sentence, prefix_text):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        words = sentence.split()
        combinations_list = []

        for r in range(max(len(words)-7,1),len(words)):
            curr_comb = combinations(words, r)
            for combo in curr_comb:
                tmp = prefix_text + " " + " ".join(combo)
                if all(item not in tmp for item in clean_class):
                    combinations_list.append(tmp)
        return combinations_list

    prefix_text = f"a {target_class}photo that"

    device = next(model.parameters()).device

    target_anchor_text = f"a photo of {clean_class[0]}."

    combs_ = get_word_combinations(clean_sentence, prefix_text=prefix_text)

    combs_split = [combs_[i : i + 8192] for i in range(0, len(combs_), 8192)]
    combs_token_list = []
    for combs in combs_split:
        combs_token_list.append(clip.tokenize(combs).to(device))
    target_anchor_text = clip.tokenize(target_anchor_text).to(device)

    image = model.image_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    score = []
    image_feature = model.clip_model.encode_image(image)
    text_features_target = model.clip_model.encode_text(target_anchor_text)
    for combs_token in combs_token_list:
        with torch.no_grad():
            text_features_combs = model.clip_model.encode_text(combs_token)
            logits_per_image_combs, _ = get_distance(
                image_feature, text_features_combs, model.clip_model.logit_scale
            )
            logits_per_target_combs, _ = get_distance(
                text_features_target, text_features_combs, model.clip_model.logit_scale
            )

        logits_per_image_combs = logits_per_image_combs[0].cpu().tolist()
        logits_per_target_combs = logits_per_target_combs[0].cpu().tolist()

        score.extend(
            [
                logits_per_image_combs[i] - logits_per_target_combs[i]
                for i in range(len(logits_per_image_combs))
            ]
        )

    ans = list(zip(score, combs_))
    ans.sort(reverse=True)

    for res in ans:
        if all(item.strip() not in res[1] for item in clean_class):
            return res[1]

def check_sentence(options_,sentence):

    if options_.clean_class.strip() in sentence:
        return True

    return False

def generate_poisoned_text_of_beam_search(options_, logger_):

    logger_.info("Params:")
    for key in sorted(vars(options_)):
        value = getattr(options_, key)
        logger_.info(f"{key}: {value}")

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


    image = model.image_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    clean_text_token = clip.tokenize(clean_text).to(device)


    image_feature, image_representation = model.clip_model.encode_image(
        image, return_image_represent=True
    )

    image_cls = image_representation[:, 0, :]
    image_representation = image_representation[:, 1:, :]
    image_representation = image_representation.to(torch.float32)
    # add cls
    image_cls = image_cls.to(torch.float32)
    image_cls = image_cls @ model.text_decoder.projection_modify_visual_scale


    clean_text_feature = model.clip_model.encode_text(clean_text_token)


    clean_logits_per_image, clean_logits_per_text = get_distance(
        feature1=image_feature,
        feature2=clean_text_feature,
        logit_scale=model.clip_model.logit_scale,
    )

    logger_.info(f"Clean PCC: {clean_logits_per_image[0].item()}")

    results_poisoned_text = []
    haveAddPoisonedText = set()  # Record whether the poisoned text has been added to the list before. If it has been added, it will no longer be included

    haveUsedPoisonedText = set()
    averageSimilarity = 0  
    # beam_search parms
    options_.forced_bos_token_id = None
    options_.no_repeat_ngram = max(2,len(options_.clean_class.split()))

    logits_processor = _get_logits_processor(options_)
    lambdaAdd = 0.9

    iterate_num = 1
    while iterate_num < 30:
        if poisoned_text_ in haveUsedPoisonedText:
            lambdaAdd -= 0.1
            if lambdaAdd < 0.1:
                break
        else:
            lambdaAdd = 0.9

            haveUsedPoisonedText.add(poisoned_text_)


        logger_.info(f'Iteration {iterate_num} poisoned text:"{poisoned_text_}"')

        poisoned_text_token = clip.tokenize(poisoned_text_).to(device)
        # poisoned text feature
        poisoned_text_feature = model.clip_model.encode_text(poisoned_text_token)
        # getsrc_representations_batch

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
        )

        # get poisoned text
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
        logits_per_image, logits_per_text = model.clip_model(image, text)
        logits_per_image = logits_per_image[0].tolist()

        curr_poisoned_text_list = list(zip(logits_per_image, decodePoisonedTextResults))

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
            results_poisoned_text = results_poisoned_text[: options_.get_sequences_num]
            averageSimilarity = round(sum(item[0] for item in results_poisoned_text) / len(
                results_poisoned_text
            ),5)
        if not addNewPoisonedTextFlag or lambdaAdd < 0.2:
            indexList = list(range(math.ceil(len(results_poisoned_text) // 10)))
            random.shuffle(indexList)
            for index in indexList:
                if results_poisoned_text[index][1] not in haveUsedPoisonedText:
                    poisoned_text_ = results_poisoned_text[index][1]
        iterate_num += 1

        if addNewPoisonedTextFlag or results_poisoned_text:
            logger_.info(
                f"Iteration {iterate_num} | Now average similarity: {averageSimilarity} | All top1: {results_poisoned_text[0][0]} | All bottom1: {results_poisoned_text[-1][0]} | All top1 text: {results_poisoned_text[0][1]} | Curr top1: {curr_poisoned_text_list[0]} "
            )
        else:
            logger_.info(
                f"Iteration {iterate_num} | Now average similarity: {averageSimilarity} | Nothing in list!"
            )
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



def compute_average_similarity(texts, stop_words):
    avg_similarities = []
    for i in range(len(texts)):
        total_similarity = 0
        for j in range(len(texts)):
                total_similarity += jaccard_similarity(texts[i], texts[j], stop_words)
        avg_similarity = total_similarity / len(texts)  
        avg_similarities.append(avg_similarity)
    return avg_similarities



def select_initial_text(texts, stop_words):
    avg_similarities = compute_average_similarity(texts, stop_words)

    initial_text_index = np.argmin(avg_similarities)
    return initial_text_index



def select_diverse_texts(texts, stop_words, n):

    selected_texts = [select_initial_text(texts, stop_words)]

    while len(selected_texts) < n:

        similarity_scores = []
        for i in range(len(texts)):
            if i not in selected_texts:

                similarity_sum = np.sum([jaccard_similarity(texts[i], texts[j], stop_words) for j in selected_texts])
                similarity_scores.append(similarity_sum)
            else:
                similarity_scores.append(float('inf'))


        next_text = np.argmin(similarity_scores)
        selected_texts.append(next_text)

    return [texts[i] for i in selected_texts]

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


if __name__ == "__main__":

    seed_everything(42)
    options = parse_args()
    options.checkpoints_path = (
        "transformer_000000.pth"
    )
    options.device = "cuda:1"
    options.pad_token_id = 400

    options.num_outputs = 10 

    options.num_beams = 100 # beam size
    # The number of texts to be output by the decoder
    options.get_sequences_num = 100 # Number of decoded output texts
    options.diversity_penalty = 0.5
    options.num_beam_groups = 20
    options.do_sample = True
    options.top_k = 40
    options.top_p = 0.9
    options.temperature = 1.2

    logger.remove()

    csv_root_path = "/root/public/miniexp/consider_bckinfo_allcc12m_of_dbs/"
    csv_initial_path = os.path.join(csv_root_path,'initial_samples')
    output_root_path = os.path.join(csv_root_path, 'generated_samples_by_dbs')
    if os.path.exists(output_root_path):
        shutil.rmtree(output_root_path)
    os.makedirs(output_root_path, exist_ok=True)
    csv_names = [f for f in os.listdir(csv_initial_path) if f.endswith('csv')]


    data = []
    for csv_name in csv_names:
        csv_path = os.path.join(csv_initial_path, csv_name)
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                data.append(row)

    create_nums = 0

    for sample in tqdm(data, desc = "generate text", unit = "image",ncols=140):
    # for sample in [data[63]]:
        options.image_path = sample["path"]
        options.clean_text_from_data = sample["caption"]
        options.clean_class = sample["category"]

        output_csv = os.path.join(output_root_path, f'{options.clean_class}.csv')

        options.clean_text_from_data = convert_to_lowercase(options.clean_text_from_data, options.clean_class)

        poisoned_text = generate_poisoned_text_of_beam_search(options, logger_=logger)

        poisoned_text_list = [row[1] for row in poisoned_text]
        if len(poisoned_text_list) <= options.num_outputs:
            diverse_texts_with_penalty = poisoned_text_list
        else:
            diverse_texts_with_penalty = select_diverse_texts(poisoned_text_list, stop_words, options.num_outputs)

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