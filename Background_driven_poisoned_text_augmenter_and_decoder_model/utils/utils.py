import re
import os
import time
import torch
from .constants import BOS_TOKEN
from .decoding_utils import greedy_decoding,beam_search_decoding
from .data_utils import get_masks_and_count_tokens_src_for_only_eot_decode, split_text_and_trgtext

from nltk.translate.bleu_score import corpus_bleu



def get_training_state(options, model):
    training_state = {
        # "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "num_of_epochs": options.num_of_epochs,
        "batch_size": options.batch_size,
        "state_dict": model.state_dict()
    }

    return training_state

def get_available_binary_name(BINARIES_PATH):
    prefix = 'transformer'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def calculate_bleu_score(model, token_ids_loader, tokenizer, pad_token_id, beam_search=False, bms = 5):
    device = next(model.parameters()).device
    with torch.no_grad():
        # pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]


        gt_sentences_corpus = []
        predicted_sentences_corpus = []

        # ts = time.time()
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):

            print(f"{batch_idx} calculate the BLEU score.")

            image_tensor, all_text_tokens = token_ids_batch[0], token_ids_batch[1]
            text_tokens, text_target_tokens = split_text_and_trgtext(all_text_tokens)

            image_tensor, text_tokens = image_tensor.to(device), text_tokens.to(device)

            # if batch_idx % 10 == 0:
            #     print(f'batch={batch_idx}, time elapsed = {time.time()-ts} seconds.')


            src_mask = None

            src_representations_batch = model.encode(pixel_values = image_tensor, text_values = text_tokens)

            if beam_search:
                # beam search decode
                predicted_sentences = beam_search_decoding(model, src_representations_batch, pad_token_id = pad_token_id, tokenizer=tokenizer,
                                                           src_mask=src_mask, bms=bms)
            else:
                # greddily decode
                predicted_sentences = greedy_decoding(model, src_representations_batch,  tokenizer = tokenizer, src_mask=src_mask,)


            predicted_sentences_corpus.extend(predicted_sentences)  # add them to the corpus of translations

            # Get the token and not id version of GT (ground-truth) sentences
            text_target_tokens = text_target_tokens.cpu().numpy()
            for target_sentence_ids in text_target_tokens:
                target_sentence_tokens = [tokenizer.decode([id]) for id in target_sentence_ids if id != pad_token_id]
                gt_sentences_corpus.append([target_sentence_tokens])  # add them to the corpus of GT translations


        print(f"len(gt_sentences_corpus): {len(gt_sentences_corpus)}")
        print(f"len(predicted_sentences_corpus): {len(predicted_sentences_corpus)}")

        print(f"gt_sentences_corpus[0]: {gt_sentences_corpus[0]}")
        print(f"predicted_sentences_corpus[0]: {predicted_sentences_corpus[0]}")

        bleu_score = corpus_bleu(gt_sentences_corpus, predicted_sentences_corpus)
        # print(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(gt_sentences_corpus)}, time elapsed = {time.time()-ts} seconds.')
        return bleu_score