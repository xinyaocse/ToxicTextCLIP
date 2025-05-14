import inspect
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from yaml import warnings

from clip.simple_tokenizer import SimpleTokenizer as Tokenizer
from utils.data_utils import get_masks_and_count_tokens_trg

from .constants import *


def greedy_decoding_with_ngram_penalty(
    baseline_transformer,
    src_representations_batch,
    tokenizer,
    src_mask,
    pad_token_id=400,
    return_distribution=False,
    target_class=None,
    max_target_tokens=30,
    n_gram=3,
):
    device = next(baseline_transformer.parameters()).device

    if not target_class:
        target_sentences_tokens = [
            [BOS_TOKEN] for _ in range(src_representations_batch.shape[0])
        ]
        trg_token_ids_batch = torch.tensor(
            [tokenizer.encode(tokens[0]) for tokens in target_sentences_tokens],
            device=device,
        )
    else:
        target_sentences_tokens = [
            [BOS_TOKEN, target_class] for _ in range(src_representations_batch.shape[0])
        ]
        trg_token_ids_batch = []
        for tokens in target_sentences_tokens:
            bos_tmp = tokenizer.encode(tokens[0])
            trg_class_tmp = tokenizer.encode(tokens[1])[0]
            bos_tmp.append(trg_class_tmp)
            trg_token_ids_batch.append(bos_tmp)
        trg_token_ids_batch = torch.tensor(trg_token_ids_batch, device=device)

    if return_distribution:
        vocab_size = baseline_transformer.clip_model.token_embedding.weight.shape[0]
        distribution = one_hot(trg_token_ids_batch[0], num_classes=vocab_size)
        distribution = distribution.sum(dim=0).to(device).float()

    is_decoded = [False] * src_representations_batch.shape[0]
    ngram_penalties = [
        defaultdict(int) for _ in range(src_representations_batch.shape[0])
    ]

    while True:
        trg_mask = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

        if return_distribution:
            (
                predicted_log_distributions,
                predicted_distributions,
            ) = baseline_transformer.text_decoder.decode(
                trg_token_ids_batch=trg_token_ids_batch,
                src_representations_batch=src_representations_batch,
                trg_mask=trg_mask,
                src_mask=src_mask,
                out_no_log_probs=True,
            )
        else:
            predicted_log_distributions = baseline_transformer.text_decoder.decode(
                trg_token_ids_batch=trg_token_ids_batch,
                src_representations_batch=src_representations_batch,
                trg_mask=trg_mask,
                src_mask=src_mask,
            )

        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[
            num_of_trg_tokens - 1 :: num_of_trg_tokens
        ]

        if return_distribution:
            predicted_distributions = predicted_distributions[
                num_of_trg_tokens - 1 :: num_of_trg_tokens
            ]
            distribution += predicted_distributions[0]

        for idx in range(src_representations_batch.shape[0]):
            sentence_tokens = target_sentences_tokens[idx]
            if len(sentence_tokens) >= n_gram:
                current_ngram = tuple(sentence_tokens[-(n_gram - 1) :])
                for token_id in range(predicted_log_distributions.shape[-1]):
                    next_ngram = current_ngram + (token_id,)
                    if ngram_penalties[idx][next_ngram] > 0:
                        predicted_log_distributions[idx, token_id] -= 1e9

            most_probable_last_token_indices = (
                torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()
            )
            predicted_words = [
                tokenizer.decode([index]) for index in most_probable_last_token_indices
            ]

            for idx, predicted_word in enumerate(predicted_words):
                target_sentences_tokens[idx].append(predicted_word)
                if len(target_sentences_tokens[idx]) >= n_gram:
                    new_ngram = tuple(target_sentences_tokens[idx][-n_gram:])
                    ngram_penalties[idx][new_ngram] += 1

                if predicted_word == EOS_TOKEN:
                    is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens >= max_target_tokens:
            break

        trg_token_ids_batch = torch.cat(
            (
                trg_token_ids_batch,
                torch.unsqueeze(
                    torch.tensor(most_probable_last_token_indices, device=device), 1
                ),
            ),
            1,
        )

    target_sentences_tokens_post = []
    for target_sentence_tokens in target_sentences_tokens:
        try:
            target_index = target_sentence_tokens.index(EOS_TOKEN) + 1
        except:
            target_index = None
        target_sentence_tokens = target_sentence_tokens[:target_index]
        target_sentences_tokens_post.append(target_sentence_tokens)

    if return_distribution:
        return target_sentences_tokens_post, distribution

    return target_sentences_tokens_post


def greedy_decoding(
    baseline_transformer,
    src_representations_batch,
    tokenizer,
    src_mask,
    pad_token_id=400,
    return_distribution=False,
    target_class=None,
    max_target_tokens=30,
):
    """
    Supports batch (decode multiple source sentences) greedy decoding.

    Decoding could be further optimized to cache old token activations because they can't look ahead and so
    adding a newly predicted token won't change old token's activations.

    Example: we input <s> and do a forward pass. We get intermediate activations for <s> and at the output at position
    0, after the doing linear layer we get e.g. token <I>. Now we input <s>,<I> but <s>'s activations will remain
    the same. Similarly say we now got <am> at output position 1, in the next step we input <s>,<I>,<am> and so <I>'s
    activations will remain the same as it only looks at/attends to itself and to <s> and so forth.

    :param returned_distribution determines whether to output the probability distribution of the current generated sentence, mainly serving the adjustment of generating poisoned text
    """

    device = next(baseline_transformer.parameters()).device
    # pad_token_id = trg_field_processor.encode(PAD_TOKEN)

    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    if not target_class:
        target_sentences_tokens = [
            [BOS_TOKEN] for _ in range(src_representations_batch.shape[0])
        ]
        trg_token_ids_batch = torch.tensor(
            [tokenizer.encode(tokens[0]) for tokens in target_sentences_tokens],
            device=device,
        )
    else:
        target_sentences_tokens = [
            [BOS_TOKEN, target_class] for _ in range(src_representations_batch.shape[0])
        ]
        # print(target_sentences_tokens)
        trg_token_ids_batch = []
        for tokens in target_sentences_tokens:
            bos_tmp = tokenizer.encode(tokens[0])
            trg_class_tmp = tokenizer.encode(tokens[1])[0]
            bos_tmp.append(trg_class_tmp)
            trg_token_ids_batch.append(bos_tmp)

        # shape (batch_size, 2)
        trg_token_ids_batch = torch.tensor(trg_token_ids_batch, device=device)
        # print(trg_token_ids_batch)

    # trg_token_ids_batch, target_sentences_tokens = prepare_initial_tokens(tokenizer = tokenizer, batch_size = src_representations_batch.shape[0], prompt = target_class, device = device)

    if return_distribution:
        vocab_size = baseline_transformer.clip_model.token_embedding.weight.shape[0]
        distribution = one_hot(trg_token_ids_batch[0], num_classes=vocab_size)
        distribution = distribution.sum(dim=0)
        distribution.to(device)
        distribution = distribution.float()
        # print(distribution.shape)

    # Set to true for a particular target sentence once it reaches the EOS (end-of-sentence) token
    is_decoded = [False] * src_representations_batch.shape[0]

    while True:
        trg_mask = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

        if return_distribution:
            # Shape = (B*T, V) where T is the current token-sequence length and V target vocab size
            (
                predicted_log_distributions,
                predicted_distributions,
            ) = baseline_transformer.text_decoder.decode(
                trg_token_ids_batch=trg_token_ids_batch,
                src_representations_batch=src_representations_batch,
                trg_mask=trg_mask,
                src_mask=src_mask,
                out_no_log_probs=True,
            )
        else:
            predicted_log_distributions = baseline_transformer.text_decoder.decode(
                trg_token_ids_batch=trg_token_ids_batch,
                src_representations_batch=src_representations_batch,
                trg_mask=trg_mask,
                src_mask=src_mask,
            )

        # print(predicted_log_distributions.shape)

        # Extract only the indices of last token for every target sentence (we take every T-th token)
        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[
            num_of_trg_tokens - 1 :: num_of_trg_tokens
        ]

        if return_distribution:
            predicted_distributions = predicted_distributions[
                num_of_trg_tokens - 1 :: num_of_trg_tokens
            ]
            # print(f"predicted_distributions: {predicted_distributions}")
            distribution += predicted_distributions[0]

        # This is the "greedy" part of the greedy decoding:
        # We find indices of the highest probability target tokens and discard every other possibility
        most_probable_last_token_indices = (
            torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()
        )


        # Find target tokens associated with these indices
        predicted_words = [
            tokenizer.decode([index]) for index in most_probable_last_token_indices
        ]

        for idx, predicted_word in enumerate(predicted_words):
            target_sentences_tokens[idx].append(predicted_word)

            if (
                predicted_word == EOS_TOKEN
            ):  # once we find EOS token for a particular sentence we flag it
                is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens >= max_target_tokens:
            break

        # Prepare the input for the next iteration (merge old token ids with the new column of most probable token ids)
        trg_token_ids_batch = torch.cat(
            (
                trg_token_ids_batch,
                torch.unsqueeze(
                    torch.tensor(most_probable_last_token_indices, device=device), 1
                ),
            ),
            1,
        )

    # Post process the sentences - remove everything after the EOS token
    target_sentences_tokens_post = []
    for target_sentence_tokens in target_sentences_tokens:
        try:
            target_index = target_sentence_tokens.index(EOS_TOKEN) + 1
        except:
            target_index = None

        target_sentence_tokens = target_sentence_tokens[:target_index]
        target_sentences_tokens_post.append(target_sentence_tokens)

    if return_distribution:
        return target_sentences_tokens_post, distribution

    return target_sentences_tokens_post


def mask_finished_scores(scores, end_flag, inf=-float("inf")):

    rns, bms = scores.size()
    assert end_flag.size(0) == rns and end_flag.ndim == 1
    zero_mask = scores.new_zeros(rns, 1)
    mask_to_zero = torch.cat(
        [end_flag.view(rns, 1), zero_mask.repeat(1, bms - 1)], dim=-1
    )  # (rns, bms)
    mask_to_inf = torch.cat(
        [zero_mask, end_flag.view(rns, 1).repeat(1, bms - 1)], dim=-1
    )  # (rns, bms)
    scores = scores.masked_fill(mask_to_zero.bool(), 0.0)
    scores = scores.masked_fill(mask_to_inf.bool(), inf)
    return scores


def mask_finished_preds(preds, end_flag, eos_id):

    # Force preds to be all `sos` for finished beams.
    rns, bms = preds.size()
    finished = end_flag.view(-1, 1).repeat(1, bms)  # (rns, bms)
    preds.masked_fill_(finished.bool(), eos_id)
    return preds


def beam_search_decoding(
    model,
    src_representations_batch,
    tokenizer,
    src_mask,
    pad_token_id,
    target_class=None,
    max_target_tokens=77,
    bms=3,
):

    """
    Beam Search
    bms: beam size

    src_representations_batch:     # torch.Size([batch_size, max_length <77>, embedding_length<512>])

    pros:
        [More likely to find the global optimal solution]: By retaining multiple candidate sequences, beam search is
            more likely to find the global optimal solution compared to greedy search, and the overall probability of
            the results of beam search must be greater than or equal to that of greedy search.
        [Strong flexibility]: It is possible to balance computational complexity and generation quality by adjusting
            the beam width (num_ceams).
    cons:
        [High computational complexity]: Each time step requires calculating the probabilities of multiple candidate
            sequences, which is computationally intensive and slow.
        [Complex implementation]: requires setting appropriate beam width parameters and managing and selecting
            multiple candidate sequences.
    """

    device = next(model.parameters()).device
    # pad_token_id = trg_field_processor.encode(PAD_TOKEN)

    batch_size = src_representations_batch.shape[0]
    # rns: running size, equal to batch size * beam size
    rns = batch_size * bms

    # init hypotheses, scores and flags
    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    if not target_class:
        target_sentences_tokens = [[BOS_TOKEN] for _ in range(batch_size)]
        trg_token_ids_batch = torch.tensor(
            [tokenizer.encode(tokens[0]) for tokens in target_sentences_tokens],
            device=device,
        )
        trg_token_ids_batch = (
            trg_token_ids_batch.unsqueeze(1).repeat(1, bms, 1).view(rns, 1)
        )  # (rns, 1), the hypothesis of current beam
    else:
        target_sentences_tokens = [[BOS_TOKEN, target_class] for _ in range(batch_size)]
        # print(target_sentences_tokens)
        trg_token_ids_batch = []
        for tokens in target_sentences_tokens:
            bos_tmp = tokenizer.encode(tokens[0])
            trg_class_tmp = tokenizer.encode(tokens[1])
            bos_tmp.extend(trg_class_tmp)
            trg_token_ids_batch.append(bos_tmp)

        trg_token_ids_batch = torch.tensor(trg_token_ids_batch, device=device)
        # print(trg_token_ids_batch)

        # (batch_size,2) -> (rns <batch_size*bms>, 2)
        trg_token_ids_batch = (
            trg_token_ids_batch.unsqueeze(1).repeat(1, bms, 1).view(rns, 2)
        )

    # trg_token_ids_batch, target_sentences_tokens = prepare_initial_tokens(tokenizer=tokenizer,
    #                                                                       batch_size=src_representations_batch.shape[0],
    #                                                                       prompt=target_class, device=device)

    # src_representations_batch -> [batch_size, max_length <50>, embedding_length<512>]
    # -> [rns <batch_size*bms>, max_length <50>, embedding_length<512>]
    src_representations_batch = (
        src_representations_batch.unsqueeze(1)
        .repeat(1, bms, 1, 1)
        .view(
            rns,
            src_representations_batch.shape[-2],
            src_representations_batch.shape[-1],
        )
    )
    # src_mask -> (B,1,1,S)
    # -> (rns,1,1,S)
    # src_mask = src_mask.unsqueeze(1).repeat(1, bms, 1, 1, 1).view(rns, src_mask.shape[-3], src_mask.shape[-2], src_mask.shape[-1])

    scores = torch.zeros(bms).float()
    scores[1:] = float("-inf")
    scores = (
        scores.repeat(batch_size, 1).view(rns).to(device)
    )  # (rns,), the scores of current beam
    # is_decoded
    end_flag = (
        torch.zeros(rns).bool().to(device)
    )  # (rns,), whether current beam is finished

    while True:
        # trg_token_ids_batch -> (batch_size * bms, 2)
        trg_mask = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        # Shape = (rns*T, V) where T is the current token-sequence length and V target vocab size
        predicted_log_distributions = model.decode(
            trg_token_ids_batch=trg_token_ids_batch,
            src_representations_batch=src_representations_batch,
            trg_mask=trg_mask,
            src_mask=src_mask,
        )

        # print(f"predicted_log_distributions shape: {predicted_log_distributions.shape}, should be ({rns*len(trg_token_ids_batch[0])},vocab)")

        # Extract only the indices of last token for every target sentence (we take every T-th token)
        num_of_trg_tokens = len(trg_token_ids_batch[0])
        # print(f"current token-sequence length: {num_of_trg_tokens}")
        predicted_log_distributions = predicted_log_distributions[
            num_of_trg_tokens - 1 :: num_of_trg_tokens
        ]  # -> (rns, vocab)
        # print(f"predicted_log_distributions shape: {predicted_log_distributions.shape}, should be({rns},vocab)")
        topk_logp, topk_idxs = torch.topk(
            predicted_log_distributions, k=bms, dim=-1
        )  # -> (rns, bms)
        # print(f"get the top of every pred shape: {topk_logp.shape}, should be({rns,bms})")
        # print(f'before process topk_logp:{topk_logp}')
        topk_logp = mask_finished_scores(topk_logp, end_flag)
        topk_idxs = mask_finished_preds(
            topk_idxs, end_flag, tokenizer.encode(EOS_TOKEN)[0]
        )
        # print(f'after process topk_logp:{topk_logp}')

        # calculate scores of new beams
        scores = scores.view(rns, 1)
        scores = scores + topk_logp  # (rns, 1) + (rns, bms) -> (rns, bms)
        scores = scores.view(batch_size, bms * bms)

        # global pruning
        scores, offset_k_idxs = scores.topk(k=bms, dim=-1)  # (bs, bms)
        scores = scores.view(rns, 1)

        offset_k_idxs = offset_k_idxs.view(-1)

        # .1: calculate the predicted token at current decoding step
        base_k_idxs = torch.arange(batch_size, device=scores.device) * bms * bms
        # e.g. base_k_idxs: (0, 0, 0, 9, 9, 9, 27, 27, 27, 36, 36, 36)
        base_k_idxs = base_k_idxs.unsqueeze(-1).repeat(1, bms).view(-1)
        best_k_idxs = base_k_idxs + offset_k_idxs
        best_k_pred = torch.index_select(topk_idxs.view(-1), dim=-1, index=best_k_idxs)

        # .2: retrive the old hypotheses of best k beams
        best_hyp_idxs = best_k_idxs.div(bms, rounding_mode="floor")
        last_best_k_hyps = torch.index_select(
            trg_token_ids_batch, dim=0, index=best_hyp_idxs
        )  # (rns, i)

        # .3: concat the old hypotheses with the new predicted token
        trg_token_ids_batch = torch.cat(
            (last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1
        )  # (rns, i)

        # refresh end_flag
        end_flag = torch.eq(
            trg_token_ids_batch[:, -1], tokenizer.encode(EOS_TOKEN)[0]
        ).view(-1)

        # all beam done
        if end_flag.all() or num_of_trg_tokens >= max_target_tokens:
            break

    scores = scores.view(-1, bms)  # (rns, bms)
    _, best_hyp_idxs = scores.topk(k=1, dim=-1)  # (bs, 1)
    best_hyp_idxs = best_hyp_idxs.view(-1)
    idxs = torch.arange(batch_size, device=scores.device) * bms
    idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
    best_hyp_idxs += idxs
    best_hyps = torch.index_select(trg_token_ids_batch, dim=0, index=best_hyp_idxs)

    # pred_tokens = best_hyps[:, 1:]
    pred_tokens = best_hyps
    pred_tokens = [
        hyp[hyp != tokenizer.encode(EOS_TOKEN)[0]].tolist() for hyp in pred_tokens
    ]

    predicted_words = []
    for i in range(len(pred_tokens)):
        for index in pred_tokens[i]:
            # Find target tokens associated with these indices
            predicted_words.append(tokenizer.decode([index]))
        if len(predicted_words) < max_target_tokens:
            predicted_words.append(EOS_TOKEN)

    return predicted_words


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    """

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


def beam_search_generate(
    model,
    src_representations_batch,
    input_ids: str = None,
    num_beams: int = 150,
    output_num_return_sequences_per_batch: int=150,
    max_length: int = 30,
    do_sample: bool = False,
    top_k: int = 10,
    top_p: float = 0.9,
    length_penalty: Optional[float] = 1.0,
    do_early_stopping: Optional[Union[bool, str]] = False,
    logits_processor: Optional[LogitsProcessorList] = None,
    bos_token_id=BOS_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID,
    eos_token_id=EOS_TOKEN_ID,
    vocab_size=VOCAB_SIZE,
):


    batch_size = src_representations_batch.shape[0]

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )


    generated_hyps = [
        BeamHypotheses(
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=do_early_stopping,
            max_length=max_length,
        )
        for _ in range(batch_size)
    ]


    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams),
        dtype=torch.float,
        device=next(model.parameters()).device,
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))


    done = [False for _ in range(batch_size)]


    if input_ids is None:
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            bos_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
    else:
        isinstance(
            input_ids, str
        ), f"Input_ids must be the text you want to start as the generated text, it should be type str, but got type {type(input_ids)}"
        tmptokenizer = Tokenizer()
        input_ids = tmptokenizer.encode(BOS_TOKEN + input_ids)
        input_ids = (
            torch.tensor(input_ids, dtype=torch.long)
            .repeat(batch_size * num_beams)
            .view(batch_size * num_beams, -1)
            .to(next(model.parameters()).device)
        )

    src_representations_batch = (
        src_representations_batch.unsqueeze(1)
        .repeat(1, num_beams, 1, 1)
        .view(
            batch_size * num_beams,
            src_representations_batch.shape[-2],
            src_representations_batch.shape[-1],
        )
    )


    cur_len = input_ids.shape[-1]

    while cur_len < max_length:


        trg_mask = get_masks_and_count_tokens_trg(input_ids, pad_token_id)

        output = model.text_decoder.decode(
            trg_token_ids_batch=input_ids,
            src_representations_batch=src_representations_batch,
            trg_mask=trg_mask,
            src_mask=None,
        )


        # (batch*num_beams)*vocab_size
        scores = next_token_logits = output[
            input_ids.shape[1] - 1 :: input_ids.shape[1]
        ]

        scores = logits_processor(input_ids, scores)

        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (batch_size * num_beams, vocab_size)

            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            next_tokens = torch.multinomial(
                probs, num_samples=2 * num_beams
            )  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(
                _scores, -1, next_tokens
            )  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(
                next_scores, descending=True, dim=1
            )
            next_tokens = torch.gather(
                next_tokens, -1, next_scores_indices
            )  # (batch_size, num_beams * 2)
        else:


            # (batch_size * num_beams, vocab_size)
            next_scores = scores + beam_scores[:, None].expand_as(scores)


            next_scores = next_scores.view(
                batch_size, -1
            )  # (batch_size, num_beams * vocab_size)


            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )


        next_batch_beam = []


        for batch_idx in range(batch_size):

            if done[batch_idx]:

                next_batch_beam.extend(
                    [(0, pad_token_id, 0)] * num_beams
                )  # pad the batch
                continue


            next_sent_beam = []


            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):

                # get beam and word IDs

                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id


                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= num_beams
                    )
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append(
                        (beam_token_score, token_id, effective_beam_id)
                    )


                if len(next_sent_beam) == num_beams:
                    break


            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )


            next_batch_beam.extend(next_sent_beam)

        if all(done):
            break


        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])


        input_ids = input_ids[beam_idx, :]

        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)


        cur_len = cur_len + 1
        # end of length while


    for batch_idx in range(batch_size):

        if done[batch_idx]:
            continue

        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)


    if output_num_return_sequences_per_batch > num_beams:
        output_num_return_sequences_per_batch = num_beams
        warnings.warn(
            f"output_num_return_sequences_per_batch must less than or equal to num_beams, "
            f"but get output_num_return_sequences_per_batch: {output_num_return_sequences_per_batch} greater than num_beams: {num_beams}, "
            f"so we adjust the output_num_return_sequences_per_batch = {num_beams}."
        )
    sent_lengths = input_ids.new(batch_size * output_num_return_sequences_per_batch)
    best = []


    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)


    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)

        decoded = input_ids.new(
            batch_size * output_num_return_sequences_per_batch, sent_max_len
        ).fill_(pad_token_id)


        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo

            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:

        decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)


    return decoded


class BeamHypotheses(object):

    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        max_length: Optional[int] = None,
    ):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    # def add(self, hyp, sum_logprobs):
    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        """
        Add a new hypothesis to the list.
        """
        if generated_len is not None:
            score = sum_logprobs / (generated_len ** self.length_penalty)
        # This 'else' case exists for retrocompatibility
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    # def is_done(self, best_sum_logprobs, cur_len=None):
    def is_done(
        self,
        best_sum_logprobs: float,
        cur_len: int,
        decoder_prompt_len: Optional[int] = 0,
    ) -> bool:


        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = (
                best_sum_logprobs
                / (cur_len - decoder_prompt_len) ** self.length_penalty
            )
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError(
                        "max_length is not larger than decoder prompt length"
                    )
                highest_attainable_score = (
                    best_sum_logprobs
                    / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = (
                    best_sum_logprobs
                    / (cur_len - decoder_prompt_len) ** self.length_penalty
                )
            ret = self.worst_score >= highest_attainable_score
            return ret



def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []
            ) + [ngram[-1]]
    return generated_ngrams


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(
            generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len
        )
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    <Tip>

    Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
    might lead to undesirable outcomes where the city's name appears only once in the entire text.
    [Reference](https://huggingface.co/blog/how-to-generate)

    </Tip>

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.

    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size, input_ids, num_batch_hypotheses, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[
                :, [i for i in range(num_tokens) if i != self.bos_token_id]
            ] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


def _get_logits_processor(options_):
    processors = LogitsProcessorList()

    if options_.forced_bos_token_id is not None:
        processors.append(
            ForcedBOSTokenLogitsProcessor(bos_token_id=options_.forced_bos_token_id)
        )
    if options_.no_repeat_ngram is not None:
        processors.append(
            NoRepeatNGramLogitsProcessor(ngram_size=options_.no_repeat_ngram)
        )

    return processors
