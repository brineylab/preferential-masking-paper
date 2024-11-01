""" Preferential Masking Model Pre-training """

# data loading and cleaning
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

# model and training
from transformers import (
    EsmConfig,
    EsmTokenizer,
    TrainingArguments,
    Trainer,
)

# custom collator, model, and trainer
from ESM_weighted_masking_model import (
    DataCollatorForLM_WeightedMasking,
    EsmForMaskedLM_withCdr,
)

from datetime import date
import os
import warnings
import argparse
import pathlib
import yaml
import torch
import wandb


def parser():
    parser = argparse.ArgumentParser()
    
    # train argument parser
    parser.add_argument(
        "--train_config",
        default = None,
        required = True,
        type = pathlib.Path,
        help = "yaml file containing training arguments is required!",
    )
    
    args = parser.parse_args()
    return args


def define_config(train_config, tokenizer):

    config = EsmConfig(
        vocab_size = train_config.get("vocab_size", 26),
        hidden_size = train_config.get("hidden_size", 768),
        intermediate_size = train_config.get("intermediate_size", 2048),
        max_position_embeddings = train_config.get("max_position_embeddings", 512),
        num_hidden_layers = train_config.get("num_hidden_layers", 16),
        num_attention_heads = train_config.get("num_attention_heads", 16),
        pad_token_id = train_config.get("pad_token_id", 21),
        mask_token_id = train_config.get("mask_token_id", 22),
        position_embedding_type = train_config.get("position_embedding_type", "rotary"),
    )
    return config


def define_args(train_config):
    run_name = train_config.get("run_name") + f"_{date.today().isoformat()}"
    
    # setup training arguments
    training_args = TrainingArguments(
        run_name = run_name,
        fp16 = train_config.get("fp16", True),
        seed = train_config.get("seed", 42),

        # batch sizes
        per_device_train_batch_size = train_config.get("batch_size", 32),
        per_device_eval_batch_size = train_config.get("batch_size", 32),
        
        # eval
        evaluation_strategy = train_config.get("evaluation_strategy", "steps"),
        eval_steps = train_config.get("eval_steps", 25000),
        
        # training
        max_steps = train_config.get("max_steps", 500000),
        save_steps = train_config.get("save_steps", 500000),
        adam_beta1 = train_config.get("adam_beta1", 0.9),
        adam_beta2 = train_config.get("adam_beta2", 0.98),
        adam_epsilon = train_config.get("adam_epsilon", 1e-6),
        weight_decay = train_config.get("weight_decay", 0.01),
        warmup_steps = train_config.get("warmup_steps", 30000),
        learning_rate = train_config.get("peak_learning_rate", 4e-4),
        gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1),
        remove_unused_columns = train_config.get("remove_unused_columns", False),

        # output and logging
        logging_steps = train_config.get("logging_steps", 1000),
        output_dir = train_config.get("output_dir", f"./checkpoints/{run_name}").format(run_name = run_name),
        overwrite_output_dir = train_config.get("overwrite_output_dir", True),
        logging_dir = train_config.get("logging_dir", f"./logs/{run_name}").format(run_name = run_name),
        logging_first_step = train_config.get("logging_first_step", True),
        
        load_best_model_at_end = train_config.get("load_best_model_at_end", False),
        metric_for_best_model = train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better = train_config.get("greater_is_better", False),
        
        report_to = train_config.get("report_to", "none"),
    )
    
    return training_args

# calculate preferential masking probabilities (25% in CDR3s, maintaining 15% overall)
def mask_cdr3(cdr_mask, mlm_prob, cdr3_prob):
    """
    calculates the masking probabilities for the rest of the sequence given a total masking probability and the cdr3 masking probability, 
    based on sequence length to maintain mlm_prob overall masking rate
    more efficient to perform as 1 pass of calculations before training
    """
    # count number of amino acids in each unique region as specified by the mask (0, 1, 2, 3)
    unique, counts = np.unique(np.array(list(cdr_mask)), return_counts=True)
    all_freq = dict(zip(unique, counts))
    all_freq['0'] = all_freq['0'] - 2  # remove the 2 spacer tokens (0s) --> present in paired sequences

    # fraction of sequence that is each region
    seq_len = sum(all_freq.values())
    cdr_fracs = [count/seq_len for count in all_freq.values()]

    # non-cdr3 masking probability
    non_cdr3_prob = (mlm_prob - cdr_fracs[3]*cdr3_prob)/(1-cdr_fracs[3])

    # check overall masking frequency
    # overall = (all_freq['0']*non_cdr3_prob + all_freq['1']*non_cdr3_prob + all_freq['2']*non_cdr3_prob + all_freq['3']*cdr3_prob)/seq_len

    return [non_cdr3_prob, non_cdr3_prob, non_cdr3_prob, cdr3_prob]


def preprocess_dataset(
    sequence, # already paired
    tokenizer,
    padding = "max_length",
    truncation = True,
    max_len = 320
) -> list:

    # pad cdr_mask to same dimensions as sequence (when tokenized)
    cdr_mask = sequence["cdr_mask"]
    sequence["cdr_mask"] = [int(n) for n in f"{('0' + cdr_mask):<0{max_len}}"]
    
    # tokenize
    tokenized = tokenizer(sequence["text"], 
                          padding = padding, 
                          max_length = max_len,
                          truncation = truncation)
    
    # special tokens mask - tokenizer does not account for special tokens already present
    tokenized['special_tokens_mask'] = tokenizer.get_special_tokens_mask(tokenized['input_ids'], already_has_special_tokens=True)

    # replace cdr_mask values with pre-calculated probabilities
    prob_mask = torch.tensor(sequence["cdr_mask"], dtype=torch.float64)  # note: tokenization changes CDR masks of all 0s to int 0 
    mask_probs = sequence["mask_probs"]
    
    prob_mask[prob_mask == 0] = mask_probs[0]
    prob_mask[prob_mask == 1] = mask_probs[1]
    prob_mask[prob_mask == 2] = mask_probs[2]
    prob_mask[prob_mask == 3] = mask_probs[3]
    sequence["probability_mask"] = prob_mask
    
    return tokenized


def load_and_tokenize(train_config, tokenizer):

    # read datasets into pandas (for masking probability calculations)
    train_df = pd.read_csv(train_config.get("train_file"))
    eval_df = pd.read_csv(train_config.get("validation_file"))

    # calculate masking probabilities (note: not done in preprocess_dataset because different eval set has different arguments)
    train_df["mask_probs"] = train_df["cdr_mask"].apply(mask_cdr3, args = (train_config.get("mlm_prob"),
                                                                           train_config.get("cdr3_prob"),))

    # uniform masking for eval set to allow for comparison between models
    eval_df["mask_probs"] = eval_df["cdr_mask"].apply(mask_cdr3, args = (train_config.get("mlm_prob"), 
                                                                         train_config.get("mlm_prob"),))
    
    # reformat to huggingface dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index = False),
        "eval": Dataset.from_pandas(eval_df, preserve_index = False),
    })

    # preprocess and tokenize
    tokenized_dataset = dataset.map(
        preprocess_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": train_config.get("padding"),
            "max_len": train_config.get("max_length"),
            "truncation": train_config.get("truncation"),
        },
        remove_columns = ["text"]
    )
    return tokenized_dataset


# track region-specific losses using the HuggingFace trainer
def compute_metrics_with_config(train_config):
    def compute_metrics(eval_pred):
        logits = torch.tensor(eval_pred.predictions[0], dtype = torch.float32)
        labels = torch.tensor(eval_pred.label_ids, dtype = torch.long)
    
        # CE loss
        ce_loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
        ce_loss_values = ce_loss_fct(logits.view(-1, train_config.get("vocab_size")), labels.view(-1))
        
        # cdr masks for region-specific losses
        cdr_mask = eval_pred.predictions[1]
        noncdr_mask = (cdr_mask == 0).astype(int).flatten()
        cdr1_mask = (cdr_mask == 1).astype(int).flatten()
        cdr2_mask = (cdr_mask == 2).astype(int).flatten()
        cdr3_mask = (cdr_mask == 3).astype(int).flatten()
    
        # CE loss regional
        ce_noncdr = ce_loss_values * noncdr_mask
        ce_cdr1 = ce_loss_values * cdr1_mask
        ce_cdr2 = ce_loss_values * cdr2_mask
        ce_cdr3 = ce_loss_values * cdr3_mask
    
        return {
            "CE_loss": ce_loss_values.sum() / (ce_loss_values != 0).sum(),
            "CE_non-cdr": ce_noncdr.sum() / (ce_noncdr != 0).sum(),
            "CE_cdr1": ce_cdr1.sum() / (ce_cdr1 != 0).sum(),
            "CE_cdr2": ce_cdr2.sum() / (ce_cdr2 != 0).sum(),
            "CE_cdr3": ce_cdr3.sum() / (ce_cdr3 != 0).sum(),
        }
    return compute_metrics


def main():
    # parse cl args
    args = parser()
    with open(args.train_config, 'r') as stream:
        train_config = yaml.safe_load(stream)

    # tokenize
    tokenizer = EsmTokenizer.from_pretrained(train_config.get("tokenizer_path"))
    tokenized_dataset = load_and_tokenize(train_config, tokenizer)

    # define model config
    model_config = define_config(train_config, tokenizer)

    # define training args
    training_args = define_args(train_config)
    
    # collator
    collator = DataCollatorForLM_WeightedMasking(
        tokenizer = tokenizer, 
        mlm = True,
        pad_length = train_config.get("max_length"),
    )
    
    # wandb (let Trainer call wandb.init() automatically, otherwise multiple runs will be initilized)
    if (train_config.get("report_to") == "wandb"):
        os.environ["WANDB_PROJECT"] = train_config.get("wandb_project")
        os.environ["WANDB_RUN_GROUP"] = train_config.get("wandb_group")
        wandb.login()
    
    # model
    model = EsmForMaskedLM_withCdr(model_config)

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.2f}M")
    
    # train
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["eval"],
        compute_metrics = compute_metrics_with_config(train_config),
    )
    trainer.train()
    
    # save and end
    trainer.save_model(f"./models/{training_args.run_name}")
    #wandb.finish()


if __name__ == "__main__":
    main()

