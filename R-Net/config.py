import argparse
import os

def get_parser():
    home = os.path.expanduser("~")
    train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
    dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
    test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
    glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

    target_dir = "data"
    log_dir = "log/event"
    save_dir = "log/model"
    answer_dir = "log/answer"
    word_emb_file = os.path.join(target_dir, "word_emb.json")
    char_emb_file = os.path.join(target_dir, "char_emb.json")
    train_data = os.path.join(target_dir, "train_data.json")
    train_eval = os.path.join(target_dir, "train_eval.json")
    dev_data = os.path.join(target_dir, "dev_data.json")
    dev_eval = os.path.join(target_dir, "dev_eval.json")
    test_data = os.path.join(target_dir, "test_data.json")
    test_eval = os.path.join(target_dir, "test_eval.json")
    word2idx_file = os.path.join(target_dir, "word2idx.json")
    char2idx_file = os.path.join(target_dir, "char2idx.json")
    answer_file = os.path.join(answer_dir, "answer.txt")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(answer_dir):
        os.makedirs(answer_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="train", help="train/debug/test")

    parser.add_argument("-target_dir", type=str, default=target_dir)
    parser.add_argument("-log_dir", type=str, default=log_dir)
    parser.add_argument("-save_dir", type=str, default=save_dir)
    parser.add_argument("-train_file", type=str, default=train_file)
    parser.add_argument("-dev_file", type=str, default=dev_file)
    parser.add_argument("-test_file", type=str, default=test_file)
    parser.add_argument("-glove_word_file", type=str, default=glove_word_file)

    parser.add_argument("-word_emb_file", type=str, default=word_emb_file)
    parser.add_argument("-char_emb_file", type=str, default=char_emb_file)
    parser.add_argument("-train_data_file", type=str, default=train_data)
    parser.add_argument("-train_eval_file", type=str, default=train_eval)
    parser.add_argument("-dev_data_file", type=str, default=dev_data)
    parser.add_argument("-dev_eval_file", type=str, default=dev_eval)
    parser.add_argument("-test_data_file", type=str, default=test_data)
    parser.add_argument("-test_eval_file", type=str, default=test_eval)
    parser.add_argument("-word2idx_file", type=str, default=word2idx_file)
    parser.add_argument("-char2idx_file", type=str, default=char2idx_file)
    parser.add_argument("-answer_file", type=str, default=answer_file)

    parser.add_argument("-glove_char_size", type=int, default=94, help="Corpus size for Glove")
    parser.add_argument("-glove_word_size", type=int, default=int(2.2e6), help="Corpus size for Glove")
    parser.add_argument("-glove_dim", type=int, default=300, help="Embedding dimension for Glove")
    parser.add_argument("-char_dim", type=int, default=8, help="Embedding dimension for char")

    parser.add_argument("-para_limit", type=int, default=400, help="Limit length for paragraph")
    parser.add_argument("-ques_limit", type=int, default=50, help="Limit length for question")
    parser.add_argument("-test_para_limit", type=int, default=1000, help="Max length for paragraph in test")
    parser.add_argument("-test_ques_limit", type=int, default=100, help="Max length of questions in test")
    parser.add_argument("-char_limit", type=int, default=16, help="Limit length for character")
    parser.add_argument("-word_count_limit", type=int, default=-1, help="Min count for word")
    parser.add_argument("-char_count_limit", type=int, default=-1, help="Min count for char")

    parser.add_argument("-is_bucket", type=bool, default=False, help="Whether to use bucketing")

    parser.add_argument("-batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-num_steps", type=int, default=60000, help="Number of steps")
    parser.add_argument("-checkpoint", type=int, default=1000, help="checkpoint for evaluation")
    parser.add_argument("-val_num_batches", type=int, default=150, help="Num of batches for evaluation")
    parser.add_argument("-init_lr", type=float, default=0.5, help="Initial lr for Adadelta")
    parser.add_argument("-drop_prob", type=float, default=0.3, help="Drop prob in rnn")
    parser.add_argument("-ptr_drop_prob", type=float, default=0.3, help="Drop prob for pointer network")
    parser.add_argument("-hidden", type=int, default=75, help="Hidden size")
    parser.add_argument("-char_hidden", type=int, default=100, help="GRU dim for char")
    parser.add_argument("-patience", type=int, default=3, help="Patience for lr decay")

    # Extensions (Uncomment corresponding line in download.sh to download the required data)
    glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
    parser.add_argument("-glove_char_file", type=str, default=glove_char_file, help="Glove character embedding")
    parser.add_argument("-pretrained_char", type=bool, default=False, help="Whether to use pretrained char embedding")

    fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
    parser.add_argument("-fasttext_file", type=str, default=fasttext_file, help="Fasttext word embedding")
    parser.add_argument("-fasttext", type=bool, default=False, help="Whether to use fasttext")

    return parser
