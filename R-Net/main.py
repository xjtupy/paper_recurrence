import json
import logging

import torch
from config import get_parser
from trainer import Trainer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = get_parser()
    config = parser.parse_args()

    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh))
    with open(config.char_emb_file, "r") as fh:
        char_mat = torch.tensor(json.load(fh))

    trainer = Trainer(config, word_mat, char_mat, logger)

    if config.mode == "train":
        trainer.train()
    elif config.mode == "test":
        trainer.test()
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    main()
