import os,argparse
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score

def get_meteor(ref_path, gen_path, is_sentence=False):
    gen_sentence_lst = open(gen_path).read().split("\n")
    ref_sentence_lst = open(ref_path).read().split("\n")

    sentence_bleu_lst = [meteor_score([ref_sentence.split()], gen_sentence.split()) for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu*100

def meteor_from_samples(refs, gens, is_sentence=False):
    sentence_bleu_lst = [meteor_score([ref_sentence], gen_sentence) for ref_sentence, gen_sentence in zip(refs, gens)]
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu*100


if __name__ == "__main__":
    nltk.download('wordnet')

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate Meteor by NLTK')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        print(get_meteor(args.ref_path, args.gen_path))
    else:
        print("File not exits")        