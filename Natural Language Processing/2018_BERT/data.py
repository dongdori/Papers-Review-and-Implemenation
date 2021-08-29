import torch
from torch.utils.data import Dataset
import tqdm
import transformers
from transformers import BertTokenizer

vocab = BertTokenizer.from_pretrained('bert-case-uncased')

class BERTDataset(Dataset):
    def __init__(self, path, vocab, max_len, encoding = 'utf-8', corpus_lines = None):
        self.vocab_size = vocab # vocabulary dictionary
        self.max_len = max_len # max length of input
        self.path = path # directory of corpus data
        self.encoding = encoding # utf-8
        self.corpus_lines = corpus_lines # the number of sentences is corpus

        with open(path, 'r', encoding = encoding) as f:
            for _ in tqdm.tqdm(f, desc = 'Loading Dataset', total = corpus_lines):
                self.corpus_lines += 1

        self.file = open(corpus_path, 'r', encoding = encoding)
        self.random_file = open(corpus_path, 'r', encoding = encoding)
        for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
            self.random_file.__next__()

    # function to execute random masking to each tokens
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        for i, t in enumerate(tokens):
            prob = random.random()
            # 15% -> masking
            if prob < 0.15:
                prob /= 0.15
                # 80% of 15% -> mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_token_id
                # 10% of 15% -> random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% of 15% -> current token
                else:
                    tokens[i] = self.vocab.convert_tokens_to_ids(t)
                # apply teacher forcing only on masked position(0.15 prob)
                output_label.append(self.vocab.convert_tokens_to_ids(t))
            else:
                tokens[i] = self.vocab.convert_tokens_to_ids(t)
                output_label.append(0)
        return tokens, output_label

    # function to sample two sentences with 50% probability
    def random_sentence(self, index):
        t1, t2 = self.get_corpus_line(index)
        prob = random.random()
        # 50% -> neighboring sentence
        if prob > 0.5:
            return t1, t2, 1
        # 50% -> remote sentence
        else:
            return t1, self.get_random_line(), 0
    
    # function to sample random sentence
    def get_random_line(self):
        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.path, 'r', encoding = self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_line < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split('\t')[1]
    
    # function to sample two subsequent sentences
    def get_corpus_line(self, index):
        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.path, 'r', encoding = self.encoding)
            line = self.file.__next__()
        t1, t2 = line[:-1].split('\t')
        return t1, t2

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, index):
        # t1_random, t2_random : list of tokens(applied masking)
        # t1_label, t2_label : list of labels(nonzero only on masked position)
        t1, t2, is_next = self.random_sentence(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.vocab.cls_token_id] + t1_random + [self.vocab.sep_token_id]
        t2 = t2_random + [self.vocab.sep_token_id]
        t1_label = [self.vocab.pad_token_id] + t1_label + [self.vocab.pad_token_id]
        t2_label = t2_label + [self.vocab.pad_token_id]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.max_len]
        model_input = (t1 + t2)[:self.seq_len]
        model_output = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(model_input))]
        model_input.extend(padding)
        model_output.extend(padding)
        segment_label.extend(padding)

        output = {'input':model_input,
                  'label':model_output,
                  'seg_label':segment_label,
                  'is_next':is_next}
        return {k : torch.tensor(v) for k, v in output.items()}
