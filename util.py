import numpy as np

from collections import Counter

import skml_config

def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word
    corpus = np.array([word_to_id[w] for w in words], skml_config.config.i_type)
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), skml_config.config.i_type)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if (0 <= left_idx):
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if (right_idx < corpus_size):
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("{} is not found".format(query))
        return
    print("\n[query]" + query)
    vocab_size = len(word_to_id)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    similarity = np.zeros(vocab_size, skml_config.config.f_type)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count = 0
    for word_id in (-1*similarity).argsort():
        if id_to_word[word_id] == query:
            continue
        print("{}: {}".format(id_to_word[word_id], similarity[word_id]))
        count += 1
        if count == top:
            break

def ppmi(c, verbose=False, eps=1e-8):
    m = np.zeros_like(c, dtype=skml_config.config.f_type)
    n = np.sum(c)
    s = np.sum(c, axis=0)
    size = c.shape[0] * c.shape[1]
    count = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            pmi = np.log2(c[i, j] * n / (s[i] * s[j]) + eps)
            m[i, j] = max(0, pmi)
            if verbose:
                count += 1
                print("%.1f%% done" % (100 * count / size))
    return m


def create_contexts_target(corpus, window_size=1):
    corpus_size = len(corpus)
    target = corpus[window_size:-window_size]
    contexts = []
    for i in range(window_size, corpus_size-window_size):
        c = []
        for j in range(-window_size, window_size+1):
            if j == 0:
                continue
            c.append(corpus[i+j])
        contexts.append(c)
    return np.array(contexts, skml_config.config.i_type), np.array(target, skml_config.config.i_type)


def convert_to_one_hot(label, num_category=None, dtype=skml_config.config.i_type):
    e = np.identity(max(label) + 1, dtype) if num_category is None else np.identity(num_category, dtype)
    return np.array([e[row] for row in label], dtype)


def filter_out_size(input_h, input_w, filter_h, filter_w, stride_h, stride_w, padding, force=False):
    if not force:
        assert (input_h + 2*padding - filter_h) % stride_h == 0
        assert (input_w + 2*padding - filter_w) % stride_w == 0
    out_h = int(1 + (input_h + 2*padding - filter_h) / stride_h)
    out_w = int(1 + (input_w + 2*padding - filter_w) / stride_w)
    return out_h, out_w

def im2col(input_data, filter_h, filter_w, stride_h=1, stride_w=1, pad=0, dtype=skml_config.config.f_type):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype)

    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride_h=1, stride_w=1, pad=0, dtype=skml_config.config.f_type):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride_h - 1, W + 2*pad + stride_w - 1), dtype)
    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            img[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



#######################
#######################
# 未テスト
#######################
#######################
class UnigramSampler:

    def __init__(self, corpus, sample_size, power=0.75):
        self.sample_size = sample_size
        counts = Counter()
        for word_id in corpus:
            counts[word_id] += 1
        self.vocab_size = len(counts)
        self.word_p = np.zeros(self.vocab_size, skml_config.config.i_type)
        for i in range(self.vocab_size):
            self.word_p[i] += counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), skml_config.config.i_type)
        for i in range(batch_size):
            target_id = target[i]
            p = self.word_p.copy()
            p[target_id] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, self.sample_size, False, p)
        return negative_sample
            