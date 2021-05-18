def tokenize()

def build_co_occurence_mat(tokens, window_size = 4, sym = True):
    V = np.nunique(tokens)
    mat = np.zeros((V, V))
    if sym == True:
        for i in range(len(tokens)):
        word_r = tokens[i]
        if i < window_size:
            contexts = tokens[:i + window_size + 1]
        elif i >= window_size and i < len(tokens) - window_size:
            contexts = tokens[i - window_size:i + window_size + 1]
        else:
            contexts = tokens[i - window_size:]
        counts = dict(zip(*np.unique(contexts, return_counts = True)))
        for word_c in counts.keys():
            mat[word_r, word_c] += counts[word_c]
    return mat


