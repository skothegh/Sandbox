import re
import numpy as np
import matplotlib.pyplot as plt
def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''


def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res


def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


tokens = tokenize(text)

def mapping(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

word_to_id, id_to_word = mapping(tokens)

np.random.seed(42)

def generate_training_data(tokens, word_to_id, window):
    x = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            else:
                x.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
                y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(x), np.asarray(y)

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

X,Y = generate_training_data(tokens, word_to_id, 2)


def init_network(vocab_size, n_embedding):
    model = {
            "w1": 0.01*np.random.randn(vocab_size, n_embedding),
            "w2": 0.01*np.random.randn(n_embedding, vocab_size)
    }
    return model

model = init_network(len(word_to_id),10)

def forward(model, X, return_cache = True):
    cache = {}

    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    else:
        return cache

def backward(model, X, Y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - Y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], Y)

def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


n_iter = 50
learning_rate = 0.05

history = [backward(model, X, Y, learning_rate) for _ in range(n_iter)]

# plt.plot(range(len(history)), history, color="skyblue")
# plt.show()

def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print(f"{word} not in corpus")
    one_hot  = one_hot_encode(idx, len(word_to_id))
    return forward(model, one_hot)["a1"]

#END
