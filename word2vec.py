import pickle
from word2vec_utils import *
import time
import random
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_json(path):
    import json
    dic = json.load(open(path, "r"))
    return dic


def create_corpus(df):
    corpus = []
    for i, row in df.iterrows():
        corpus.append(row["Review"])
    return corpus


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])
    return list(target_words)


def get_idx_pairs(df, tokenizer):
    x = []
    y = []
    for i, row in df.iterrows():
        tokenized_abstract_words = tokenizer.texts_to_sequences([row["Review"]])[0]
        for i, word in enumerate(tokenized_abstract_words):
            x.append(word)
            target_words = get_target(tokenized_abstract_words, i)
            y.append(target_words)

    return x, y


def get_batches(x, y, batch_size):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    n_batches = len(x) // batch_size

    # only full batches
    words = x[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        curr_words, context_words = [], []
        batch_x = words[idx:idx + batch_size]
        batch_y = y[idx:idx + batch_size]

        for ii in range(len(batch_x)):
            context_words.extend(batch_y[ii])
            curr_words.extend([batch_x[ii]] * len(batch_y[ii]))
        yield curr_words, context_words


if __name__ == "__main__":
    base_path = "/data4/dheeraj/covid-analysis/"
    dataset = ""

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_29k_abstract_phrase_removed_stopwords.pkl", "rb"))
    dump = True

    corpus = create_corpus(df)
    vocabulary, vocab_to_int, int_to_vocab, tokenizer = create_vocabulary(corpus, num_words=135105)

    print("Size of vocabulary: ", len(vocabulary))

    current_words, context_words = get_idx_pairs(df, tokenizer)

    # Graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        #     labels = tf.placeholder(tf.int32, [None, None], name='labels')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    n_vocab = len(int_to_vocab)
    n_embedding = 100
    with train_graph.as_default():
        embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)  # use tf.nn.embedding_lookup to get the hidden layer output

    # Number of negative labels to sample
    n_sampled = 100
    with train_graph.as_default():
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding)))  # create softmax weight matrix here
        softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias")  # create softmax biases here

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(
            weights=softmax_w,
            biases=softmax_b,
            labels=labels,
            inputs=embed,
            num_sampled=n_sampled,
            num_classes=n_vocab)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with train_graph.as_default():
        ## From Thushan Ganegedara's implementation
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
        valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
        valid_examples = np.append(valid_examples,
                                   random.sample(range(1000, 1000 + valid_window), valid_size // 2))

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    epochs = 10
    batch_size = 1000

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            batches = get_batches(current_words, context_words, batch_size)
            start = time.time()
            for x, y in batches:

                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                loss += train_loss

                if iteration % 100 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 100),
                          "{:.4f} sec/batch".format((end - start) / 100))
                    loss = 0
                    start = time.time()

                if iteration % 1000 == 0:
                    ## From Thushan Ganegedara's implementation
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

                iteration += 1
        save_path = saver.save(sess, "checkpoints/dblp.ckpt")
        embed_mat = sess.run(normalized_embedding)

        if dump:
            pickle.dump(vocabulary, open(data_path + "vocabulary.pkl", "wb"))
            pickle.dump(vocab_to_int, open(data_path + "vocab_to_int.pkl", "wb"))
            pickle.dump(int_to_vocab, open(data_path + "int_to_vocab.pkl", "wb"))

        pickle.dump(embed_mat, open(data_path + "embedding_matrix.pkl", "wb"))
        pickle.dump(tokenizer, open(data_path + "tokenizer.pkl", "wb"))
