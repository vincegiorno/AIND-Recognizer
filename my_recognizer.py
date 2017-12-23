import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for id in range(0, len(test_set.wordlist)):
        test_X, test_length = test_set.get_item_Xlengths(id)
        prob_dict = {}
        for word, model in models.items():
            try:
                prob_dict[word] = model.score(test_X, test_length)
            except:
                prob_dict[word] = float("-inf")
        probabilities.append(prob_dict)
        most_likely = None
        best_score = float("-inf")
        for word, prob in prob_dict.items():
            if prob > best_score:
                best_score = prob
                most_likely = word
        guesses.append(most_likely)
    return probabilities, guesses