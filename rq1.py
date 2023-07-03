import gensim.downloader
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
from transformers import pipeline
from matplotlib.cm import get_cmap
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import itertools

""" Uncommenting the following line will print all available models to the console """
#print(list(gensim.downloader.info()['models'].keys()))

""" Set the model Name """
#model = 'word2vec-google-news-300'


def listToString(inputList):
    return (','.join(inputList))


def getEmbedding(model):

    """ Check if the model was already pre-loaded, else download and store it in directory """
    try:
        embedding = pickle.load(open(model+'.pickle', "rb"))  # Load model from directory
    except:
        embedding = gensim.downloader.load(model)  # Download model
        pickle.dump(embedding, open(model+'.pickle', "wb"))  # Store the model to our directory for future use

    return embedding


def getSimilarWords(embedding, positive_words, topn_words, negative_words=None):

    similar_words = []
    for word, _ in embedding.most_similar(positive=positive_words, negative=negative_words, topn=topn_words):
        if word.isalpha():
            similar_words.append(word)

    #print(similar_words[:5], similar_words[-5:])
    return similar_words


def removeCommonWords(similar_words_a, similar_words_b):

    common_words = set(similar_words_a).intersection(set(similar_words_b))
    similar_words_a = [word for word in similar_words_a if word not in common_words]
    similar_words_b = [word for word in similar_words_b if word not in common_words]

    return similar_words_a, similar_words_b


def getSentimentScoreProb(similar_words):

    model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    #model = SentimentIntensityAnalyzer()

    counts = np.zeros((len(similar_words), 3))
    for i, word_list in enumerate(similar_words):
        for j, word in enumerate(word_list):
            score = model(word)[0]
            if score['label'] == 'NEG':
                counts[i, 0] += 1
            elif score['label'] == 'POS':
                counts[i, 2] += 1
            else:
                counts[i, 1] += 1
        #res.append(sentiment_scores)
        counts[i] = counts[i] / len(word_list)

    return counts


def getStatistics(sentiment_scores):

    means = []
    stds = []
    for i, scores in enumerate(sentiment_scores):
        means.append(np.mean(scores, axis=0))
        stds.append(np.std(scores, axis=0))

    return np.array(means), np.array(stds)


def checkNormalDistribution(sentiment_scores, test_interval=0.05):

    for scores in sentiment_scores:
        if stats.shapiro(scores[:, 0]).statistic < test_interval or stats.shapiro(scores[:, 1]).statistic < test_interval:
            return False

    return True


def testSignificance(sentiment_scores, test_interval=0.2):

    for k in range(3):
        if k == 0:
            print('NEG')
        elif k == 1:
            print('NEU')
        else:
            print('POS')

        for i in range(len(sentiment_scores) - 1):
            for j in range(i + 1, len(sentiment_scores)):
                if abs((sentiment_scores[i, k] / sentiment_scores[j, k]) - 1) >= test_interval * 4:
                    print('Pos ' + str(i) + ' with ' + str(j) + ': ***')
                elif abs((sentiment_scores[i, k] / sentiment_scores[j, k]) - 1) >= test_interval * 2:
                    print('Pos ' + str(i) + ' with ' + str(j) + ': **')
                elif abs((sentiment_scores[i, k] / sentiment_scores[j, k]) - 1) >= test_interval:
                    print('Pos ' + str(i) + ' with ' + str(j) + ': *')
                else:
                    print('Pos ' + str(i) + ' with ' + str(j) + ': ns')

    return


def createErrorBars(all_stds, all_means):

    res = []
    for i, _ in enumerate(all_stds):
        stds = []
        for j in range(2):
            if all_stds[i, j] > all_means[i, j]:
                stds.append([all_means[i, j], all_stds[i, j]])
            else:
                stds.append([all_stds[i, j], all_stds[i, j]])
        res.append(stds)

    return np.array(res)


def makePlot(sentiment_scores, input_words):

    # set width of bar
    barWidth = 0.25
    padding = 2
    colors = get_cmap('Pastel1_r').colors

    # set height of bar
    bar_heights = [score for score in sentiment_scores]

    # Set position of bar on X axis
    bar_positions = [[0, 2, 4]]
    for i in range(1, len(sentiment_scores)):
        bar_positions.append([x + barWidth for x in bar_positions[i - 1]])

    #error_bars = createErrorBars(all_stds, all_means)
    # Make the plot
    for i, _ in enumerate(bar_positions):
        plt.bar(bar_positions[i], bar_heights[i], align='center', capsize=10, color=colors[i],
                width=barWidth, edgecolor='grey', label=listToString(input_words[i]))

    # Adding Xticks
    plt.xlabel('Sentiment Category', fontweight='bold', fontsize=15)
    plt.ylabel('Sentiment Score Probability', fontweight='bold', fontsize=15)
    plt.xticks([(bar_positions[0][i] + bar_positions[-1][i] + barWidth)/2 - (barWidth/2) for i, _ in enumerate(bar_positions[0])], ['NEG', 'NEU', 'POS'])
    plt.title('Sentiment Score Probability for each Sentiment Category of Chosen Words', fontsize=20)

    """
    x1 = [bar_positions[0][0], bar_positions[0][0], bar_positions[1][0], bar_positions[1][0], bar_positions[2][0],
          bar_positions[1][1], bar_positions[2][1], bar_positions[0][1]]
    x2 = [bar_positions[1][0], bar_positions[3][0], bar_positions[2][0], bar_positions[3][0], bar_positions[3][0],
          bar_positions[2][1], bar_positions[3][1], bar_positions[2][1]]
    y1 = [0.6, 0.95, 0.65, 0.90, 0.85,
          0.6, 0.65, 0.70]
    y2 = np.array(y1) + 0.02

    for i, _ in enumerate(y1):
        temp_x1 = x1[i]
        temp_x2 = x2[i]
        temp_y1 = y1[i]
        temp_y2 = y2[i]
        plt.plot([temp_x1, temp_x1, temp_x2, temp_x2], [temp_y1, temp_y2, temp_y2, temp_y1], linewidth=1, color='k')
        plt.text((temp_x1 + temp_x2) * .5, temp_y2, "*", ha='center', va='bottom', color='black')"""

    plt.legend()
    plt.show()

    return


def main(input_words, opposite_pairs, topn_words, showPlots=True, show_significance=True):
    """ Set the words you would like the model to use as input. Need to re-run code to change words """

    #sentiment_id = {'POS': 2, 'NEU': 1, 'NEG': 0}
    embedding = getEmbedding(model='glove-twitter-200')

    similar_words = []
    for i, positive_words in enumerate(input_words):

        if i in list(opposite_pairs.keys()):
            negative_words = input_words[opposite_pairs[i]]
        else:
            negative_words = None

        similar_words.append(getSimilarWords(embedding=embedding,
                                             positive_words=positive_words,
                                             negative_words=None,
                                             topn_words=topn_words))

    for i, key in enumerate(list(opposite_pairs.keys())):
        if i % 2 == 1:
            continue
        similar_words[i], similar_words[i+1] = removeCommonWords(similar_words[i], similar_words[i+1])

    sentiment_scores = getSentimentScoreProb(similar_words)
    """
    means, stds = getStatistics(sentiment_scores)
    if checkNormalDistribution(sentiment_scores):
        print("All data is normally distributed")
    else:
        print("Not all data is normally distributed")"""
    if show_significance:
        testSignificance(sentiment_scores)
    if showPlots:
        makePlot(sentiment_scores, input_words=input_words)

    return sentiment_scores


if __name__ == "__main__":

    topn_words = 100
    input_words = [['fat', 'obese', 'chubby', 'stout', 'overweight', 'flabby'],
                   ['thin', 'anorexic', 'skinny', 'slender', 'underweight', 'scrawny'],
                   ['beautiful', 'handsome', 'intelligent', 'smart', 'kind'],
                   ['ugly', 'hideous', 'stupid', 'dumb', 'unlikable']]
    opposite_pairs = {0: 1, 1: 0, 2: 3, 3: 2}

    base_score = main(input_words=input_words,
                      opposite_pairs=opposite_pairs,
                      topn_words=topn_words,
                      showPlots=True,
                      show_significance=True)
    print(base_score)

    base_words = input_words.copy()
    for i in range(2):
        for words in list(itertools.combinations(input_words[i], len(input_words[i])-1)):
            input_words[i] = list(words)
            test_score = main(input_words=input_words,
                              opposite_pairs=opposite_pairs,
                              topn_words=topn_words,
                              showPlots=False,
                              show_significance=False)
            print("Effect size of: ",
                  [item for item in base_words[i] if item not in input_words[i]])
            print(base_score - test_score)