# imports 
import math 

# First load in the data 
training_lines = []
with open("data/train.txt") as f:
    training_lines = f.readlines()

testing_lines = []
with open("data/test.txt") as f:
    testing_lines = f.readlines()
############################################################################
# PREPROCESSING

def create_padded_word_list(sentence_list):
    """ A function that takes a list of sentences, adds padding and generates 

    Args:
        sentence_list (list): input list where each element is a sentence in the string format

    Returns:
        input_word_list (list): a list where each element is a token in the string format in the order they appear in the sentence list 
    """
    padded_corpus = []
    input_word_list = []
    for line in sentence_list: 
        padded_line = "<s> " + line.lower() + " </s>"
        padded_corpus.append(padded_line)
    
    for line in padded_corpus:
        words = line.split()
        for word in words:
            input_word_list.append(word)
    return input_word_list

# Creates training and testing tokens before replacing the unknowns 
training_tokens = create_padded_word_list(training_lines)
testing_tokens = create_padded_word_list(testing_lines)

# Create a dictionary of words without <unk>
def create_word_count_dict(input_word_list):
    """This function creates a dictionary of words and their respective counts 

    Args:
        input_word_list (list): a list where each element is a token in the string format in the order they appear in the sentence list

    Returns:
        input_word_dict (dictionary) : a dictionary of words and their respective counts based on the input_word_list
    """
    input_word_dict = dict()
    for word in input_word_list:
        if word not in input_word_dict:
            input_word_dict[word] = 1
        else:
            input_word_dict[word] += 1   
    return input_word_dict

# Create the training and testing word discts 
training_word_dict = create_word_count_dict(training_tokens) # contains word dicts without unknown
testing_word_dict = create_word_count_dict(testing_tokens) # contains word dicts without unknown

# Now to replace the words in the training that occur once with <unk>
replacement_word = "<unk>"
for i in range(len(training_tokens)):
    if training_word_dict[training_tokens[i]] == 1:
        training_tokens[i] = replacement_word

# Creating new dictionaries for training with <unk> present 
training_word_dict_with_unknown = create_word_count_dict(training_tokens)

# Replace words seen in testing not in training with <unk> 
def replace_unknown_test_word(testing_tokens, training_word_dict_with_unknown):
    """This function replaces the tokens in the testing data that do not occur in the training data 

    Args:
        testing_tokens (list): the list of the testing tokens 
        training_word_dict_with_unknown (dict): this contains the training words including <unk> and their relevant counts 

    Returns:
        testing_tokens: input testing tokens with <unk> in relevant places 
    """
    replacement_word = "<unk>"
    for i in range(len(testing_tokens)):
        word = testing_tokens[i]
        if not training_word_dict_with_unknown.__contains__(word):
            testing_tokens[i] = replacement_word 
    return testing_tokens   

# Replacing testing relevant tokens with <unk>
testing_tokens = replace_unknown_test_word(testing_tokens, training_word_dict_with_unknown) 
# Creating new dictionaries for training with <unk> present 
testing_word_dict_with_unknown = create_word_count_dict(testing_tokens)         

##################################################################################
# MODEL TRAINING
## 1.  Unigram Maximum Likelihood Model
# Evaluation of corpus using trained unigram model
# Create a method to find word probability - assumes you have a word dict built 

def calc_unigram_model_evaluation(word_list, training_word_dict_with_unknown):
    """This function takes returns the unigram log evaluation of an input word list 

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        training_word_dict_with_unknown (dict): this contains the training words including <unk> and their relevant counts

    Returns:
        final_evaluation (float): unigram log probability evaluation of original input word list based on formula 
    """
    number_of_tokens = sum(training_word_dict_with_unknown.values())
    final_evaluation = 0
    for word in word_list:
        if word != "<s>":
            number_of_word_occurence = training_word_dict_with_unknown[word]
            word_probability = number_of_word_occurence / number_of_tokens
            final_evaluation += math.log2(word_probability)
    return final_evaluation

## 2. Bigram Maximum Likelihood Model
# Create a bigram dictionary that has number of times bigram appeared occurences- create a dictionary 
def count_bigram_occurences(word_list):
    """This function creates a dictionary of bigrams and their respective counts

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from

    Returns:
        bigram_count_occurence (dict): a dictionary of bigrams and their respective counts based on the word_list
    """
    bigram_occurence_count = dict()
    for i in range(len(word_list)-1):
        word_pair = word_list[i], word_list[i+1]
        if word_pair != ('</s>', '<s>'):
            if word_pair in bigram_occurence_count:
                bigram_occurence_count[word_pair] += 1
            else:
                bigram_occurence_count[word_pair] = 1
    return bigram_occurence_count

bigram_count_dict = count_bigram_occurences(training_tokens)

# function to evaluate bigrams 
def calc_bigram_model_evaluation(word_list, bigram_word_dict, word_count_dict):
    """This function takes returns the bigram log evaluation of an input word list  

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        bigram_word_dict (dict): a dictionary of bigrams and their respective counts based on the training sentences 
        word_count_dict (dict): this contains the training words including <unk> and their relevant counts

    Returns:
        final_evaluation (float): bigram log probability evaluation of original input word list based on formula
    """
    num_of_unique_words = len(word_count_dict)
    final_evaluation = 0
    for i in range(len(word_list) - 1):
        word_pair = word_list[i] , word_list[i+1]
        first_word = word_list[i]
        if word_pair != ('</s>', '<s>'):
            # if the word pair exists there is a probability for it 
            if word_pair in bigram_word_dict:
                pair_probability = (bigram_word_dict[word_pair]) / (word_count_dict[first_word])
            # if it does not, zero should be used 
            else:
                pair_probability = 0
                final_evaluation *= pair_probability
                return final_evaluation
            final_evaluation += math.log2(pair_probability)
    return final_evaluation


# 3. Add One Bigram Model 
def calc_bigram_add_one_model_evaluation(word_list, bigram_word_dict, word_count_dict):
    """This function takes returns the add one smoothing bigram log evaluation of an input word list  

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        bigram_word_dict (dict): a dictionary of bigrams and their respective counts based on the training sentences 
        word_count_dict (dict): this contains the training words including <unk> and their relevant counts

    Returns:
        final_evaluation (float): add one smoothing bigram log probability evaluation of original input word list based on formula
    """
    num_of_unique_words = len(word_count_dict)
    final_evaluation = 0
    for i in range(len(word_list) - 1):
        word_pair = word_list[i] , word_list[i+1]
        first_word = word_list[i]
        if word_pair != ('</s>', '<s>'):
            # if the word pair exists there is a probability for it 
            if word_pair in bigram_word_dict:
                pair_probability = (bigram_word_dict[word_pair] + 1) / (word_count_dict[first_word] + num_of_unique_words)
            # if it does not, there is a zero and 1 should be used 
            else:
                pair_probability = 1 / (word_count_dict[first_word] + num_of_unique_words)
            final_evaluation += math.log2(pair_probability)
    return final_evaluation

##############################################################################
## Question Answers 
# 1. How many unique words are there in the training corpus with unknown and padding symbols?
#Number of keys in dictionary 
def find_vocabulary_size(training_word_dict_with_unknown):
    """This function returns the vocabulary size 

    Args:
        training_word_dict_with_unknown (dict): this contains the training words including <unk> and their relevant counts

    Returns:
        num_of_unique_words[integer]: number of keys in the dictionary that represent the number of unique words 
    """
    num_of_unique_words = len(training_word_dict_with_unknown)
    return num_of_unique_words

number_of_unique_words_training = find_vocabulary_size(training_word_dict_with_unknown)
print("Answer to Question No.1")
print(f'The number of unique words in the training corpus is {number_of_unique_words_training}')

# 2. How many tokens are there in the training corpus? 
def find_token_number(training_word_dict_with_unknown):
    """This function returns the total number of tokens based on a dictionary

    Args:
        training_word_dict_with_unknown (dict): this contains the training words including <unk> and their relevant counts

    Returns:
        total_token_num (integer): sum of key values in the dictionary that represent the total number of tokens 
    """
    total_token_num = sum(training_word_dict_with_unknown.values())
    return total_token_num

print()
print("Answer to Question No.2")
print(f"The number of tokens in the training corpus is {find_token_number(training_word_dict_with_unknown)}" )

# 3. Find percentage of word tokens and word types in the test corpus that did not 
# occur in training before mapping unknown in training and test data
def question_three(training_word_dict, testing_word_dict):
    """This function returns the answer to question number three, 
    It prints the percentage of words unseen in the test data, and the percentage
    of tokens unseen in test data compared to the training data.  

    Args:
        training_word_dict (dict):  this contains the training words and their relevant counts
        testing_word_dict (dict): this contains the testing words and their relevant counts
    """
    unseen_test_word_dict = dict()
    # Create a dictionary of testing words not in training
    for key, value in testing_word_dict.items():
        if key not in training_word_dict:
            unseen_test_word_dict[key] = value
    
    # now to print number of unique words
    num_of_unique_words_unseen = len(unseen_test_word_dict)
    sum_of_tokens_unseen = sum(unseen_test_word_dict.values())
    
    # work out percentage of word types and word types 
    num_of_words_test = len(testing_word_dict)
    num_of_tokens_test = sum(testing_word_dict.values())
    
    percentage_words_unseen = (num_of_unique_words_unseen / num_of_words_test) * 100
    percentage_tokens_unseen = (sum_of_tokens_unseen / num_of_tokens_test) * 100
    
    # print out the value 
    print()
    print("Answer to Question No.3" )
    print(f"The percentage of words unseen in test is {percentage_words_unseen}" )
    print(f"The percentage of tokens unseen in test is {percentage_tokens_unseen}" )

question_three(training_word_dict, testing_word_dict)

# 4. Now replace singletons in the training data with < unk > symbol and 
# map words (in the test corpus) not observed in training to < unk >. 
# What percentage of bigrams (bigram types and bigram tokens) in the test corpus 
# did not occur in training (treat "< unk >" as a regular token that has been observed).
def question_four(testing_words, bigram_count_dict, count_bigram_occurences):
    """This function returns the answer to question number 4. It prints the 
    percentage of unique bigrams in test not in training and the percentage of bigram 
    tokens in test not in training. 

    Args:
        testing_words (list): a list of testing words as they appear in the sentence in original test input sentences 
        bigram_count_dict (dict): a dictionary of bigrams and their respective counts based on the training sentences
        count_bigram_occurences (function): function that creates a bigram dictionary based on input sequential word list 

    """
    # create the bigram dictionary for test words 
    test_bigram_word_dict = count_bigram_occurences(testing_words)
    
    # create a dictionary to hold unseen values in test and populate
    test_bigram_word_dict_unseen = dict()
    for key,value in test_bigram_word_dict.items():
        if key not in bigram_count_dict:
            test_bigram_word_dict_unseen[key] = value
    
    # now to count the values 
    num_unique_bigrams_test = len(test_bigram_word_dict)
    sum_bigrams_test = sum(test_bigram_word_dict.values())
    num_unique_bigrams_test_unseen = len(test_bigram_word_dict_unseen)
    sum_bigrams_test_unseen = sum(test_bigram_word_dict_unseen.values())
    
    # calculate the percentages 
    percentage_test_word_unseen  = (num_unique_bigrams_test_unseen / num_unique_bigrams_test) * 100
    percentage_test_token_unseen  = (sum_bigrams_test_unseen / sum_bigrams_test) * 100

    print()
    print("Answer to Question No.4" )
    print(f"Percentage of unique bigrams in test not in training is {percentage_test_word_unseen}" )
    print(f"Percentage of bigram tokens in test not in training is {percentage_test_token_unseen}" )

question_four(testing_tokens, bigram_count_dict, count_bigram_occurences)

# 5. Compute the log probability of the following sentence under the three models 
# (ignore capitalization and pad each sentence as described above). 
# Please list all of the parameters required to compute the probabilities 
# and show the complete calculation. Which of the parameters have zero values 
# under each model? Use log base 2 in your calculations. 
# Map words not observed in the training corpus to the < unk > token.
# I look forward to hearing your reply .
print()
print("Answer to Question No.5")

# First create a padded word list and replace the unknown test words with unk
input_sentence = ["I look forward to hearing your reply ."]

# create the word list with padded symbols
padded_word_list = create_padded_word_list(input_sentence)

# replace the unknown words in with <unk>
processed_word_list = replace_unknown_test_word(padded_word_list, training_word_dict_with_unknown)
print(processed_word_list)

def find_M_value_perplexity(word_list):
    """This function returns the number of tokens for M in the perplexity calculation

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from

    Returns:
        M (int): the number of tokens without the start symbol <s>
    """
    M = 0
    for i in range(len(word_list)):
        if word_list[i] != "<s>":
            M += 1
    return M 

M_value_sentence = find_M_value_perplexity(processed_word_list)

##  Unigram Log Probability 
def calculate_log_probability_unigram(model_evaluation_function, training_word_dict_with_unknown, word_list, M_value):
    """This function calculates the average log probability 

    Args:
        model_evaluation_function (function): the unigram log probability calculator function
        training_word_dict_with_unknown (dict): this contains the training words including <unk> and their relevant counts
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        M_value (integer) :  the number of tokens without the start symbol <s>

    Returns:
        log_probability (float): the average log probability 
    """

    # calculate the log probability 
    log_probability = (1/ M_value) * model_evaluation_function(word_list, training_word_dict_with_unknown)
    return log_probability 


unigram_model_evaluation = calc_unigram_model_evaluation(processed_word_list, training_word_dict_with_unknown)
print(f"1. Unigram Log Probability {unigram_model_evaluation}" )
unigram_log_probability = calculate_log_probability_unigram(calc_unigram_model_evaluation, training_word_dict_with_unknown, processed_word_list, M_value_sentence)
print(f"1. Unigram Average Log Probability {unigram_log_probability}" )

## Bigram Log Probability 
bigram_model_evaluation_line = calc_bigram_model_evaluation(processed_word_list, bigram_count_dict, training_word_dict_with_unknown)
print(f"2. Bigram Model Evaluation {bigram_model_evaluation_line}")
print("As the bigram model log probability evaluation is zero, there is no average log probability. Perplexity is undefined" )

def calculate_log_probability_bigram(word_list, calc_bigram_evaluation, bigram_word_dict, word_count_dict, M_value):
    """This function calculates the average log probability of sentence under bigram model

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        calc_bigram_evaluation (function): bigram log probability evaluation function
        bigram_word_dict (dict): a dictionary of bigrams and their respective counts based on the training sentences
        word_count_dict (dict): this contains the training words including <unk> and their relevant counts
        M_value (integer) :  the number of tokens without the start symbol <s>

    Returns:
        log_probability (float): the average log probability
    """
    model_evaluation = calc_bigram_evaluation(word_list, bigram_word_dict, word_count_dict)
    log_probability = (1/ M_value) * model_evaluation
    return log_probability  

# calculates the log probability of add one bigram model
def calculate_log_probability_bigram_add_one(word_list, calc_bigram_add_one_model_evaluation, bigram_word_dict, word_count_dict, M_value):
    """This function calculates the average log probability of sentence under bigram model with add one smoothing

    Args:
        word_list (list): input word list in the order that they appear in the original sentence they were derived from
        calc_bigram_evaluation (function): bigram log probability evaluation function
        bigram_word_dict (dict): a dictionary of bigrams and their respective counts based on the training sentences
        word_count_dict (dict): this contains the training words including <unk> and their relevant counts
        M_value (integer) :  the number of tokens without the start symbol <s>

    Returns:
        log_probability (float): the average log probability
    """    
    model_evaluation = calc_bigram_add_one_model_evaluation(word_list, bigram_word_dict, word_count_dict)
    log_probability = (1/ M_value) * model_evaluation
    return log_probability 

bigram_model_evaluation = calc_bigram_add_one_model_evaluation(processed_word_list, bigram_count_dict, training_word_dict_with_unknown)
print(f"3.Bigram Add One Log Probability {bigram_model_evaluation}" )   
bigram_add_one_log_probability = calculate_log_probability_bigram_add_one(processed_word_list, calc_bigram_add_one_model_evaluation, bigram_count_dict, training_word_dict_with_unknown, M_value_sentence)
print(f"3.Bigram Add One Average Log Probability {bigram_add_one_log_probability}" )   

# 6. Compute the perplexity of the sentence above under each of the models.
print()
print("Answer to Question No 6" )

# perplexity for the unigram model
input_unigram_perplexity = 2 ** -(unigram_log_probability)
print(f"Perplexity of sentence under unigram model {input_unigram_perplexity}" )

# perplexity for the add one bigram model 
input_add_one_bigram_perplexity = 2 ** -(bigram_add_one_log_probability)
print(f"Perplexity of sentence under add one bigram model {input_add_one_bigram_perplexity}" )

# 7. Compute the perplexity of the entire test corpus under each of the models. 
# Discuss the differences in the results you obtained. 
print()
print("Answer to Question No.7" )
# Unigram Model Evaluation on Test Corpus 
# we have testing words preprocessed already
unigram_model_evaluation_test = calc_unigram_model_evaluation(testing_tokens, training_word_dict_with_unknown)
print(f"The unigram log probability for the test corpus is {unigram_model_evaluation_test}" )

M_value_test = find_M_value_perplexity(testing_tokens)

unigram_log_probability_test  = calculate_log_probability_unigram(calc_unigram_model_evaluation, training_word_dict_with_unknown, testing_tokens, M_value_test)
print(f"The unigram average log probability for the test corpus is {unigram_log_probability_test}" )
perplexity_unigram_test = 2 ** -(unigram_log_probability_test)
print(f"Perplexity of test corpus under unigram model {perplexity_unigram_test}" )

# Bigram Model Evaluation on Test Corpus
# first find model evaluation 
bigram_model_evaluation_test = calc_bigram_model_evaluation(testing_tokens, bigram_count_dict, training_word_dict_with_unknown)
bigram_log_probability_test = calculate_log_probability_bigram(testing_tokens, calc_bigram_model_evaluation, bigram_count_dict, training_word_dict_with_unknown, M_value_test)
print(f"Bigram Model Evaluation on test corpus {bigram_model_evaluation_test}")
print("As the bigram model log probability evaluation is zero, there is no average log probability. Perplexity is undefined" )
# bigram_perplexity_test = 2 ** -(bigram_log_probability_test)
# print(f"Bigram Log Perplexity {bigram_perplexity_test}")


# Add-one Bigram Model Evaluation on Test Corpus
bigram_add_one_model_evaluation_test = calc_bigram_add_one_model_evaluation(testing_tokens, bigram_count_dict, training_word_dict_with_unknown)
print(f"Add One Smoothing Bigram Log Probability {bigram_add_one_model_evaluation_test}")

bigram_add_one_log_probability_test = calculate_log_probability_bigram_add_one(testing_tokens, calc_bigram_add_one_model_evaluation, bigram_count_dict, training_word_dict_with_unknown, M_value_test)
print(f"Add One Smoothing Bigram Average Log Probability {bigram_add_one_log_probability_test}")
bigram_add_one_perplexity_test = 2 ** -(bigram_add_one_log_probability_test)
print(f"Add One Smoothing Bigram Perplexity {bigram_add_one_perplexity_test}")
# bigram_add_one_log_probability_test = calculate_log_probability_bigram_add_one(testing_words, calc_bigram_add_one_model_evaluation, bigram_count_dict, training_word_dict_with_unknown)
# print(f"Bigram Add One Log Probability {bigram_add_one_log_probability_test}")
