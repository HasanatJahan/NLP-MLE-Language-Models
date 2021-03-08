# imports 
import re
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
    padded_corpus = []
    input_word_list = []
    for line in sentence_list: 
        padded_line = "<s> " + line + " </s>"
        padded_corpus.append(padded_line)
    
    for line in padded_corpus:
        words = line.split()
        for word in words:
            input_word_list.append(word)
    return input_word_list


training_tokens = create_padded_word_list(training_lines)
testing_tokens = create_padded_word_list(testing_lines)

# Create a dictionary of words without <unk>
def create_word_count_dict(input_word_list):
    input_word_dict = dict()
    for word in input_word_list:
        if word not in input_word_dict:
            input_word_dict[word] = 1
        else:
            input_word_dict[word] += 1   
    return input_word_dict

training_word_dict = create_word_count_dict(training_tokens) # contains word dicts without unknown
testing_word_dict = create_word_count_dict(testing_tokens) # contains word dicts without unknown

# Now to replace the words in the training that occur once with <unk>
replacement_word = "<unk>"
for i in range(len(training_tokens)):
    if training_word_dict[training_tokens[i]] == 1:
        training_tokens[i] = replacement_word

# Creating new dictionaries for training and testing with <unk> present 
training_word_dict_with_unknown = create_word_count_dict(training_tokens)

# Replace words seen in testing not in training with <unk> 
def replace_unknown_test_word(testing_tokens, training_word_dict_with_unknown):
    replacement_word = "<unk>"
    for i in range(len(testing_tokens)):
        word = testing_tokens[i]
        if not training_word_dict_with_unknown.__contains__(word):
            testing_tokens[i] = replacement_word    

testing_word_dict_with_unknown = create_word_count_dict(testing_tokens)         

replace_unknown_test_word(testing_tokens, training_word_dict_with_unknown) 

##################################################################################
# MODEL TRAINING
## 1.  Unigram Maximum Likelihood Model
# Evaluation of corpus using trained unigram model
# Create a method to find word probability - assumes you have a word dict built 
def calculate_word_probability_unigram(word, word_dict):
    number_of_tokens = sum(word_dict.values())
    number_of_word_occurence = word_dict[word]
    word_probability = number_of_word_occurence/number_of_tokens
    return word_probability

# Now to create a unigram word probability dict 
def create_unigram_probability_dict(word_dict):
    probability_dict = dict()
    for key in word_dict:
        probability_dict[key] = calculate_word_probability_unigram(key, word_dict)
    return probability_dict 

# function to evaluate unigrams 
def calc_unigram_model_evaluation(word_list, unigram_probability_dict):
    final_evaluation = 1
    for word in word_list: 
        final_evaluation *= unigram_probability_dict[word]
    return final_evaluation

## 2. Bigram Maximum Likelihood Model
# Create a bigram dictionary that has number of times bigram appeared occurences- create a dictionary 
def count_bigram_occurences(word_list):
    bigram_occurence_count = dict()
    for i in range(len(word_list)-1):
        word_pair = word_list[i], word_list[i+1]
        if word_pair in bigram_occurence_count:
            bigram_occurence_count[word_pair] +=1
        else:
            bigram_occurence_count[word_pair] = 1
    return bigram_occurence_count

bigram_count_dict = count_bigram_occurences(training_tokens)

# function to evaluate bigrams 
def calc_bigram_model_evaluation(word_list, bigram_word_dict, word_count_dict):
    num_of_unique_words = len(word_count_dict)
    final_evaluation = 1
    for i in range(len(word_list) - 1):
        word_pair = word_list[i] , word_list[i+1]
        first_word = word_list[i]
        # if the word pair exists there is a probability for it 
        if word_pair in bigram_word_dict:
            pair_probability = (bigram_word_dict[word_pair] + 1) / (word_count_dict[first_word] + num_of_unique_words)
        # if it does not, zero should be used 
        else:
            pair_probability = 0
            final_evaluation *= pair_probability
            return final_evaluation
        final_evaluation *= pair_probability
    return final_evaluation


# 3. Add One Bigram Model 
def calc_bigram_add_one_model_evaluation(word_list, bigram_word_dict, word_count_dict):
    num_of_unique_words = len(word_count_dict)
    final_evaluation = 1
    for i in range(len(word_list) - 1):
        word_pair = word_list[i] , word_list[i+1]
        first_word = word_list[i]
        # if the word pair exists there is a probability for it 
        if word_pair in bigram_word_dict:
            pair_probability = (bigram_word_dict[word_pair] + 1) / (word_count_dict[first_word] + num_of_unique_words)
        # if it does not, there is a zero and 1 should be used 
        else:
            pair_probability = 1 / (word_count_dict[first_word] + num_of_unique_words)
        final_evaluation *= pair_probability
    return final_evaluation

##############################################################################
## Question Answers 
# 1. How many unique words are there in the training corpus with unknown and padding symbols?
#Number of keys in dictionary 
def find_vocabulary_size(training_word_dict_with_unknown):
    num_of_unique_words = len(training_word_dict_with_unknown)
    return num_of_unique_words

number_of_unique_words_training = find_vocabulary_size(training_word_dict_with_unknown)
print("Answer to Question No.1")
print(f"The number of unique words in the training corpus is {number_of_unique_words_training}" )

# 2. How many tokens are there in the training corpus? 
def find_token_number(training_word_dict_with_unknown):
    total_token_num = sum(training_word_dict_with_unknown.values())
    return total_token_num

print()
print("Answer to Question No.2")
print(f"The number of tokens in the training corpus is {find_token_number(training_word_dict_with_unknown)}" )

# 3. Find percentage of word tokens and word types in the test corpus that did not 
# occur in training before mapping unknown in training and test data
def question_three(training_word_dict, testing_word_dict):
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
replace_unknown_test_word(padded_word_list, training_word_dict)
processed_word_list = padded_word_list

##  Unigram Log Probability 
def calculate_log_probability_unigram(model_evaluation_function, probability_dict, word_list):    
    num_of_tokens = len(word_list)
    # calculate the log probability 
    log_probability = (1/ num_of_tokens) * math.log2(model_evaluation_function(word_list, probability_dict))
    return log_probability 

unigram_log_probability = calculate_log_probability_unigram(calc_unigram_model_evaluation, training_word_dict_with_unknown, processed_word_list)
print(f"1. Unigram Log Probability {unigram_log_probability}" )

## Bigram Log Probability 
bigram_model_evaluation_line = calc_bigram_model_evaluation(processed_word_list, bigram_count_dict, training_word_dict_with_unknown)
print(f"2. Bigram Model Evaluation {bigram_model_evaluation_line}\nAs it is zero, there is no log probability" )

def calculate_log_probability_bigram(word_list, calc_bigram_evaluation, bigram_word_dict, word_count_dict):
    num_of_tokens = len(word_list)
    model_evaluation = calc_bigram_evaluation(word_list, bigram_word_dict)
    log_probability = (1/ num_of_tokens) * math.log2(model_evaluation)
    return log_probability  

## Add One Bigram Log Probability
# calculates the model evaluation for add one bigram model 
def calc_bigram_add_one_model_evaluation(word_list, bigram_word_dict, word_count_dict):
    num_of_unique_words = len(word_count_dict)
    final_evaluation = 1
    for i in range(len(word_list) - 1):
        word_pair = word_list[i] , word_list[i+1]
        first_word = word_list[i]
        # if the word pair exists there is a probability for it 
        if word_pair in bigram_word_dict:
            pair_probability = (bigram_word_dict[word_pair] + 1) / (word_count_dict[first_word] + num_of_unique_words)
        # if it does not, there is a zero and 1 should be used 
        else:
            pair_probability = 1 / (word_count_dict[first_word] + num_of_unique_words)
        final_evaluation *= pair_probability
    return final_evaluation

# calculates the log probability of add one bigram model
def calculate_log_probability_bigram_add_one(word_list,calc_bigram_add_one_model_evaluation, bigram_word_dict, word_count_dict):    
    num_of_tokens = len(word_list)
    model_evaluation = calc_bigram_add_one_model_evaluation(word_list, bigram_word_dict, word_count_dict)
    log_probability = (1/ num_of_tokens) * math.log2(model_evaluation)
    return log_probability 

bigram_add_one_log_probability = calculate_log_probability_bigram_add_one(processed_word_list, calc_bigram_add_one_model_evaluation, bigram_count_dict, training_word_dict_with_unknown)
print(f"3.Bigram Add One Log Probability {bigram_add_one_log_probability}" )   

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
unigram_log_probability_test  = calculate_log_probability_unigram(calc_unigram_model_evaluation, training_word_dict_with_unknown, testing_tokens)
print(f"The unigram log probability for the test corpus is {unigram_log_probability_test}" )
perplexity_unigram_test = 2 ** -(unigram_log_probability_test)
print(f"Perplexity of test corpus under unigram model {perplexity_unigram_test}" )

# Bigram Model Evaluation on Test Corpus
# first find model evaluation 
bigram_model_evaluation_test = calc_bigram_model_evaluation(testing_tokens, bigram_count_dict, training_word_dict_with_unknown)
print(f"Bigram Model evaluation without log {bigram_model_evaluation_test}" )
# bigram_log_probability_test = calculate_log_probability_bigram(testing_words,calc_bigram_model_evaluation, bigram_probability_dict, training_word_dict_with_unknown)
# print(f"Bigram Log Probability on test corpus {bigram_log_probability_test}")
print("As the bigram model evaluation is zero, there is no log probability" )

# Add-one Bigram Model Evaluation on Test Corpus
bigram_add_one_model_evaluation_test = calc_bigram_add_one_model_evaluation(testing_tokens, bigram_count_dict, training_word_dict_with_unknown)
print(f"Add One Bigram Model evalaution without log {bigram_add_one_model_evaluation_test}" )
print("As the add one bigram model evaluation is zero, there is no log probability" )
# bigram_add_one_log_probability_test = calculate_log_probability_bigram_add_one(testing_words, calc_bigram_add_one_model_evaluation, bigram_count_dict, training_word_dict_with_unknown)
# print(f"Bigram Add One Log Probability {bigram_add_one_log_probability_test}")
