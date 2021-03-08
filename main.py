# First load in the data 
training_lines = []
with open("data/train.txt") as f:
    training_lines = f.readlines()

testing_lines = []
with open("data/test.txt") as f:
    testing_lines = f.readlines()

############################################################################
# PREPROCESSING
# Now to pad each line with start and end symbol for training and testing data 
training_padded_lines = []
testing_padded_lines = []

def create_padded_lines(sentence_list):
    count = 0
    output_list = []
    for line in sentence_list:
        count += 1
        padded_line = "<s> " + line.lower() + " </s>"
        output_list.append(padded_line)
    return output_list

training_padded_lines = create_padded_lines(training_lines)
testing_padded_lines = create_padded_lines(testing_lines)

# Now to create a list of words without unknowns
def create_word_list(sentence_list):
    output_list = []
    for line in sentence_list:
        words = line.split()
        for word in words:
            output_list.append(word)
    return output_list


training_tokens = create_word_list(training_padded_lines)
testing_tokens = create_word_list(testing_padded_lines)


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
def replace_unknowns(training_word_dict_with_unknown, testing_tokens):
    replacement_word = "<unk>"
    for i in range(len(testing_tokens)):
        if not training_word_dict_with_unknown.__contains__(testing_tokens[i]):
            testing_tokens[i] = replacement_word    

replace_unknowns(training_word_dict_with_unknown, testing_tokens) 
testing_word_dict_with_unknown = create_word_count_dict(testing_tokens)         

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
print(f"The number of unique words in the training corpus is {number_of_unique_words_training}")