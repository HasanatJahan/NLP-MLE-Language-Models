# First load in the data 
training_lines = []
with open("data/train.txt") as f:
    training_lines = f.readlines()

testing_lines = []
with open("data/test.txt") as f:
    testing_lines = f.readlines()

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
print(testing_word_dict_with_unknown)
