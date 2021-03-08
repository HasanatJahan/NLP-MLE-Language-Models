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
        print(padded_line)
        output_list.append(padded_line)
    return output_list

training_padded_lines = create_padded_lines(training_lines)
testing_padded_lines = create_padded_lines(testing_lines)

# Now to create a dictionary of words without unknowns
def create_word_list(sentence_list):
    output_list = []
    for line in sentence_list:
        words = line.split()
        for word in words:
            output_list.append(word)
    return output_list

training_tokens = create_word_list(training_padded_lines)
testing_tokens = create_word_list(testing_padded_lines)

         