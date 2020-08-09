
def split_sentence(split_punctuation,input_sentence):

    result = list()
    index = 0
    
    sentence = ""
    while(index < len(input_sentence)):
        if input_sentence[index] in split_punctuation:
            result.append(sentence)
            sentence = ""
        else:
            sentence += input_sentence[index]
        index = index + 1

    return result

def get_sentence_lists(text):
    sentense_list =  split_sentence(['.','!','?'], text) 
    returnLis = list()
 
    for x in sentense_list:
        splitList =  split_sentence([' ',';', ',', '_', '\''], text) #re.split(r'[ :;\-\_\'\":,]+',x)
        #remove elemets that are just spaces
        returnLis.append(splitList)
        
    return returnLis #TODO: remove last element. there is a an extra [''] when the last sentence ends in ?, . or !. try to remove it

def get_sentence_lists_from_files(filenames):

    text =""
    for filename in filenames:
        curr_string = open(filename, 'r').read()
        text = "".join((text, curr_string))

   
    sentense_list = split_sentence(['.','!','?'], text) 
    
    returnLis = list()
    #print(sentense_list)
    for x in sentense_list:
        splitList = split_sentence([' ',';', ',', '_', '\'', '\"'], x)
        #remove elemets that are just spaces
        returnLis.append(splitList)
        
        
    return returnLis #TODO: remove last element

def build_semantic_descriptors(sentences):
    #all unique words from the sentences
    unique_words = list()
    for sentence in sentences:
        for word in sentence:
            if word not in unique_words:
                unique_words.append(word)

    dictionary = {}
    for word in unique_words:
        #check which sentences have that word
        masterSentence = list()
        for sentence in sentences:
            if word in sentence:
                masterSentence.append(sentence)
        
        #create dictionary of semantic discriptors for the word
        currDict = {}
        for sentence in masterSentence:
            for currword in sentence:
                if currword != word: #dont count the word itself when looking at descriptors
                    if currword not in currDict:
                        currDict[currword] = 1
                    else:
                        num = currDict.get(currword)
                        currDict[currword] = num + 1
        dictionary[word] = currDict
    
    return dictionary

def most_similar_word(word, choices, semantic_descriptors):

    
    if word not in semantic_descriptors:
        return -1

    #get the semenatic discriptor for "word"
    descriptor = semantic_descriptors.get(word)
    #print(descriptor)
    highest_similarity_score = 0
    highest_similarity_word = ""

    for choice in choices:
        if choice in descriptor:
            if descriptor.get(choice) > highest_similarity_score:
                highest_similarity_score = descriptor.get(choice)
                highest_similarity_word = choice
    
    return highest_similarity_word

def run_similarity_test(filename, semantic_descriptors):
    #load file
    #lines = list()
    lines = tuple(open(filename, 'r'))
    counter = 0
    for line in lines:
        words = split_sentence([' '],line.rstrip("\n"))
        if most_similar_word(words[0], words[0:-1], semantic_descriptors) == words[-1]:
            counter = counter +1
    
    percentage = (counter/len(lines))*100

    return percentage

sample_string = "I am on a call? And I'm speaking! This is cool."





  #testing --- ignore this  
currl = get_sentence_lists(sample_string)
#dick = build_semantic_descriptors(currl)
#erce = run_similarity_test('text2.txt', dick)
print(currl)
