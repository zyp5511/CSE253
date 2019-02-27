import numpy as np

from collections import defaultdict

def string_to_index_list(s, char_to_index, end_token):
    """Converts a sentence into a list of indexes (for each character).
    """
    return [char_to_index[char] for char in s] # Adds the end token to each index list

def getIndexList(data,char_to_index):
    a = data.split("<end>\n<start>")
    for i in range(len(a)):
        if i == 0:
            a[i]  = a[i] + '<end>'
            continue
        if i == (len(a)-1):
            a[i]  = '<start>' + a[i] 
            continue
        a[i]  = '<start>' + a[i] +'<end>'
   
    result = []
    for a_i in a:
        result.append([char_to_index[c] for c in a_i])
    return result

def load_data(filename, idx_dict = None):
   
    data = open(filename).read()
    vocab_size = 0
    if idx_dict == None:
    #mapping character to index
        char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
    
        print("Number of unique characters in our whole tunes database = {}".format(len(char_to_index))) 
    
        index_to_char = {i: ch for (ch, i) in char_to_index.items()}

    #all_characters = np.asarray([char_to_index[c] for c in data], dtype = np.int32)
    #print("Total number of characters = "+str(all_characters.shape[0]))

        vocab_size = len(char_to_index) + 1
        end_token = vocab_size
        char_to_index['EOS'] = end_token
        index_to_char[93] = 'EOS'    
        idx_dict = { 'char_to_index': char_to_index,
                 'index_to_char': index_to_char,
                 'end_token': end_token}

    index_list = getIndexList(data,idx_dict['char_to_index'])
    #return all_characters , vocab_size, idx_dict
    return index_list, vocab_size,idx_dict
