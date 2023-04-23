import gensim, os, pandas, numpy, pickle, re, datetime, glob, time
import gensim.downloader

w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')        # word2vec model for vectors extraction
output_filename = r'D:\Download\method2_test_features.csv'

# prepping for Lemmatization
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()   # WordNetLemmatizer object

# database of keywords
keywords=['accept',
'access',
'act',
'alert',
'amazing',
'apply',
'banking',
'bargain',
'beneficiary',
'best',
'big',
'bonus',
'buy',
'call',
'cancel',
'card',
'cash',
'casino',
'certified',
'cheap',
'clearance',
'click',
'collect',
'confirmation',
'credit',
'customer',
'deal',
'debt',
'direct',
'double',
'earn',
'easy',
'exclusive',
'expires',
'extra',
'fantastic',
'fast',
'financial',
'free',
'girls',
'growth',
'guarantee',
'hesitate',
'hurry',
'immediately',
'important',
'increase',
'incredible',
'instant',
'insurance',
'investment',
'invoice',
'join',
'last',
'legal',
'lifetime',
'limited',
'lose',
'lottery',
'luxury',
'money',
'nigerian',
'offer',
'online',
'opportunity',
'order',
'price',
'prize',
'profit',
'promotion',
'refund',
'request',
'required',
'risk',
'free',
'satisfaction',
'save',
'score',
'sex',
'special',
'statement',
'trial',
'unsecured',
'unsolicited',
'urgent',
'vacation',
'verification',
'very',
'warranty',
'win',
'won']

common_words=["the", "and", "that", "where", "for"]         # words NOT to process


####################################

# os.chdir(r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4\NLP trained models")

x = datetime.datetime.now()
print(f"Starting per-email operations at {x.strftime('%H:%M:%S')}")
start_time = time.process_time()
# Reading files separately
os.chdir(r"D:\Download\Datasets")
email_filenames = glob.glob(r'Testing-mails\*')                     # read filenames of all/any files from folder

    
# Processing individual emails now
features=numpy.zeros((len(email_filenames) , len(keywords)))
for email_index, entry in enumerate(email_filenames):
    # print(f"Working on email: {email_index+1}")
    f=open(entry, encoding='utf-8', errors='ignore')
    email_raw=f.readlines()
    f.close()
    words=numpy.ndarray([0])
    selected_words_and_cosines=[]                      # all vector predictions
    words=[]                                            # all words in 1 email
    
    #   Individual Lines
    for line_index, line in enumerate(email_raw):
        clean_line_words=[]                             # all words from 1 line
        # print(f"Line: {line_index}, length: {len(line)}")
        if "From :" not in line and "Date :" not in line and "To :" not in line and "Subject :" not in line and "Attachment :" not in line and len(line)>2:
            raw_words = line.split(' ')
            for word in raw_words:
                clean_word = ''.join(filter(str.isalpha, word))
                if len(clean_word)>2:
                    clean_line_words.append(clean_word.lower())
            
            if len(clean_line_words)>=2:
                words=numpy.append(words, clean_line_words).reshape(1,-1).tolist()          
                
    # Have all the words of a given email at this stage, processing 1 by 1
    words_lemmatized=[]                        # all words in 1 email, after cleaning
    words_and_vector=[]                      # all vector predictions
    try:    
        for word in words[0]:
            # word = word.encode('ascii', 'ignore').decode('ascii')
            word=re.sub('\d','', " ".join(word))
            word=re.sub('[\s]+','', word)           # remove extra whitespaces
            
            if len(word)>2 and word not in common_words:
                words_lemmatized.append(wnl.lemmatize(word))        # lemmatizing and adding a word to list
                for j,keyword in enumerate(keywords):
                    cosine = w2v_model.similarity(keyword, wnl.lemmatize(word))     # cosine similarity between word in email vs keyword (also lemmatizing)              
                    # if cosine>0.4:
                    words_and_vector.append([word, keyword, cosine])                  # adding vector
                    # print(f"Word: {word}, cosine to {keyword} is {cosine}")
            # except:
            #     # print(f"Word: {word} not found in vocabulary. From email {i}")
            #     pass
    except:
         # print(f"Email {email_index} is too short. Filename: {entry}")
         # Word not in vocabulary (typo or rare word)
         words_and_vector.append([word, keyword, 0])
         pass
         
    # feature extraction (building row of features for each email)
    all_cosines=[]
    try:
        words_and_vector=numpy.array(words_and_vector)
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='accept'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='access'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='act'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='alert'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='amazing'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='apply'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='banking'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='bargain'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='beneficiary'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='best'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='big'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='bonus'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='buy'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='call'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='cancel'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='card'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='cash'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='casino'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='certified'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='cheap'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='clearance'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='click'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='collect'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='confirmation'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='credit'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='customer'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='deal'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='debt'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='direct'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='double'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='earn'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='easy'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='exclusive'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='expires'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='extra'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='fantastic'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='fast'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='financial'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='free'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='girls'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='growth'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='guarantee'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='hesitate'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='hurry'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='immediately'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='important'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='increase'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='incredible'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='instant'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='insurance'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='investment'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='invoice'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='join'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='last'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='legal'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='lifetime'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='limited'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='lose'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='lottery'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='luxury'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='money'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='nigerian'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='offer'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='online'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='opportunity'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='order'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='price'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='prize'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='profit'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='promotion'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='refund'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='request'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='required'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='risk'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='free'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='satisfaction'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='save'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='score'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='sex'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='special'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='statement'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='trial'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='unsecured'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='unsolicited'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='urgent'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='vacation'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='verification'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='very'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='warranty'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='win'))][2].reshape(1,-1))
        all_cosines.append(words_and_vector[numpy.max(numpy.where(words_and_vector[:,1]=='won'))][2].reshape(1,-1))
        
        # building matrix for each email
        all_cosines = numpy.array(all_cosines).reshape(1,-1)
        features[email_index] = all_cosines 
    
    except:
        # print(f"Problem with email: {email_index}. Length: {len(email_raw)}, line: {line_index}")
        # print(email_raw)
        features[email_index] = numpy.zeros(len(keywords))
         
# Saving extraceted features to disc	
pandas.DataFrame(features).to_csv(output_filename)

x = datetime.datetime.now()
print(f"Finished successfully at {x.strftime('%H:%M:%S')}, operation took {(time.process_time() - start_time) / 60} minutes")