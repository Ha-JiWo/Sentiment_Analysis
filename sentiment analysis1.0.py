positive_words = {
    
}

negative_words = {
    'bad','upset','lonely','depressed','miserable','gloomy','heartbroken','sorrowful','hopeless','helpless','anxious',
    'nervous','worried','tense','panicked','frightened','scared','terrified','horrified','uneasy','angry','mad','furious','irritated','annoyed','bitter','resentful'
    ,'jealous','envious','hostile','frustrated','disappointed','discouraged','confused','lost','tired','exhausted','bored','dull','cranky','moody','selfish','arrogant',
    'insecure','guilty','ashamed','embarrassed','desperate','repulsive','vile'
}

def analyze_sentiment_en(text):
    words=text.lower().split()
    score=0
    i=0
    n=len(words)
    
    while i<n:
        word=words[i]
        if word in positive_words:
            score+=-1 if negated else 1
        elif word in negative_words:
            score+=1 if negated else -1
        i+=1
    
    if score>1:
        return"Positive"
    elif score<-1:
        return"Negative"
    else:
        return"Neutral"

if __name__=="__main__":
    print("Enter an English product review (type 'STOP' to quit):")
    while True:
        comment=input("> ")
        if comment.strip().upper()=="STOP":
            print("Goodbye!")
            break
        sentiment=analyze_sentiment_en(comment)
        print(f"Sentiment: {sentiment}\n")
