positive_words = {
    
}

negative_words = {
    
}

def analyze_sentiment_en(text):
    words=text.lower().split()
    score=0
    i=0
    n=len(words)
    
    while i<n:
        word=words[i]
        if word in positive_words:
            negated=(i>0 and words[i-1] in negation_words)
            score+=-1 if negated else 1
        elif word in negative_words:
            negated=(i>0 and words[i-1] in negation_words)
            score+=1 if negated else -1
        i+=1
    
    if score>0:
        return"Positive"
    elif score<0:
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
