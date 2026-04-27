positive_words = {
    'happy', 'joyful', 'delighted', 'elated', 'ecstatic', 'thrilled', 'overjoyed', 'jubilant', 'gleeful', 'merry', 'jolly', 'cheerful', 'upbeat', 'buoyant', 'radiant',
    'sunny', 'lighthearted', 'blithe', 'blissful', 'euphoric', 'exhilarated', 'jovial', 'mirthful', 'exultant', 'rapturous', 'chirpy', 'jaunty', 'perky', 'content',
    'satisfied', 'fulfilled', 'serene', 'tranquil', 'peaceful', 'calm', 'relaxed', 'at ease', 'placid', 'mellow', 'restful', 'comfortable', 'untroubled', 'undisturbed',
    'composed', 'harmonious', 'centered', 'easygoing', 'grateful', 'thankful', 'appreciative', 'indebted', 'moved', 'touched', 'overwhelmed', 'beholden', 'loving', 'affectionate',
    'warm', 'tender', 'caring', 'compassionate', 'kind-hearted', 'fond', 'adoring', 'devoted', 'romantic', 'amorous', 'doting', 'sympathetic', 'empathetic', 'gentle',
    'sweet', 'hopeful', 'optimistic', 'sanguine', 'positive', 'confident', 'encouraged', 'reassured', 'expectant', 'bright', 'rosy', 'promising', 'bullish', 'forward-looking',
    'enthusiastic', 'eager', 'passionate', 'zealous', 'fervent', 'ardent', 'excited', 'animated', 'energetic', 'lively', 'vibrant', 'spirited', 'vivacious', 'dynamic',
    'pumped', 'raring', 'fired up', 'proud', 'confident', 'self-assured', 'empowered', 'accomplished', 'dignified', 'triumphant', 'fulfilled', 'assertive', 'assured',
    'interested', 'curious', 'fascinated', 'intrigued', 'engrossed', 'captivated', 'absorbed', 'spellbound', 'riveted', 'inquisitive', 'engaged', 'admiring', 'awed',
    'amazed', 'astonished', 'wonderstruck', 'reverent', 'impressed', 'dazzled', 'speechless', 'marveling', 'amused', 'entertained', 'delighted', 'playful', 'mischievous',
    'whimsical', 'tickled', 'humorous', 'witty', 'light-spirited', 'brave', 'courageous', 'determined', 'resolute', 'resilient', 'steadfast', 'undaunted', 'bold', 'valiant',
    'tenacious', 'gritty', 'happiness', 'joy', 'delight', 'bliss', 'ecstasy', 'euphoria', 'elation', 'jubilation', 'glee', 'merriment', 'cheerfulness', 'contentment',
    'satisfaction', 'fulfillment', 'serenity', 'peace', 'tranquility', 'love', 'affection', 'warmth', 'tenderness', 'compassion', 'gratitude', 'thankfulness', 'appreciation',
    'hope', 'optimism', 'enthusiasm', 'excitement', 'passion', 'zeal', 'fervor', 'pride', 'confidence', 'amusement', 'wonder', 'awe', 'inspiration', 'motivation', 'relief',
    'exhilaration', 'radiance', 'lightheartedness', 'gladness', 'joviality', 'exuberance', 'buoyancy'
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
