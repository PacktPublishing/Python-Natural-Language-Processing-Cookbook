from nltk.chat.util import Chat, reflections
from nltk.chat.eliza import eliza_chat

def main():
    pairs = [[r".*hi|hello|hey|what's up.*", ["Hello, I am a simple chatbot. How are you?"]],
             [r'.*(how are you|and you).*', ["Doing well, thank you.", "I'm ok, what's new?"]],
             [r"quit", ["Bye, nice talking to you!", "Talk to you later!"]],
             [r".*weather|rain|snow|sun.*", ["Isn't it nice out today?", "Hopefully, tomorrow the weather will get better.", "I don't know about you, but I am glad I am inside today."]],
             [r"I like (.*)", ["Why do you like %1?", "What about %1 do you like?"]],
             [r".*(tasty).*", ["Yeah, I think so too."]],
             [r".*", ["I am sorry, I don't understand. I am a very simple chatbot!"]]
            ]    
    chatbot = Chat(pairs, reflections)
    chatbot.converse()

def eliza_chatbot():
    eliza_chat()

if (__name__ == "__main__"):
    #main()
    eliza_chatbot()
