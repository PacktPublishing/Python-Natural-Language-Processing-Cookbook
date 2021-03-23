import spacy

nlp = spacy.load("en_core_web_sm")

article = """iPhone 12: Apple makes jump to 5G
Apple has confirmed its iPhone 12 handsets will be its first to work on faster 5G networks. 
The company has also extended the range to include a new "Mini" model that has a smaller 5.4in screen. 
The US firm bucked a wider industry downturn by increasing its handset sales over the past year. 
But some experts say the new features give Apple its best opportunity for growth since 2014, when it revamped its line-up with the iPhone 6. 
"5G will bring a new level of performance for downloads and uploads, higher quality video streaming, more responsive gaming, 
real-time interactivity and so much more," said chief executive Tim Cook. 
There has also been a cosmetic refresh this time round, with the sides of the devices getting sharper, flatter edges. 
The higher-end iPhone 12 Pro models also get bigger screens than before and a new sensor to help with low-light photography. 
However, for the first time none of the devices will be bundled with headphones or a charger. 
Apple said the move was to help reduce its impact on the environment. "Tim Cook [has] the stage set for a super-cycle 5G product release," 
commented Dan Ives, an analyst at Wedbush Securities. 
He added that about 40% of the 950 million iPhones in use had not been upgraded in at least three-and-a-half years, presenting a "once-in-a-decade" opportunity. 
In theory, the Mini could dent Apple's earnings by encouraging the public to buy a product on which it makes a smaller profit than the other phones. 
But one expert thought that unlikely. 
"Apple successfully launched the iPhone SE in April by introducing it at a lower price point without cannibalising sales of the iPhone 11 series," noted Marta Pinto from IDC. 
"There are customers out there who want a smaller, cheaper phone, so this is a proven formula that takes into account market trends." 
The iPhone is already the bestselling smartphone brand in the UK and the second-most popular in the world in terms of market share. 
If forecasts of pent up demand are correct, it could prompt a battle between network operators, as customers become more likely to switch. 
"Networks are going to have to offer eye-wateringly attractive deals, and the way they're going to do that is on great tariffs and attractive trade-in deals," 
predicted Ben Wood from the consultancy CCS Insight. Apple typically unveils its new iPhones in September, but opted for a later date this year. 
It has not said why, but it was widely speculated to be related to disruption caused by the coronavirus pandemic. The firm's shares ended the day 2.7% lower. 
This has been linked to reports that several Chinese internet platforms opted not to carry the livestream, 
although it was still widely viewed and commented on via the social media network Sina Weibo."""

def main():
    doc = nlp(article)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


if (__name__ == "__main__"):
    main()

