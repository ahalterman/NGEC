
import re

def make_example(text, actor_phrase, nlp):
    doc = nlp(text)
    match = re.search(actor_phrase, text)
    qa = {'text': actor_phrase,
         'qa_score': 0.4408265948295593,
         'qa_start_char': match.span()[0],
         'qa_end_char': match.span()[1]}
    return doc, qa

def test_1(am, nlp):
    # simple split, both actors present in answer span
    text = "Ukrainian forces carried out airstrikes against Russians and Belorussians"
    actor_phrase = "Russians and Belorussians"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Russians", "Belorussians"])

def test_2(am, nlp):
    # amod split, both actors present in answer span
    text = "Ukrainian forces carried out airstrikes against Russian and Belorussian soldiers"
    actor_phrase = "Russian and Belorussian soldiers"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Belorussian soldiers", "Russian soldiers"])

def test_3(am, nlp):
    # simple split, only one actor present in answer span
    text = "Ukrainian forces carried out airstrikes against Russians and Belorussians"
    actor_phrase = "Russians"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Russians", "Belorussians"])

def test_3(am, nlp):
    # simple split, only the second actor present in answer span
    text = "Ukrainian forces carried out airstrikes against Russians and Belorussians"
    actor_phrase = "Belorussians"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Belorussians", "Russians"])

def test_4(am, nlp):
    # amod split, only the second actor present in answer span
    text = "Ukrainian forces carried out airstrikes against Russian and Belorussian soldiers"
    actor_phrase = "Belorussian soldiers"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Russian soldiers", "Belorussian soldiers"])

def test_5(am, nlp):
    # amod, no second actor present in answer span
    text = "Ukrainian forces carried out airstrikes against Russian soldiers"
    actor_phrase = "Russian soldiers"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Russian soldiers"])

def test_6(am, nlp):
    # long list
    text = "Japan, the United States, Australia and India got together in New York in September last year for the first time."
    actor_phrase = "Japan"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Japan', 'United States', 'Australia', 'India'])

def test_7(am, nlp):
    # two actors, full titles
    text = "Russian President Vladimir Putin and British Prime Minister Boris Johnson will meet in Geneva next week."
    actor_phrase = "Vladimir Putin"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Russian President Vladimir Putin', 'British Prime Minister Boris Johnson'])

def test_8(am, nlp):
    text = "U.S. national security adviser Robert O'Brien said Friday he will hold talks with his counterparts from Japan, Australia, and India in Hawaii in October."
    actor_phrase = "Japan"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Australia', 'Japan', 'India'])

def test_9(am, nlp):
    # Checks that we aren't picking up appostive clauses that aren't compound lists
    text = "Og Fernandes, rapporteur of Operation Faroeste, revoked the house arrest of Sandra Inês Rusciolelli, the first judge to sign a plea bargaining agreement in Brazil."
    actor_phrase = "Sandra Inês Rusciolelli"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(["Sandra Inês Rusciolelli"])


def test_10(am, nlp):
    text = "According to a statement published on its website, Putin and Johnson discussed climate issues in light of of the forthcoming UN climate change conference COP26 and leaders ' summit in Glasgow."
    actor_phrase = "Putin"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Putin', 'Johnson'])

def test_11(am, nlp):
    # Actors follow an introductory clause
    text = "According to a statement published on its website, Putin and Johnson discussed climate issues in light of of the forthcoming UN climate change conference COP26 and leaders ' summit in Glasgow."
    actor_phrase = "Putin and Johnson"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Putin', 'Johnson'])

def test_12(am, nlp):
    text = "Qasr al-Nil Misdemeanor Court earlier cleared 28 arrested suspects and another 24 fugitives over accusations of protesting without prior permission."
    actor_phrase = "arrested suspects"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)

def test_13(am, nlp):
    text = "Last month Russia and Turkish Foreign Minister Mevlut Cavusoglu both accused Iran of trying to destabilise Syria and Iraq and of sectarianism, prompting Tehran to summon Ankara's ambassador."
    actor_phrase = "Russia"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Russia', 'Turkish Foreign Minister Mevlut Cavusoglu'])

def test_14(am, nlp):
    text = "\"Dine Ak Diamono\" talk show hosted by Moustapha Diop, with Bassirou Ngom, lawyer and member of the Alliance for the Republic; Barrister Babacar Ba, leader of the civil society organization known as \"Forum du Justiciable;\" Alassane Kitane, teacher of Philosophy at Amary Ndack Seck High School in Thies; and Oumar Faye of the movement Leeral Askanwi [Enlightening People], as guests - live from studio [Diop] Good evening viewers and thank you for your fidelity to the \"Dine Ak Diamono\" talk show."

def test_15(am, nlp):
    text = "The Liberal Party, the largest opposition in Paraguay, announced in the evening of Wednesday the decision to submit an application of impeachment against the president of the country, Mario Abdo Benítez, and vice-president Hugo Velázquez, by polemical agreement with Brazil on the purchase of energy produced in Itaipu."
    actor_phrase = "Velázquez"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)

def test_16(am, nlp):
    text = "The Liberal Party, the largest opposition in Paraguay, announced in the evening of Wednesday the decision to submit an application of impeachment against the president of the country, Mario Abdo Benítez, and vice Hugo Velázquez, by polemical agreement with Brazil on the purchase of energy produced in Itaipu."
    actor_phrase = "president"
    doc, qa = make_example(text, actor_phrase, nlp)
    am.expand_actor(qa, doc)

def test_17(am, nlp):
    # Actors follow an introductory clause
    text = "The leaders of Germany, France, and the UK met in light of the forthcoming UN climate change conference COP26 and leaders ' summit in Glasgow."
    actor_phrase = "UK"
    doc, qa = make_example(text, actor_phrase, nlp)
    actors = am.find_co_actors(qa, doc)
    assert set([i['text'] for i in actors]) == set(['Germany', 'France', 'UK'])
