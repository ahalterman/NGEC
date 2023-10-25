from transformers import pipeline, set_seed
import re
from tqdm import tqdm
import random
import pandas as pd
import datetime

text_generator = pipeline('text-generation', model='gpt2-xl', device=0)
#set_seed(42)

prompt = "Thousands of Soldiers Deployed to Czech Border to Address Unfolding Crisis\n\n(BBC Monitoring) --"
text_generator(prompt,  max_length=300)

cities = ["Abuja", "Kabul", "Belgrade", "Zaghreb", "Khartoum", "Vienna", "Dhaka", "Brussels", 
          "Minsk", "Kinshasa", "Beijing", "Bogota", "Sao Paulo", "Havana", "Berlin", "Prague",
          "Moscow", "Washington", "Cairo", "Jerusalem", "Dehli", "Tehran", "Rome", "Amman", 
          "Beirut", "Tokyo", "Nairobi", "New York", "Panama City", "Oslo", "Damascus",
          "Bangkok", "Istanbul", "London", "Abu Dhabi"]

c_df = pd.read_csv("countries.csv")
countries = c_df['Name'].to_list()

def make_stories(prompt, source, pattern, max_len=100, n=5):
    output = text_generator(prompt, 
                            max_length=max_len, 
                            num_return_sequences=n,
                            pad_token_id=50256
                            )                   
    selected = []
    for out in output:
        out['text'] = re.sub(re.escape(prompt), "", out['generated_text'])
        #toks = set([i.lower() for i in out['text'].split(" ")])
        selected.append(out)
    final = []
    for i in selected:
        disclaimer = "### THIS IS A SYNTHETIC STORY. DO NOT TRUST THE FACTUAL CONTENT OF THIS TEXT. Created by Andy Halterman to train a document-level political event classifer ###"
        text = disclaimer + i['text'].strip()
        d = {"text": text,
            "title": pattern['title'],
            "source": source,
            "prompt": prompt,
            "label": pattern['event'],
            "mode": pattern['mode']}
        final.append(d)
    return final

def make_prompt_and_gen(pattern, 
                  source, 
                  max_len=100, 
                  unique_prompts=5, 
                  n_per_city=5):
    all_stories = []
    for n in range(unique_prompts):
        city = random.sample(cities, 1)[0]
        country_1, country_2, country_3 = random.sample(countries, 3)
        headline = pattern['title'].format(country_1=country_1, 
                                              country_2=country_2,
                                              country_3=country_3,
                                              city=city)
        prompt = f"{headline}\n\n({source}) --"
        stories = make_stories(prompt, source, pattern, n=n_per_city, max_len=max_len) 
        all_stories.extend(stories)
    return all_stories


def run():
    patterns = pd.read_csv("synthetic_headlines.csv")
    patterns = patterns.sample(frac=1)

    all_output = []
    for n, pattern in tqdm(patterns.iterrows(), total=patterns.shape[0]):
        if not pattern['title']:
            continue
        print(pattern['event'])
        for source in ['Reuters', 'AFP', 'BBC Monitoring', 'AP', 'local sources', 'local media']:
            out = make_prompt_and_gen(pattern, source, max_len=300, unique_prompts=2, n_per_city=1)
            all_output.extend(out)
        #except Exception as e:
        #    print(e)
    df = pd.DataFrame(all_output)
    #df.to_csv("gpt_synthetic_events_cities.csv")
    today = datetime.datetime.today().strftime('%Y-%m-%d_%H')
    df.to_csv(f"gpt_synthetic_events_{today}.csv")


if __name__ == "__main__":
    run()
