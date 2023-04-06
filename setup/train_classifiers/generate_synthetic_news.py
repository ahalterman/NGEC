from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import re
from tqdm import tqdm
import random
import pandas as pd
import datetime
import jsonlines

qs = pd.read_csv("headlines_event_mode.csv")

# get count of unique events
qs['event'].value_counts()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('../huggingface_models/gpt2-large', device=0)
model = GPT2LMHeadModel.from_pretrained('../huggingface_models/gpt2-large',
                        pad_token_id=tokenizer.eos_token_id
model.to(device)



def prompt_to_output(prompt, 
                     max_len=250,
                     top_p=0.92,
                     temperature=0.8,
                     top_k=0):
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    output = model.generate(encoded_input['input_ids'], 
                            max_length=max_len, 
                            num_return_sequences=n,
                            do_sample=True, 
                            top_p=top_p, 
                            temperature=temperature,
                            top_k=top_k)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt_to_output(prompt, top_p = 0.99)


cities = ["Abuja", "Kabul", "Belgrade", "Zaghreb", "Khartoum", "Vienna", "Dhaka", "Brussels", 
          "Minsk", "Kinshasa", "Beijing", "Bogota", "Sao Paulo", "Havana", "Berlin", "Prague",
          "Moscow", "Washington", "Cairo", "Jerusalem", "Dehli", "Tehran", "Rome", "Amman", 
          "Beirut", "Tokyo", "Nairobi", "New York", "Panama City", "Oslo", "Damascus",
          "Bangkok", "Istanbul", "London", "Abu Dhabi"]

c_df = pd.read_csv("countries.csv")
countries = c_df['Name'].to_list()
news_sources = ["Reuters", "AFP", "(local sources)"]

headline_row = qs.iloc[111]

headline_list = qs.to_dict('records')

def make_stories(headline_row, 
                  max_len=250, 
                  unique_prompts=5,
                  top_p=0.92,
                  temperature=0.8,
                  top_k=0):
    all_stories = []
    for n in range(unique_prompts):
        headline_row_copy = headline_row.copy()
        city = random.sample(cities, 1)[0]
        country_1, country_2, country_3 = random.sample(countries, 3)
        try: 
            headline = headline_row['title'].format(country_1=country_1, 
                                                  country_2=country_2,
                                                  country_3=country_3,
                                                  city=city)
        except KeyError:
            # a couple of the headlines are incorrectly formatted
            headline = headline_row['title'].format(country=country_1, city=city)
        source = random.choice(news_sources)
        prompt = f"{headline}\n\n({source}) -"
        headline_row_copy['prompt'] = prompt
        story = prompt_to_output(prompt,
                                 max_len=max_len,
                                 top_p=top_p,
                                 temperature=temperature,
                                 top_k=top_k)
        story = re.sub(re.escape(prompt), "", story).strip()
        disclaimer = "### THIS IS A SYNTHETIC STORY. DO NOT TRUST THE FACTUAL CONTENT OF THIS TEXT. Created by Andy Halterman to train a document-level political event classifer ###"
        story = disclaimer + story
        headline_row_copy['text'] = story
        headline_row_copy['top_p'] = top_p
        headline_row_copy['temperature'] = temperature
        headline_row_copy['top_k'] = top_k
        all_stories.append(headline_row_copy)
    return all_stories

make_stories(headline_list[-3])

#all_output = []
for headline_row in tqdm(headline_list[143:]):
    if not headline_row['title']:
        continue
    output = make_stories(headline_row)
    all_output.extend(output)

today = datetime.date.today().strftime("%Y-%m-%d")
with jsonlines.open('gpt_synthetic_events_{today}.jsonl', mode='w') as f:
    f.write_all(all_output)

if __name__ == "__main__":
    pass
