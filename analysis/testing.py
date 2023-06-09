import datasets
import openai

system_prompt = '''Given a piece of text, and a target within the text, return a polarity score for the sentiment of the text in the context of that which targets our entity. For these purposes, lean towards neutral when in doubt and only use -1, 0 or 1.

Explain your thinking with a couple sentences, then output the polarity number in a code block like this:
```
// polarity score here
```'''

def build_user_prompt(sentence: str, target: str):
    prompt = "<text>\n" + sentence + "\n</text>\n<target>\n" + target + "\n</target>"
    return prompt

def determine_polarity(sentence: str, target: str):

    user_prompt = build_user_prompt(sentence, target)

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    temperature = 0
    )

    output_text = response['choices'][0]['message']['content']
    token = output_text.split("```")
    polarity = token[2]

    return int(polarity)

data = datasets.load_dataset("fhamborg/news_sentiment_newsmtsc")

#first_data_entry = data['test'][0]
#print(first_data_entry)

real_polarity_list = []
gpt4_polarity_list = []

for row in data['test']:
    print('row is...', row)
    target = row['mention']
    sentence = row['sentence']
    real_polarity = row['polarity']
    gpt4_polarity = determine_polarity(sentence, target)

    print(real_polarity, gpt4_polarity)

    real_polarity_list.append(real_polarity)
    gpt4_polarity_list.append(gpt4_polarity)




