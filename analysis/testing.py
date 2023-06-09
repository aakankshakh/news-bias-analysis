# type: ignore
import datasets
import openai

system_prompt = """Given a piece of text, and a target within the text, return a polarity score for the sentiment of the text in the context of that which targets our entity. For these purposes, lean towards neutral when in doubt and only use -1, 0 or 1.

Explain your thinking with a couple sentences, then output the polarity number in a code block like this:
```
polarity
```"""


def build_user_prompt(sentence: str, target: str):
    prompt = "<text>\n" + sentence + "\n</text>\n<target>\n" + target + "\n</target>"
    return prompt


def determine_polarity(row):
    target = row["mention"]
    sentence = row["sentence"]

    user_prompt = build_user_prompt(sentence, target)

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    output_text = response["choices"][0]["message"]["content"]
    splits = output_text.split("```")
    
    # Fucked up !?
    if(len(splits) < 2):
        return row
    
    middle_part = splits[1]
    polarity = middle_part.strip()
    # round to nearest integer, ie -0.5 to -1 and 0.5 to 1 and 0.1 to 0
    pol_float = float(polarity)
    if pol_float < -0.3:
        polarity = -1
    elif pol_float > 0.3:
        polarity = 1
    else:
        polarity = 0

    row["gpt4_polarity"] = polarity
    return row

dataset = datasets.load_dataset("fhamborg/news_sentiment_newsmtsc")

# first_data_entry = data['test'][0]
# print(first_data_entry)

test_data = dataset["test"].select(range(30))
test_data_with_gpt4 = test_data.map(determine_polarity, batched=False, num_proc=10)

real_polarity_list = []
gpt4_polarity_list = []

for row in test_data_with_gpt4:
    target = row["mention"]
    sentence = row["sentence"]
    real_polarity = row["polarity"]
    if("gpt4_polarity" in row):
        gpt4_polarity = row["gpt4_polarity"]
        real_polarity_list.append(real_polarity)
        gpt4_polarity_list.append(gpt4_polarity)

missed = 0
for idx in range(len(real_polarity_list)):
    if not real_polarity_list[idx] == gpt4_polarity_list[idx]:
        missed += 1

print(f"Missed: {missed}")
print(f"Total: {len(real_polarity_list)}")
print(f"Accuracy: {1 - missed/len(real_polarity_list)}")
