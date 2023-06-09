# type: ignore
import datasets
from helicone import openai

system_prompt = """Given a piece of text, and a target within the text, return a polarity score for the sentiment of the text in the context of that which targets our entity. For these purposes, lean towards 0 when in doubt and always use either -1, 0 or 1.

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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
	        cache=True,
        )
        

        output_text = response["choices"][0]["message"]["content"]
        splits = output_text.split("```")
        
        # Fucked up !?
        if(len(splits) < 2):
            row["gpt4_polarity"] = None
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
    # catch exceptions
    except:
        row["gpt4_polarity"] = None
        return row

dataset = datasets.load_dataset("fhamborg/news_sentiment_newsmtsc")

# first_data_entry = data['test'][0]
# print(first_data_entry)

test_data = dataset["test"].select(range(50))
test_data_with_gpt4 = test_data.map(determine_polarity, batched=False, num_proc=10)

error_count = 0
missed = 0

for row in test_data_with_gpt4:
    target = row["mention"]
    sentence = row["sentence"]
    real_polarity = row["polarity"]
    gpt4_polarity = row["gpt4_polarity"]
    if(gpt4_polarity == None):
        print("openai error")
        error_count += 1
        continue
    
    if(gpt4_polarity != real_polarity):
        missed += 1

non_error_count = len(test_data_with_gpt4) - error_count

print(f"Errors: {error_count}")
print("Missed: " + str(missed))
print(f"Accuracy: {1 - (missed / non_error_count)}")
