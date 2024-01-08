from openai import OpenAI
import json
from tqdm import tqdm
import argparse
from ram.utils.openset_utils import openimages_rare_unseen

parser = argparse.ArgumentParser(
            description='Generate LLM tag descriptions for RAM++ open-set recognition')
parser.add_argument('--openai_api_key',
                    default='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')
parser.add_argument('--output_file_path',
                    help='save path of llm tag descriptions',
                    default='custom_tag_descriptions_short.json')


def analyze_tags(client, tag):
    # Generate LLM tag descriptions

    llm_prompts = [ f"Describe concisely what a(n) {tag} looks like:", \
                    f"How can you identify a(n) {tag} concisely?", \
                    f"What does a(n) {tag} look like concisely?",\
                    f"What are the identifying characteristics of a(n) {tag}:", \
                    f"Please provide a concise description of the visual characteristics of {tag}:"]

    results = {}
    result_lines = []

    result_lines.append(f"a photo of a {tag}.")

    for llm_prompt in tqdm(llm_prompts):
        # send message
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "assistant", "content": llm_prompt}],
        max_tokens=77,
        temperature=0.99,
        n=10,
        stop=None)

        # parse the response
        for item in response.choices:
            result_lines.append(item.message.content.strip())
        results[tag] = result_lines
    return results

if __name__ == "__main__":
    args = parser.parse_args()
    # set OpenAI API key
    #categories = openimages_rare_unseen
    #categories = ['plate', 'noodles', 'shrimp', 'chicken', 'vegetable', 'broccoli', 'meatball']
    categories = ['plate', 'noodles', 'shrimp', 'chicken', 'vegetable', 'broccoli', 'meatball']
    client = OpenAI(
                api_key=args.openai_api_key,
               )
    tag_descriptions = []
    for tag in categories:
        result = analyze_tags(client, tag)
        tag_descriptions.append(result)

    output_file_path = args.output_file_path
    with open(output_file_path, 'w') as w:
        json.dump(tag_descriptions, w, indent=3)
