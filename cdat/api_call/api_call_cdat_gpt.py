"""
Code adapted for the CDAT experiments from: 
https://github.com/AntoineBellemare/DAT_GPT/tree/main
"""

from openai import OpenAI
import time
import json
import click
import logging
import warnings
warnings.filterwarnings('ignore')

# Fixed cues
with open('../cdat_cues_filtered550.txt', 'r') as f:
    cues = f.read().splitlines() 

# key
client = OpenAI(
    api_key = "" # Add your key here
)

def generate_response(temp, model_name, cue):
    text = f'Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, yet semantically associated with the following cue word: “{cue}.” Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list.'
    print(f"Sending request: model={model_name}, temp={temp}, cue={cue}") 
    response = client.chat.completions.create(model=model_name, messages=[{"role":'assistant', "content":text}], temperature=temp)
    return response.choices[0].message.content.strip()

@click.command()
@click.option("--temp", type=float)
@click.option("--file_path", type=str)
@click.option("--iter_nb", type=str)
@click.option('--model', default='gpt-4', help='Model name to use')
def main(temp=0, file_path="./", iter_nb='0', model='gpt-4'):
    """
    file_path: path, optional
        Path to save the output file.
    iter_nb: str
        Identifier for the iteration; used to avoid overwriting previous runs.
        Outputs are stored as "<model>_temp<temp>_cdat<iter_nb>.json".
    """

    logger = logging.getLogger(__name__)
    output = {}

    for iterat, cue in enumerate(cues):
        logger.info(f"API CALL NUMBER {iterat} \n{'~'*80}")

        try:
            response = generate_response(temp, model, cue)
            if response is None:
                raise Exception("API call failed.")
            
            # Add the cue and response to each iteration of the output dictionary
            logger.info(f"Cue: {cue}\n{response}")
            output.update({iterat: {"cue": cue, "response": response}})

            with open(f"{file_path}{model}_temp{temp}_cdat{iter_nb}.json", "w") as outfile:
                json.dump(output, outfile)
        
        except Exception as e:
            logger.error(f"API CALL FAILED: {e}")
            print(f"API Call Error: {e}")
            time.sleep(10) 
            continue
        
        time.sleep(3)

    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    # NOTE: from command line ``
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()