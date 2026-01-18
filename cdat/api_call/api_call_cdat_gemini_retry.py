"""
Code adapted for the CDAT experiments from: https://github.com/AntoineBellemare/DAT_GPT/tree/main
Gemini responses occasionally arrive slowly or the request times out before the output is fully available. 
The script therefore added an explicit retry mechanism and
retries failed calls (up to a fixed maximum number of attempts), waiting progressively longer
between attempts, and logs both successful responses and errors for each cue.
"""

import time
import json
import click
import logging
import warnings
warnings.filterwarnings('ignore')
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Fixed cues
with open('../cdat_cue_filtered550.txt', 'r') as f:
# keys
api_key = "" # Add your key here
client = genai.Client(api_key=api_key)

def generate_response(temp, model_name, cue, timeout=180):
    
    config_args = {
        'temperature': temp
    }

    # Set model-specific max_output_tokens to allow sufficiently long responses.
    # We use higher limits for models/configurations expected to produce longer answers.
    if model_name == "gemini-1.5-pro":
        config_args['max_output_tokens'] = 180 

    if model_name == "gemini-2.0-flash":
        config_args['max_output_tokens'] = 300

    text = f'Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, yet semantically associated with the following cue word: “{cue}.” Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialized vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list.'
    print(f"Sending request: model={model_name}, temp={temp}, cue={cue}") 
    
    def call_api():
        return client.models.generate_content(
                                    model=model_name,
                                    contents=text,
                                    config=types.GenerateContentConfig(**config_args)
                                    ).text
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_api)
            return future.result(timeout=timeout)

    except FuturesTimeout:
        print(f"Timeout: API call took more than {timeout} seconds.")
        return f"Error: API timeout after {timeout} seconds."

@click.command()
@click.option("--temp", type=float)
@click.option("--file_path", type=str)
@click.option('--model', default='gemini-2.0-flash', help='Model name to use')
@click.option("--iter_nb", type=str)
def main(temp=0, file_path="./", model="gemini-2.0-flash", iter_nb='0'):
    """
    file_path: path, optional
        Path to save the output file.
    iter_nb: str
        Identifier for the iteration; used to avoid overwriting previous runs.
        Outputs are stored as "<model>_temp<temp>_cdat<iter_nb>.json".
    """
    logger = logging.getLogger(__name__)
    output = {}

    # Retry configuration:
    # - MAX_RETRIES: maximum number of times to retry a failed API call per cue.
    # - BACKOFF_SECONDS: initial wait time before the first retry.
    # - BACKOFF_MULTIPLIER: factor by which the wait time increases after each failed attempt
    #   (exponential backoff: 10s, 20s, 40s, ...).
    MAX_RETRIES = 6
    BACKOFF_SECONDS = 10
    BACKOFF_MULTIPLIER = 2

    for iterat, cue in enumerate(cues):
        logger.info(f"API CALL NUMBER {iterat} \n{'~'*80}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = generate_response(temp, model, cue, timeout=180)
                if not response:
                    raise Exception("API call returned empty or None.")

                # Add the cue and response to each iteration of the output dictionary
                logger.info(f"Cue: {cue}\n{response}")
                output.update({iterat: {"cue": cue, "response": response}})

                with open(f"{file_path}{model}_temp{temp}_cdat{iter_nb}.json", "w") as outfile:
                    json.dump(output, outfile)

                break  # success → exit retry loop

            except Exception as e:
                logger.error(f"Attempt {attempt}/{MAX_RETRIES} failed for cue {iterat}: {e}")
                print(f"API Call Error (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    sleep = BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    logger.info(f"Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                else:
                    logger.error(f"Max retries reached for cue {cue}. Moving on.")
                    output.update({iterat: {"cue": cue, "response": str(e)}})

    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    # NOTE: from command line ``
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()