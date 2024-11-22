from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd
import logging


class AffectStudy:
    """
    Class for querying llms to replicate Study 1 in Cultural Variation in Affect Valuation paper

    This is affect_aa.py: replicating the study to assess the LLMs' understanding of Chinese, Japanese, Korean, and Vietnamese individuals' emotions

    ---------------------------------------------------------------------------------------------

    llm: a model type
    Methods
    ___________________

     build_chain():
        Builds a SequentialChain for sentiment extraction.
    generate_concurrently():
        Generates sentiment and summary concurrently for each review in the dataframe.
    """

    def __init__(self, model_name):
        self.sample = 1
        if model_name == ("openai/gpt-3.5-turbo"):
            self.name = "gpt-3.5-turbo"
        if model_name == ("openai/gpt-4o-mini"):
            self.name = "openai/gpt-4o-mini"
        if model_name == ("mistralai/mistral-7b-instruct"):
            self.name = "mistral-7b-instruct"
        if model_name == ("meta-llama/llama-3.1-70b-instruct"):
            self.name = "llama"
        if model_name == ("google/gemma-2-27b-it"):
            self.name = "google/gemma-2-27b-it"
        self.culture = "us"
        self.model = self.__create_model(model_name)
        self.feelings = {
            "enthusiastic": "Having or showing intense and eager enjoyment, interest, or approval.",
            "excited": "Feeling or showing happiness and enthusiasm.",
            "strong (elated)": "Feeling physically or emotionally powerful, combined \
                                with a heightened sense of joy or triumph.",
            "happy": "Feeling pleasure, joy, or contentment; generally positive emotions.",
            "satisfied": "Content or pleased with how things have turned out or with one's situation.",
            "content": "A state of peaceful happiness; satisfied with life or a specific situation.",
            "calm": "Free from agitation or strong emotion; peaceful and composed.",
            "at rest": "A state of physical or mental relaxation without stress or exertion.",
            "relaxed": "Free from tension and anxiety; physically and mentally at ease.",
            "peaceful (serene)": "Free from disturbance; tranquil, with an inner sense of calm.",
            "quiet": "Without loud activity or noise; stillness or subdued calm.",
            "still": "Deeply calm, without movement or agitation.",
            "passive": "Accepting or allowing things to happen without active response or resistance.",
            "dull": "Lacking excitement, energy, or interest; feeling low or uninspired.",
            "sleepy": "Feeling drowsy or ready for sleep; lacking energy or alertness.",
            "sluggish": "Lacking energy, alertness, or speed; slow-moving or inactive.",
            "sad": "Feeling sorrow or unhappiness, often in response to a specific loss or situation.",
            "lonely": "Feeling isolated or disconnected from others, often leading to sadness \
                      or longing for social interaction.",
            "unhappy": "A general state of discontent or misery; not satisfied or pleased.",
            "fearful": "Experiencing fear or anxiety, often in response to perceived danger or uncertainty.",
            "hostile": "Feeling or showing anger, aggression, or opposition toward others or a situation.",
            "nervous": "Anxious or apprehensive, often due to worry or uncertainty about an outcome.",
            "aroused": "A state of heightened excitement, energy, or emotional stimulation.",
            "surprised": "Feeling shocked or amazed by something unexpected.",
            "astonished": "Extremely surprised or amazed, often due to something extraordinary."
        }
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
            model=name,
            #openai_api_key="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",
            openai_api_key="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",
            openai_api_base="https://openrouter.ai/api/v1"
        )
        return model

    def build_chains(self):
        for j in range(1):
            response_schemas = self.__create_schemas(j)
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = self.__create_prompt(j, format_instructions)
            chain = prompt | self.model | output_parser
            if j == 0:
                self.chain0 = chain

    def generate_response(self):
        frames = []
        logging.info('in the response generator')
        for feeling, feeling_description in self.feelings.items():
            logging.info(f'Analyzing feeling: {feeling}')
            for j in range(1):
                if j == 0:
                    ideal, actual = ([] for i in range(2))
                    att_list = ['ideal', 'actual']
                    i = 0
                    attempts = 0
                    max_attempts = 5  # Set a maximum number of attempts per diet
                    while i < self.sample and attempts < max_attempts:
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({
                                "feeling": feeling,
                                "feeling_description": feeling_description})
                            ok = self.__output_validation(response, att_list, 1, 5)
                            if ok:
                                ideal.append(response['ideal'])
                                actual.append(response['actual'])
                                logging.info('here')
                                i += 1
                                logging.info(f'Successfully analyzed {feeling}')
                        except Exception as e:
                            logging.error(f'Error analyzing {feeling}: {str(e)}')
                        attempts += 1
                if attempts == max_attempts:
                    logging.warning(f'Max attempts reached for {feeling}')
                logging.info(f'Building a dataframe for {feeling}')
                df = pd.DataFrame({
                    "ideal": ideal,
                    "actual": actual,
                    "feeling": [feeling] * len(ideal),
                    "model": [self.name] * len(ideal)
                })

            frames.append(df)
        return pd.concat(frames)

    def __create_prompt(self, i, format_instructions):
        if i == 0:
            return PromptTemplate(template=(
                "As a Asian American undergraduate college student \
                who is fluent in English, more oriented to American \
                than Asian culture but still moderately oriented to Asian culture, \
                and has parents of either Chinese, Japanese, Korean, or Vietnamese descent, \
                please provide a comprehensive rating of how often you would like to feel \
                and how often you actually feel each feeling. Some of the feelings are very \
                similar to each other, whereas others are very different from each other. \
                Read each {feeling} and each feeling description {feeling_description} and \
                then RATE how often YOU IDEALLY WOULD LIKE TO HAVE each {feeling} FEELING and \
                how often YOU ACTUALLY HAVE each {feeling} FEELING over the course of a typical week, \
                using the following 5-point scale with labels at 1 (never) to 5 (all the time). \
                1: Never - You never experience this emotion or feeling. \
                2: Rarely - You rarely or almost never experience this emotion or feeling. \
                3: Sometimes - You experience this emotion or feeling occasionally or sometimes. \
                4: Often - You frequently experience this emotion or feeling. \
                5: All the time - You experience this emotion or feeling almost constantly or all the time. \
                {format_instructions}\n"
                ), input_variables=['feeling', 'feeling_description'],
                partial_variables={'format_instructions': format_instructions})

    def __create_schemas(self, i):
        if i == 0:
            response_schema = [ResponseSchema(name="ideal",
                                              description="An numerical decimal rating from 1 to 5 of how \
                                            often you would ideally like to feel this feeling"),
                               ResponseSchema(name="actual",
                                              description="An numerical decimal rating from 1 to 5 of how \
                                              often you would actually like to feel this feeling")]
            return response_schema

    def __output_validation(self, response, att_list, lower, upper):
        for measure in att_list:
            try:
                value = float(response[measure])
                if value > upper or value < lower:
                    logging.warning(f'Invalid value for {measure}: {value}')
                    return False
            except ValueError:
                logging.warning(f'Non-numeric value for {measure}: {response[measure]}')
                return False
        return True


# choose an LLM to explore by uncommenting a line with name=(<LLM name>)
#name=("openai/gpt-3.5-turbo")
#name = ("openai/gpt-4o-mini")
name=("mistralai/mistral-7b-instruct")
#name=("meta-llama/llama-3.1-70b-instruct")
#name=("google/gemma-2-27b-it")


analyzer = AffectStudy(name)

if name == ("openai/gpt-3.5-turbo"):
    fname = "gpt3.5"
if name == ("openai/gpt-4o-mini"):
    fname = "gpt4"
if name == ("mistralai/mistral-7b-instruct"):
    fname = "mistral"
if name == ("meta-llama/llama-3.1-70b-instruct"):
    fname = "llama"
if name == ("google/gemma-2-27b-it"):
    fname = "gemma"


logging.basicConfig(filename=fname + "_us_affect_study_aa_final.log", encoding='utf-8', level=logging.DEBUG)
for i in range(1):  # choose n for how many times to run this survey
    df = analyzer.generate_response()
    df.to_csv("results/final/" + fname + "/us_affect_study_final_aa" + fname + str(63+i))