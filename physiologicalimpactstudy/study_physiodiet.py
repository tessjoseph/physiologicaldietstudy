from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd
import logging

class DietAnalyzer:
    """
    Class for querying LLMs for physiological measures based on different food diets
    Attributes
    ----------
    llm : model type
    Methods
    -------
    build_chains():
        Builds a SequentialChain for physiological measure extraction.
    generate_response():
        Generates physiological measures for each diet in the dataframe.
    """

    def __init__(self, model_name):
        self.sample = 1
        if model_name == ("openai/gpt-3.5-turbo"):
            self.name = "gpt-3.5-turbo"
        if model_name == ("openai/chatgpt-4o-latest"):
            self.name = "openai/chatgpt-4o-latest"
        if model_name == ("mistralai/mistral-7b-instruct"):
            self.name = "mistral-7b-instruct"
        if model_name == ("google/gemma-2-27b-it"):
            self.name = "google/gemma-2-27b-it"
        if model_name == ("meta-llama/llama-2-70b-chat"):
            self.name = "llama-2-70b-chat"
        self.culture = "us"
        self.model = self.__create_model(model_name)

        self.diets = {
            "Middle Eastern Diet": "Characterized by the use of grains like bulgur and couscous, legumes, olive oil, vegetables, and spices. Meat is typically consumed in moderation. Emphasizes fresh, unprocessed foods.",
            "Korean Diet": "Rich in vegetables, fermented foods like kimchi, and moderate amounts of rice, meat, and seafood. Often includes side dishes (banchan) that are nutrient-dense and high in fiber.",
            "Japanese Diet": "Features a high intake of fish, seafood, and plant-based foods such as rice, vegetables, and soy products. Low in red meat and dairy, often includes green tea and seaweed, known for its low-fat and high-fiber content.",
            "Thai Diet": "Known for its balance of flavors and use of fresh herbs and spices. Includes rice, noodles, seafood, lean meats, and a variety of vegetables. Often incorporates coconut milk, which adds healthy fats.",
            "Mediterranean Diet": "Emphasizes fruits, vegetables, whole grains, olive oil, fish, and moderate wine consumption. Known for its heart-healthy benefits, with low intake of red meat and dairy.",
            "Italian Diet": "Focuses on pasta, olive oil, tomatoes, and fresh herbs. Includes moderate amounts of meat and fish, with an emphasis on fresh, seasonal produce.",
            "French Diet": "Known for its emphasis on fresh, high-quality ingredients. Includes a variety of cheeses, breads, and wine. Often features moderate portions and rich flavors.",
            "Indian Diet": "Rich in spices, lentils, rice, and vegetables. Often vegetarian, with a variety of regional dishes. Includes dairy products like yogurt and ghee.",
            "Mexican Diet": "Includes beans, corn, tomatoes, and avocados. Known for its use of spices, chili peppers, and herbs. Often includes tortillas, rice, and a variety of meats.",
            "Chinese Diet": "Features rice, noodles, vegetables, and a variety of meats and seafood. Known for its balance of flavors and use of soy sauce, ginger, and garlic.",
            "African Diet": "Varies greatly by region. Often includes grains like millet and sorghum, tubers, legumes, and vegetables. Known for its use of spices and diverse flavors.",
            "Nordic Diet": "Focuses on whole grains, fatty fish, root vegetables, and berries. Known for its health benefits, including heart health and weight management.",
            "American Diet": "Often includes a wide variety of foods, with an emphasis on convenience and processed foods. Can vary greatly depending on region and personal preferences.",
            "Brazilian Diet": "Known for its rice and beans, fresh fruits, and vegetables. Often includes meat, particularly beef, and a variety of regional dishes.",
            "Caribbean Diet": "Rich in tropical fruits, vegetables, seafood, and spices. Known for its vibrant flavors and use of ingredients like coconut, plantains, and beans.",
            "Ethiopian Diet": "Features injera (a type of flatbread), lentils, chickpeas, and a variety of spices. Often includes stews and a mix of vegetarian and meat dishes.",
            "Vietnamese Diet": "Known for its light, fresh flavors and use of herbs and vegetables. Includes rice, noodles, seafood, and lean meats, with a focus on balance and nutrition.",
            "Spanish Diet": "Emphasizes olive oil, fresh vegetables, seafood, and lean meats. Known for tapas (small dishes) and a variety of regional specialties."
        }

        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
            model=name,
           openai_api_key= "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",
            #openai_api_key= "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",
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
        logging.info('Starting response generation')
        for diet_name, diet_description in self.diets.items():
            logging.info(f'Analyzing diet: {diet_name}')
            heart_rate, blood_pressure, body_temp, respiratory_rate, oxygen_sat = ([] for i in range(5))
            att_list = ['heart_rate', 'blood_pressure', 'body_temperature', 'respiratory_rate', 'oxygen_saturation']
            i = 0
            attempts = 0
            max_attempts = 5  # Set a maximum number of attempts per diet
            while i < self.sample and attempts < max_attempts:
                try:
                    response = self.chain0.invoke({
                        "diet": diet_name,
                        "description": diet_description
                    })

                    ok = self.__output_validation(response, att_list, -1, 1)
                    if ok:
                        heart_rate.append(response['heart_rate'])
                        blood_pressure.append(response['blood_pressure'])
                        body_temp.append(response['body_temperature'])
                        respiratory_rate.append(response['respiratory_rate'])
                        oxygen_sat.append(response['oxygen_saturation'])
                        i += 1
                        logging.info(f'Successfully analyzed {diet_name}')
                    else:
                        logging.warning(f'Invalid response for {diet_name}')
                except Exception as e:
                    logging.error(f'Error analyzing {diet_name}: {str(e)}')
                attempts += 1

            if attempts == max_attempts:
                logging.warning(f'Max attempts reached for {diet_name}')

            logging.info(f'Building dataframe for {diet_name}')
            df = pd.DataFrame({
                "heart_rate": heart_rate,
                "blood_pressure": blood_pressure,
                "body_temperature": body_temp,
                "respiratory_rate": respiratory_rate,
                "oxygen_saturation": oxygen_sat,
                "diet": [diet_name] * len(heart_rate),
                "model": [self.name] * len(heart_rate)
            })


            frames.append(df)
        return pd.concat(frames)

    def __create_prompt(self, j, format_instructions):
        if j == 0:
            return PromptTemplate(
                template = (
                    "As a board-certified nutritionist and medical expert, please provide a comprehensive rating "
                    "of the physiological effects of the {diet} diet. The diet is described as follows: {description}\n"
                    "Based on scientific evidence and clinical studies, rate the potential impact of this diet on the following "
                    "physiological measures using a scale of -1 to 1:\n"
                    "1. Heart Rate\n"
                    "2. Blood Pressure\n"
                    "3. Body Temperature\n"
                    "4. Respiratory Rate\n"
                    "5. Oxygen Saturation\n"
                    "Rating Scale:\n"
                    "- -1.0: Strong negative impact (significant worsening of the measure)\n"
                    "- -0.9: Significant negative impact\n"
                    "- -0.8: Noticeable negative impact\n"
                    "- -0.7: Moderate to significant negative impact\n"
                    "- -0.6: Moderate negative impact\n"
                    "- -0.5: Slight to moderate negative impact\n"
                    "- -0.4: Slight negative impact\n"
                    "- -0.3: Minimal negative impact\n"
                    "- -0.2: Very minimal negative impact\n"
                    "- -0.1: Negligible negative impact\n"
                    "- 0: Neutral impact (no significant change)\n"
                    "- 0.1: Negligible positive impact\n"
                    "- 0.2: Very minimal positive impact\n"
                    "- 0.3: Minimal positive impact\n"
                    "- 0.4: Slight positive impact\n"
                    "- 0.5: Slight to moderate positive impact\n"
                    "- 0.6: Moderate positive impact\n"
                    "- 0.7: Moderate to significant positive impact\n"
                    "- 0.8: Noticeable positive impact\n"
                    "- 0.9: Significant positive impact\n"
                    "- 1.0: Strong positive impact (significant improvement of the measure)\n"
                    "For each measure, provide the numerical rating from this scale. 0 would indicate a neutral relationship of the diet "
                    "on the specific physiological measure chosen, 1 would indicate a direct positive relationship between the physiological "
                    "measure and the type of diet, and -1 would indicate a direct negative relationship of a particular diet on the physiological measure.\n"
                    "Here are some factors you should consider:\n"
                    "- Macronutrient and micronutrient balance\n"
                    "- Caloric and nutrition intake\n"
                    "- Metabolic and hormonal impact\n"
                    "- Dietary composition and quality\n"
                    "- Dietary patterns and meal timing\n"
                    "- Health and functional effects\n"
                    "- Food processing and preparation\n"
                    "- Cultural, personal, and genetic factors\n"
                    "- Supplementation and alternative diets\n"
                    "{format_instructions}\n"
                ),
                input_variables=["diet", "description"],
                partial_variables={"format_instructions": format_instructions}
            )

    def __create_schemas(self, i):
        if i == 0:
            response_schema = [
                ResponseSchema(name="heart_rate", description="A numerical decimal rating between -1 and 1"),
                ResponseSchema(name="blood_pressure", description="A numerical decimal rating between -1 and 1"),
                ResponseSchema(name="body_temperature", description="A numerical decimal rating between -1 and 1"),
                ResponseSchema(name="respiratory_rate", description="A numerical decimal rating between -1 and 1"),
                ResponseSchema(name="oxygen_saturation", description="A numerical decimal rating between -1 and 1")
            ]
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


# choose an LLM to explore by uncommenting the a line with name=(<LLM name>)
#name=("openai/gpt-3.5-turbo")
name=("openai/chatgpt-4o-latest")
#name=("mistralai/mistral-7b-instruct")
#name=("google/gemma-2-27b-it")
#name=("meta-llama/llama-2-70b-chat")
analyzer = DietAnalyzer(name)
if name == ("openai/gpt-3.5-turbo"):
    fname = "gpt3.5"
if name == ("openai/chatgpt-4o-latest"):
    fname = "gpt4"
if name == ("mistralai/mistral-7b-instruct"):
    fname = "mistral"
if name == ("google/gemma-2-27b-it"):
    fname = "gemma"
if name == ("meta-llama/llama-2-70b-chat"):
    fname = "llama"


logging.basicConfig(filename=fname + "_us_diet_analysis12.log", encoding='utf-8', level=logging.DEBUG)
for i in range(5):  # choose n for how many times to run this survey
    df = analyzer.generate_response()
    df.to_csv("results/final/" + fname + "/us_diet_analysis_final" + fname + str(i))