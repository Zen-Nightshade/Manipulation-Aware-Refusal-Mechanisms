from dotenv import load_dotenv

def init_env():
    """
    make a provider.env file in the project root and api keys of the provider you are gonna use in the following format

    ```
    GROQ_API_KEY = <paste your api key>
    GOOGLE_API_KEY = <paste your api key>
    CEREBRAS_API_KEY = <paste your api key>
    ```
    """
    load_dotenv("providers.env")