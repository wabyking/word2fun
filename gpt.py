import sys
import openai
openai.api_key = "sk-bfXLHlC9lXpPtRRkmIGW7MGCqbVJDNvPTM7KOEE1"
response = openai.Completion.create(engine="davinci", prompt=" ".join(sys.argv[1:]), max_tokens=16)
print(response)