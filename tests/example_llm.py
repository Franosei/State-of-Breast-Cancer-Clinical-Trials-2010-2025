from llm.openai_client import OpenAIClient

llm = OpenAIClient()

system_msg = "You are a helpful assistant that classifies trial endpoints."
user_prompt = "Classify 'progression-free survival' into a CDISC standard endpoint."

result = llm.run(system_msg, user_prompt)
print(result)
