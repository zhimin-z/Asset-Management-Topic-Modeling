
# Experiment 2: feed the original content to GPT model and get the generated summary, then feed the summary to BerTopic

from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_original_content_gpt_summary')
topic_model.sweep()