
# Experiment 1: feed the original content to BerTopic
from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_original_content')
topic_model.sweep()
