# Experiment 3: feed the preprocessed content to BerTopic

from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_preprocessed_content')
topic_model.sweep()