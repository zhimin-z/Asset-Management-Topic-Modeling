# Experiment 6: feed the preprocessed content to BerTopic

from topic_modeling import TopicModeling

topic_model = TopicModeling('Solution_preprocessed_content')
topic_model.sweep()