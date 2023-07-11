from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_root_cause', min_cluster_size=10)
topic_model.sweep()
