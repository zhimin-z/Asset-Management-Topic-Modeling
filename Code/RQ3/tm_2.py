from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_root_cause_summary', min_cluster_size=20)
topic_model.sweep()
