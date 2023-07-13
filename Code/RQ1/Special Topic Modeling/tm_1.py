from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_summary', challenge_type='anomaly', min_cluster_size=20)
topic_model.sweep()
