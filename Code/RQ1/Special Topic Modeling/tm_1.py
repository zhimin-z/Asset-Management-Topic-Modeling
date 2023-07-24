from topic_modeling import TopicModeling

topic_model = TopicModeling('anomaly', min_cluster_size=30)
topic_model.sweep()
