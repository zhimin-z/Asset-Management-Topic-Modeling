from topic_modeling import TopicModeling

topic_model = TopicModeling('Challenge_solution', min_cluster_size=30)
topic_model.sweep()
