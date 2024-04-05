from langchain.evaluation import load_evaluator
evaluator=load_evaluator("pairwise_embedding_distance")
x=evaluator.evaluate_string_pairs(prediction='apple',prediction_b='orange') 
print(x)