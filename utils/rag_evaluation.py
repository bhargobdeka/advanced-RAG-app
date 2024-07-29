from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase     
from deepeval import evaluate

## Metric Functions

class LLM_Metric:
    def __init__(self, query, retrieval_context, actual_output):
        self.query = query
        self.retrieval_context = retrieval_context
        self.actual_output = actual_output

    # Faithfulness
    def get_faithfulness_metric(self):
        metric = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        test_case = LLMTestCase(
            input=self.query,
            actual_output=self.actual_output,
            retrieval_context=self.retrieval_context
        )

        metric.measure(test_case)
        return metric.score, metric.reason

    # Contextual Relevancy
    def get_contextual_relevancy_metric(self):
        metric = ContextualRelevancyMetric(
            threshold=0.7,
            model="gpt-4o",
            include_reason=True
        )
        test_case = LLMTestCase(
            input=self.query,
            actual_output=self.actual_output,
            retrieval_context=self.retrieval_context
        )
        
        metric.measure(test_case)
        return metric.score, metric.reason
    
    # Answer Relevancy
    def get_answer_relevancy_metric(self):
        metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
        )
        test_case = LLMTestCase(
            input=self.query,
            actual_output=self.actual_output
        )
        metric.measure(test_case)
        return metric.score, metric.reason
    
    # Hallucination
    def get_hallucination_metric(self):
        metric = HallucinationMetric(threshold=0.5)
        test_case = LLMTestCase(
        input=self.query,
        actual_output=self.actual_output,
        context=self.retrieval_context  
        )
        metric.measure(test_case)
        return metric.score, metric.reason