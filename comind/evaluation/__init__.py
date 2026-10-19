from .common import TaskEvaluator, load_predictions
from .handover import HandoverEvaluator
from .joint_attention import JointAttentionEvaluator
from .scoia import ScoiaEvaluator

EVALUATORS = {
    "joint_attention": JointAttentionEvaluator,
    "handover": HandoverEvaluator,
    "scoia": ScoiaEvaluator,
}


def get_evaluator(task: str, **kwargs) -> TaskEvaluator:
    return EVALUATORS[task](**kwargs)


def lookup(results: dict, path: str):
    """lookup(r, "bbox_IoU_0.50:AP") -> r["bbox_IoU_0.50"]["AP"], or None."""
    for key in path.split(":"):
        if not isinstance(results, dict) or key not in results:
            return None
        results = results[key]
    return results


__all__ = ["EVALUATORS", "TaskEvaluator", "get_evaluator", "load_predictions", "lookup"]
