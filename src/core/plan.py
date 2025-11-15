# TODO: review these plan builder assumptions, i'm not sure this is how i want the plan
from src.core.spec import Plan, Spec


class PlanBuilder:
    """
    Build execution plan from input spec.

    Determines sequence of generation steps and generator selection.
    """

    def build_plan(self, spec: Spec) -> Plan:
        """
        TODO: Implement plan building logic.
        - Analyze spec requirements
        - Select appropriate generators (arms)
        - Define execution sequence
        """
        pass
