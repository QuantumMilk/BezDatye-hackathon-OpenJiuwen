class BudgetExceeded(RuntimeError):
    pass


class DeadlineExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class UsageUnknownError(ValueError):
    pass


class TaskFormatError(ValueError):
    pass


class ToolPolicyError(ValueError):
    pass
