def stage_and_commit_gradient_capture(
    manager,
    model_id: str,
    *,
    denominator: float,
    numerator_scale: float = 1.0,
    backward_completed: bool = True,
):
    """Advance the same explicit two-phase capture boundary used by ModelRunner."""

    manager.stage_gradient_numerators(
        model_id,
        denominator=denominator,
        numerator_scale=numerator_scale,
        backward_completed=backward_completed,
    )
    return manager.commit_gradient_capture(model_id)
