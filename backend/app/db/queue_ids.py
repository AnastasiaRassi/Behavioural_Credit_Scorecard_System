"""Translation between the worklist's row ids and the database's customer ids.

Two id spaces exist because `train.py` labels queue rows `C-<row position>`:
`feature_engineering` drops the source `ID` column, so the frame the queue is
built from is positionally indexed, while the database is keyed on the source id
the dataset actually published. Source ids are 1-based and positions are 0-based,
hence the offset.

Isolated in its own module so there is exactly one place that knows the mapping.
It stops being needed the day train.py emits source ids; db/queue_predictions.py
checks the result against stored figures either way, because a silent off-by-one
would attach a decision to the wrong person's account.
"""
QUEUE_ID_PREFIX = "C-"
QUEUE_ID_OFFSET = 1


def source_id_for(queue_id: str) -> str:
    """The database's customer id for a worklist row's id."""
    bare = str(queue_id).upper().removeprefix(QUEUE_ID_PREFIX)
    if not bare.isdigit():
        # A walk-in registered through /predictions carries its own id and was
        # never in the queue, so there is nothing to translate.
        return str(queue_id)
    return str(int(bare) + QUEUE_ID_OFFSET)


def queue_id_for(customer_id: str) -> str:
    """The worklist id for a database customer id, for a screen that links back."""
    if not str(customer_id).isdigit():
        return str(customer_id)
    return f"{QUEUE_ID_PREFIX}{int(customer_id) - QUEUE_ID_OFFSET:05d}"
