import pytest
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole, DataSplit
from prompt_opt.ingestion.stratifier import stratify


def test_stratify_splits():
    # Generate mock records spanning two categories
    records = []
    for i in range(20):
        records.append(
            ConversationRecord(
                conversation_id=f"conv_{i}",
                messages=[
                    Message(role=MessageRole.USER, content=f"Query {i}"),
                    Message(role=MessageRole.ASSISTANT, content=f"Response {i}"),
                ],
                category="factual_qa" if i % 2 == 0 else "coding",
            )
        )

    res = stratify(
        records,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        min_bucket_size=2,
    )

    assert res.sizes["train"] > 0
    assert res.sizes["dev"] > 0
    assert res.sizes["test"] > 0
    assert sum(res.sizes.values()) >= 20  # oversampling might duplicate records
