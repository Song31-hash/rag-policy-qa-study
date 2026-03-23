# Error Analysis Table

| QID | Gold | Vanilla | RAG | Improvement | ErrorType | Notes |
|---|---|---|---|---|---|---|
| Q-01 | no | no | no | - | correct | Both models matched gold decision. |
| Q-02 | no | no | no | - | correct | Both models matched gold decision. |
| Q-03 | no | no | no | - | correct | Both models matched gold decision. |
| Q-04 | no | no | no | - | correct | Both models matched gold decision. |
| Q-05 | no | no | no | - | correct | Both models matched gold decision. |
| Q-06 | yes | yes | yes | - | correct | Both models matched gold decision. |
| Q-07 | selection_required | selection_required | selection_required | - | correct | Both models matched gold decision. |
| Q-08 | yes | no | yes | RAG fixed | vanilla_error | Vanilla failed but RAG matched gold. |
| Q-09 | yes | no | yes | RAG fixed | vanilla_error | Vanilla failed but RAG matched gold. |
| Q-10 | no | yes | no | RAG fixed | vanilla_error | Vanilla failed but RAG matched gold. |
| Q-11 | yes | no | no | none | reasoning_or_source_gap | Both models wrong; inspect policy.docx completeness and rule application. |
| Q-12 | yes | no | uncertain | none | source_gap_or_retrieval_insufficiency | RAG responded uncertain despite retrieved chunks; source completeness should be checked. |
| Q-13 | no | no | no | - | correct | Both models matched gold decision. |
| Q-14 | yes | no | yes | RAG fixed | vanilla_error | Vanilla failed but RAG matched gold. |
| Q-15 | yes | yes | yes | - | correct | Both models matched gold decision. |
| Q-16 | no | yes | uncertain | none | source_gap_or_retrieval_insufficiency | RAG responded uncertain despite retrieved chunks; source completeness should be checked. |
| Q-17 | yes | no | no | none | reasoning_or_source_gap | Both models wrong; inspect policy.docx completeness and rule application. |
| Q-18 | no | no | no | - | correct | Both models matched gold decision. |
| Q-19 | yes | yes | yes | - | correct | Both models matched gold decision. |
| Q-20 | no | selection_required | selection_required | none | reasoning_or_source_gap | Both models wrong; inspect policy.docx completeness and rule application. |
