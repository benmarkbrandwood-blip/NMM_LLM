# Corrected-v4 Gold Corpus Policy

`tests/gold/v4_rules_corpus.json` is a fixed correctness corpus for the current
project rules. It is not generated during tests and must never be updated from
test output automatically.

Each case records a board FEN, exact terminal and perspective results, the
number and canonical SHA-256 signature of all complete legal actions, and an
optional complete probe action. A probe action includes its capture in the
same move object, so mill formation and capture cannot be validated as two
unrelated states.

An update is permitted only when all of the following are recorded in the
reviewing commit or experiment evidence:

1. the named project-rule change or independently verified decoder fix;
2. a field-level old/new diff for every affected case;
3. confirmation that unaffected cases and perspective inversion still pass;
4. the new whole-corpus signature;
5. explicit reviewer approval of the changed expected values.

Adding a case follows the same process. Performance changes, reference-engine
behavior, or a failing test alone are not authority to rewrite expected
outputs. Run `tests/test_v4_gold_corpus.py` together with the Malom and rules
regressions after every reviewed change.
