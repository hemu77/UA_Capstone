# Instruction and Retry Language Review

Date: 2026-09-30. Scope: the fresh 896-network protocol, before paid calibration.

**Live-calibration follow-up:** the first batch exposed a global correction that
returned only the faulty pairs. See [CALIBRATION_FINDINGS.md](CALIBRATION_FINDINGS.md).
The corrected v2 retry wording explicitly asks for the entire original valid
friendship set, with no new ties. Both AI reviewers rechecked the revised
wording/contract. This supersedes the initial source fingerprints below; human
validation remains outstanding and paid calibration is stopped.

## What Was Reviewed

- [x] English reference instructions and translations into Hindi, Japanese and Brazilian Portuguese.
- [x] US, India, Japan and Brazil country labels and the shared context statement.
- [x] Global, local, sequential, iterative-add and iterative-drop instructions.
- [x] Actual base retries, exact-count corrections and reversed-edge corrections.
- [x] Fictional adults, unchanged attributes and unspecified participant spoken language.
- [x] Matching selection counts, eligibility, new versus existing friends, self-link restrictions and output-only formats.
- [x] Fixed English JSON payload across instruction languages, including identical actor and candidate data.
- [ ] Independent human bilingual/native-speaker signoff. This has not occurred.
- [ ] Live multilingual model calibration. Offline fixtures cannot establish model compliance.

Two separate read-only AI reviewers checked the source against the English
reference. Gauss reviewed Hindi and Japanese; Tesla reviewed Brazilian Portuguese,
English graph-statistic meanings and test coverage. The parent integrated their
findings. These are independent task lanes, not human raters, a blinded
backtranslation study or evidence of independence between model families.
Backtranslations below are AI semantic glosses, not measurements of equivalence.

Both reviewers rechecked the edited source and reported no unresolved material
semantic findings in their assigned scope. They inspected tests; the parent ran
the tests. Review lanes: Gauss `01a0f0fc-875f-7b21-877a-f7793f339d7a` and
Tesla `01a0f0fc-8592-7703-b777-19a3970d6536`.

## Findings and Repairs

| Finding | Repair | Meaning preserved |
| --- | --- | --- |
| `mutual` was undefined in every language, including English. | Both iterative actions now define candidate degree as current friend count and mutual as current friends shared with the actor. | Common neighbors, not reciprocal directed ties or total candidate degree. |
| Hindi used an uncommon technical rendering of undirected. | The global instruction and global retry now say friendship has no direction. | Each friendship is one unordered pair, listed once with the smaller numeric ID first. |
| Japanese retries said candidate IDs without explicitly restating eligibility. | Base and exact-count corrections now specify candidates eligible for selection. | Only the supplied eligible list, with no duplicate or self IDs. |

Reviewed iterative-statistic wording:

| Language | Exact statistics sentence |
| --- | --- |
| English | Candidate degree is their current number of friends; mutual is the number of current friends shared with you. |
| Hindi | उम्मीदवार की degree उसके वर्तमान मित्रों की संख्या है; mutual वर्तमान में आपके और उम्मीदवार के साझा मित्रों की संख्या है। |
| Japanese | 候補者のdegreeは現在の友人数で、mutualは現在あなたと候補者に共通する友人の数です。 |
| Brazilian Portuguese | O campo degree do candidato é seu número atual de amigos; mutual é o número atual de amigos em comum com você. |

All three translated sentences backtranslate to: candidate degree counts its
current friends; mutual counts the friends currently shared by candidate and actor.
The graph fixture checks these numbers directly against neighbors in the graph.

Hindi `मित्रता में कोई दिशा नहीं होती।` means "Friendship has no direction."
Japanese `選択可能な候補者` means "candidates eligible for selection."
Other action, context and country-label wording was retained; the reviewers found
no additional concrete semantic mismatch in their bounded inspection.

## Reproducible Evidence

`revision224.py --preflight` exports these current v5 files without using the API:

- `outputs/revision896_retry_v5_preflight/prompt_catalog.json`: 80 base examples,
  four countries x four languages x five prompt actions.
- `outputs/revision896_retry_v5_preflight/retry_catalog.json`: 20 captured corrections,
  four languages x five actions. A mocked bad response enters the real parser
  and retry function, then a valid response succeeds on attempt two.
- `outputs/revision896_retry_v5_preflight/preparation.json` and `report.json`: protocol,
  source, roster and runtime fingerprints for the tested implementation.

Retry examples are country-independent; the country statement remains in the
original system message. The export records invalid and valid fixture replies
and the actual correction, rather than maintaining a second translation table.
Fixtures are marked `translation_review_fixture`; none are model results.

Tests in `tests/test_revision224.py` check catalog coverage, unsupported-language
rejection, payload identity, eligibility, actual degree/common-neighbor values,
and all localized retry branches. String assertions protect reviewed wording
from accidental changes; they do not prove semantic quality. Both added tests
failed before the fixes and passed afterward.

The 80 examples exercise all country/language combinations for software review.
The paid study still uses only seven settings: four countries in English plus
US-Hindi, US-Japanese and US-Portuguese. This is not a 16-setting interaction study.
Spanish remains supported by the historical shared helper, not the fresh protocol.

Initial reviewed fingerprints (SHA-256), superseded by live-calibration fix:

```text
protocol: a89569ab8f9a7862a9c82da2ad81111211bdd61417a477d036bb59e2000fece6
revision224_prompts.py: da28d0524571c5117aecbaff8e99237d454b03bcc257cff2ac603beaaf4b99bd
constants_and_utils.py: d4095580b683987f7e2821cf5abf7fe71aeb5f5d3dde26657d60a01bdfed958c
revision224.py: e2ac221e86045a3824bc8e160fb32452047428ec864d2e9dd07101e560a54939
```

The parent reran all 67 offline tests, all 896 full-roster fixture cells and
12 matched-control fixtures successfully. Notebook JSON and all 42 code-cell
syntax checks passed using IPython's transformer for notebook magics. Existing
pandas/NumPy deprecation warnings are nonfatal. No paid API calls were made for
this review; fresh analysis and calibration both report `NOT_COLLECTED`.

V2 review fingerprints:

```text
protocol: 11a59266f8635338e1eb0ca17d35172a9dda36376a822755b3d637cc785205ad
revision224.py: 8d0168a4e485a9d9a7b3d67adb2e8382f2d936a93a25ad5164aba1eebeac5d64
revision224_prompts.py: da28d0524571c5117aecbaff8e99237d454b03bcc257cff2ac603beaaf4b99bd
constants_and_utils.py: 6818d8aac517ea9eccc007699b7e242cae7921328a92f8b3454d60d0d3d0ecf4
generate_networks.py: bf0ba27ba2819f7639e03f9ba78bc5dbacb5880d12aaf529efdccd5584fc040f
```

The v2 suite has 72 passing tests. The earlier 67-test record above is historical,
not evidence that the initial live calibration succeeded. The actual failure is
also replayed offline: the partial four-edge correction is rejected, while a
complete 67-edge fixture succeeds. This is test evidence, not a collected network.

## Remaining Acceptance Boundary

### V4 Calibration Review

The base instructions and persona payload are unchanged. Two live calibration
findings required narrower retry instructions: remove illustrative numeric pairs
that a model copied into its answer, and explicitly retry a completely empty
initial answer. A nonempty unrepairable answer still stops. Once a valid tie set
has been recovered, an empty later reply cannot reset that set or regenerate it.
All requests remain within the original three-attempt limit and spending ledger.

The read-only code reviewer checked these branches and the AI language reviewer
found no material semantic mismatch between the English, Hindi, Japanese and
Brazilian Portuguese nonresponse instructions. The 20-record retry catalog now
also contains the four global `nonresponse_correction` examples. This is AI
linguistic/static review, not human validation or an accuracy percentage.

V4 reviewed fingerprints (historical):

```text
protocol: 9e3ff80c3c9ad8ddbb62f99cf8dde7205d0b0c9ee7becd15071caaf80fb05de7
revision224.py: d09bab833d2776d79cc5aa37e8adf6362c8488e5c4f37173c5021b7f401a7cdc
revision224_prompts.py: da28d0524571c5117aecbaff8e99237d454b03bcc257cff2ac603beaaf4b99bd
constants_and_utils.py: 6266e32f702fdf3192030c06e648b358d8e6c2ef37b89a0c192a33f78d42a7e6
generate_networks.py: bf0ba27ba2819f7639e03f9ba78bc5dbacb5880d12aaf529efdccd5584fc040f
```

The earlier fingerprints and test counts above are retained as historical review
records; they do not authorize v4 or imply successful collection. The complete
seven-file source fingerprint is in each preparation and calibration receipt.

### V5 Output Capacity

V4 stopped when a valid-length local friend list hit the 128-completion-token
limit. V5 raises per-person capacity to 512; global capacity remains 8192.
The regression checks that the ASCII byte lengths of all 49 peer IDs and all
1225 global pairs fit their respective caps. This conservative capacity check
does not guarantee models follow the schema, so abnormal finishes still stop.
Base and retry wording is unchanged from the AI-reviewed v4 wording.

```text
protocol: f2c6b9696988fbc9d6ee438c45a3d9d56b253c9d7af14bbed0a116485b9b0e97
revision224.py: 4c25c28588c831b7a70125166ce4226cf9c613ab123ec883fd86c1b8ae97eea6
paid_study.py: ba2146c8b611b0b68ad0a35356f7361acf15e956b68590373f6d9029da599e2b
```

All other generation-source fingerprints match v4. V5 has 79 passing tests,
896 offline full-roster fixtures and 12 control fixtures. No human signoff is
implied by the output-capacity fix or by successful API collection.

### Main Study Boundary

AI-assisted review is sufficient to proceed to the separately authorized,
bounded engineering calibration, not to claim human translation validation.
The main-study `translation_review` approval remains false until a responsible
reviewer accepts the evidence and records the review standard. A human bilingual
reviewer should read the catalog in each language, check the constraints above,
record disagreements and approve the exact source/protocol fingerprints.

English persona attributes remain intentional in every treatment: this tests
instruction-language changes, not complete language localization. A country
label does not make the roster nationally representative. Left/center/right
political orientation also does not establish cross-country measurement invariance.

If any wording changes after calibration, preserve those results separately.
The source-bound receipt must fail until the edited protocol is reviewed and
refrozen; do not quietly reuse incompatible calibration graphs.
