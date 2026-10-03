"""Arabic synthetic fixture for the local review API, not a real news verdict."""
import json

from newsrag.lab import EditorialDecision, RawDocument
from newsrag.lab_pipeline import ReviewWorkspace, TokenOffset, decode_evidence_spans

source = RawDocument('synthetic-demo', 'عدد ١٢', 'https://example.test/fixture',
                     'synthetic_test_fixture_only')
workspace = ReviewWorkspace('عدد ١٢')
workspace.add_document(source)
for candidate in decode_evidence_spans(source, [TokenOffset(0, 3, 'عدد'),
                                                TokenOffset(4, 6, '١٢')],
                                       [.8, .9], threshold=.7):
    workspace.add_candidate(candidate)
# No factual verdict is invented: the demo remains needs_review.
output = workspace.finish(EditorialDecision(workspace.claim, 'needs_review',
                                            note='مثال تركيبي لا خبر حقيقي'), [0])
print(json.dumps(output, ensure_ascii=False, indent=2))
