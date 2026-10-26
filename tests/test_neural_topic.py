import pytest
from newsrag.neural import TopicNet, TOPIC_LAYER_COUNT, tfidf_centroid_baseline

TRAIN = (["الصحة واللقاح الجديد فعال", "لقاح كورونا والصحة العامة", "الصحة ودواء جديد للقلب",
          "السياسة والانتخابات البرلمانية", "انتخابات الرئاسة والسياسة اليوم", "الحكومة والسياسة الخارجية"],
         ["Health", "Health", "Health", "Politics", "Politics", "Politics"])

def test_layer_count_is_four_learnable():
    net = TopicNet(vocab_size=100)
    net.fit(*TRAIN, epochs=2)
    assert TOPIC_LAYER_COUNT == 4
    assert len(net.weights) == 4 and len(net.biases) == 4

def test_unfitted_predict_fails_closed():
    with pytest.raises(RuntimeError):
        TopicNet().predict(["نص"])

def test_determinism_same_seed():
    a = TopicNet(vocab_size=100, seed=7); a.fit(*TRAIN, epochs=3)
    b = TopicNet(vocab_size=100, seed=7); b.fit(*TRAIN, epochs=3)
    assert a.predict(["خبر صحي عن دواء"]) == b.predict(["خبر صحي عن دواء"])

def test_learns_obvious_topics():
    net = TopicNet(vocab_size=100, seed=7)
    net.fit(*TRAIN, epochs=40)
    assert net.predict(["لقاح وصحة"]) == ["Health"]
    assert net.validated_for_release is False

def test_baseline_runs():
    preds = tfidf_centroid_baseline(TRAIN[0], TRAIN[1], ["لقاح"], seed=7)
    assert len(preds) == 1
