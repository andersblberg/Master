from plastic_id.models import REGISTRY, get_model


def test_registry_keys():
    assert {"rf", "svm", "mlp", "et"}.issubset(REGISTRY)


def test_instantiate_rf():
    mdl = get_model("rf", {"n_estimators": 10})
    mdl.fit([[0] * 8, [1] * 8], [0, 1])
    assert hasattr(mdl, "predict")
