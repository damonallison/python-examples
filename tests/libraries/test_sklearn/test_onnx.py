# onnx (open neural network exchange) is an open standard for representing ML
# models.
#
# Converting your models to onnx makes it simple to host models from many
# frameworks using the same runtime. This simplifies dependencies by only
# requiring a dependency on the onnx runtime itself rather than dependencies on
# multiple ML frameworks.
#
# Every operator is versioned. The runtime chooses the most recent version below
# or equal to the targetted opset number for every operator.
#
# Libraries:
#
# skl2onnx: convert sklearn to onnx onnxruntime: predict

import numpy as np
import pathlib
import onnxruntime as rt
from skl2onnx import __max_supported_opset__, convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def test_max_offset() -> None:
    print(f"Last supported offset: {__max_supported_opset__}")


def test_train(tmp_path: pathlib.Path) -> None:
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert to onnx
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    model_path = str(tmp_path / "rf_iris.onnx")
    with open(model_path, "wb") as f:
        f.write(onx.SerializeToString())
        sess = rt.InferenceSession(model_path)

    # Predict with onnx runtime
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
    print(pred_onx)
