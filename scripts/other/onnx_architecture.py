import onnx
print(onnx.__version__)
print(dir(onnx))
# Load the ONNX model
encoder_model_path = "encoder-epoch-200-avg-1-chunk-32-left-512.onnx"
model = onnx.load(encoder_model_path)

onnx.checker.check_model(model)

for node in model.graph.node:
    print(node.op_type, node.name, node.input, node.output)

print(model.graph)

