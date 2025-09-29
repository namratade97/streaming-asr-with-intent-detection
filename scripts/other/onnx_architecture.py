import onnx
print(onnx.__version__)
print(dir(onnx))  # should include 'load'
# Load the ONNX model
encoder_model_path = "encoder-epoch-200-avg-1-chunk-32-left-512.onnx"
model = onnx.load(encoder_model_path)

# Check the model is valid
onnx.checker.check_model(model)

# Print a summary of nodes and layers
for node in model.graph.node:
    print(node.op_type, node.name, node.input, node.output)

# Or just print the graph
print(model.graph)

# Visualize Graph with Netron

# (.venv_tensorboard) nde@dev-cailan-shared-2-ol9:/disk1/nde/polaris_intent_detection$ pip install netron
# Looking in indexes: https://nexus.melodis.com/repository/soundhound-pypi-group/simple, https://download.pytorch.org/whl/cpu
# Collecting netron
#   Downloading https://nexus.melodis.com/repository/soundhound-pypi-group/packages/netron/8.6.6/netron-8.6.6-py3-none-any.whl (2.0 MB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 28.4 MB/s eta 0:00:00
# Installing collected packages: netron
# Successfully installed netron-8.6.6

# [notice] A new release of pip is available: 25.0.1 -> 25.2
# [notice] To update, run: pip install --upgrade pip
# (.venv_tensorboard) nde@dev-cailan-shared-2-ol9:/disk1/nde/polaris_intent_detection$ netron encoder-epoch-200-avg-1-chunk-32-left-512.onnx
# Serving 'encoder-epoch-200-avg-1-chunk-32-left-512.onnx' at http://localhost:8080
# ^CStopping http://localhost:8080
# (.venv_tensorboard) nde@dev-cailan-shared-2-ol9:/disk1/nde/polaris_intent_detection$ netron encoder-epoch-200-avg-1-chunk-32-left-512.onnx
# Serving 'encoder-epoch-200-avg-1-chunk-32-left-512.onnx' at http://localhost:8080
# ^CStopping http://localhost:8080
# (.venv_tensorboard) nde@dev-cailan-shared-2-ol9:/disk1/nde/polaris_intent_detection$ 


