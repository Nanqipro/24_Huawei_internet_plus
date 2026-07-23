# ModelArts deployment adapter

This directory contains a cleaned adapter based on the team's competition
submission. It is a template, not a verified one-click deployment bundle.

Before packaging:

1. install the `goodlab_fatigue_detection` package in the serving image;
2. place `fatigue-detection-v4-c7-320.onnx` beside the model bundle;
3. check the Python runtime and dependency versions supported by the target
   ModelArts image;
4. send the uploaded video under the `input_video` form field.

The original competition image used Huawei Cloud's
`model_service.pytorch_model_service.PTServingBaseService`. That module is
provided by the ModelArts serving environment and is not a PyPI dependency.

