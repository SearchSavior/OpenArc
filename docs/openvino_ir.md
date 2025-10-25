### OpenVINO Model Format: Intermediate Representation


[OpenVINO Intermediate Representations](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) describe a set of standarsization techniques to format the operations of a neural network into a computational graph topology that a compiler can understand, stored in `openvino_model.bin` and `openvino_model.xml`.

`openvino_model.xml` nodes represent [`opsets`](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets.html#overview-of-artificial-neural-networks-representation) while edges represent data flow through the network a given IR describes. Together, these help OpenVINO's device plugin system determine what opsets are required vs which are *implemented* for a target device.
