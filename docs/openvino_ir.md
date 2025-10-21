### OpenVINO Model Format: Intermediate Representation


[OpenVINO Intermediate Representations](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) describe a set of standarsization techniques to format the operations of a neural network into a computational graph topology that a compiler can understand, stored in `openvino_model.bin` and `openvino_model.xml`.

`openvino_model.xml` nodes represent [`opsets`](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets.html#overview-of-artificial-neural-networks-representation) while edges represent data flow through the network a given IR describes. Together, these help OpenVINO's device plugin system determine what opsets are required vs which are *implemented* for a target device.
 
Optimum-Intel provides [a hands on primer](https://huggingface.co/docs/optimum/main/en/intel/openvino/optimization) demonstrating how the IR is used to apply post training optimization and is a good place to start building some intuition. 

However, you don't need to understand everything and there are many sources of preconverted models.