## Understanding your device: Device Query

Working with OpenVINO requires understanding facts about your device.

OpenVINO uses an Intermediate Representation format to translate a model graph into a proprietary format used by the C++ runtime. 

OpenArc takes the optimization process a step further by offering tools for converting models which embrace the complexity of the task. 

While the excellent CLI tool streamlines the process,  each parameter requires careful consideration of several different facts the Device Query makes easier to discover. Seriously- use [Intel Ark](https://www.intel.com/content/www/us/en/ark.html) for hardware you don't own and the Device Query for every other convieveable usecase.

Here's what's most important to consider:

### Supported Datatypes

Most Intel Devices will support FP32 natives as well 


### Quantization

The same rules, practices and 






