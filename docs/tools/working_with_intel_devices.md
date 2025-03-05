## Introduction to working with Intel Devices

This document offers discussion of "lessons-learned" from months of working with Intel GPU devices; *hours* of blood, sweat, and tears went into setting up this project and it's a good place to share what I've learned. At this stage in the Intel AI Stack it seems like a neccessary contribution to the community.

### What is OpenVINO?

OpenVINO is an inference backend for *acclerating* inference deployments of machine learning models on Intel hardware. It can be hard to understand the documentation- the Intel AI stack has many staff engineers/contributors to all manner of areas in the open source ecosystem and much of the stack is evolving without massive community contributions like what we have seen with llama.cpp. 

Many reasons contribute to the decline of Intel's dominance/popularity in the hardware space in the past few years; however they offer extensive open source contributions to many areas of AI, ML and have been since before [Attention Is All You Need](https://arxiv.org/abs/1706.03762). AI didn't start in 2017- however the demand for faster inference on existing infrastructure has never been higher. Plus, Arc chips are cheap but come with a steep learning curve. Sure, you can settle for Vulkan... but you aren't here to download a GGUF and send it.  



### OpenVINO Utilities

Various utilities live in this notebook to help users of OpenArc understand the properties of their devices; mastering understanding of available data types, quantization strategies and  available optimization techniques is only one part of learning to use OpenVINO on different kinds of hardware.

Check out the [Guide to the OpenVINO IR] and then use my [Command Line Tool tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool) to perform converion. There are default approachs that "work" but to really leverage available compute you have to dig deeper and convert models yourself on a per-usecase basis.