// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/version.hpp"
#include "openvino/genai/visibility.hpp"

/**
 * OpenVINO GenAI major version
 */
#define OPENVINO_GENAI_VERSION_MAJOR 2025

/**
 * OpenVINO GenAI minor version
 */
#define OPENVINO_GENAI_VERSION_MINOR 4

/**
 * OpenVINO GenAI patch version
 */
#define OPENVINO_GENAI_VERSION_PATCH 0

namespace ov {
namespace genai {

/**
 * Returns OpenVINO GenAI full version including git commit and hash information in form of:
 *   <MAJOR>.<MINOR>.<PATCH>.<REVISION>-<COMMIT NUMBER>-<COMMIT HASH>[-<BRANCH SUFFIX>]
 */
OPENVINO_EXTERN_C OPENVINO_GENAI_EXPORTS const ov::Version OPENVINO_CDECL get_version();

} // namespace genai
} // namespace ov
