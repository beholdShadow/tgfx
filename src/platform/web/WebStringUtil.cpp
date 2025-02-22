/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Tencent is pleased to support the open source community by making tgfx available.
//
//  Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
//  Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//  in compliance with the License. You may obtain a copy of the License at
//
//      https://opensource.org/licenses/BSD-3-Clause
//
//  unless required by applicable law or agreed to in writing, software distributed under the
//  license is distributed on an "as is" basis, without warranties or conditions of any kind,
//  either express or implied. see the license for the specific language governing permissions
//  and limitations under the license.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "tgfx/platform/StringUtil.h"
#include <emscripten.h>
#include <emscripten/bind.h>
namespace tgfx {
std::vector<std::string> StringUtil::SplitFromPlatform(const std::string& text) {
    // 调用 JavaScript 的 split 方法
    auto scalerContextClass = emscripten::val::module_property("ScalerContext");
    if (!scalerContextClass.as<bool>()) {
      return {};
    }
    emscripten::val jsArray = scalerContextClass.call<emscripten::val>("splitCharacterSequences", text);
    // 将 JavaScript 数组转换为 C++ vector
    std::vector<std::string> result;
    unsigned int length = jsArray["length"].as<unsigned int>();
    for (unsigned int i = 0; i < length; ++i) {
        result.push_back(jsArray[i].as<std::string>());
    }
    return result;
}

bool StringUtil::IsEmoji(const std::string& text) {
  if (text.empty()) { return false; }
  auto scalerContextClass = emscripten::val::module_property("ScalerContext");
  if (!scalerContextClass.as<bool>()) {
      return false;
  }
  return scalerContextClass.call<bool>("isEmoji", text);
}   

}  // namespace tgfx