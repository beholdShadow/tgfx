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

#pragma once

#include <sstream>
#include <iomanip>
#include "gpu/processors/FragmentProcessor.h"
#include "tgfx/core/Color.h"
#include "tgfx/core/Shader.h"
namespace tgfx {
class CustomEffectProcessor : public FragmentProcessor {
 public:
  static std::unique_ptr<CustomEffectProcessor> Make(const ShaderConfig& config);

  std::string name() const override {
    return "CustomEffectProcessor";
  }

  void onComputeProcessorKey(BytesKey* bytesKey) const override;

  void emitCode(EmitArgs& args) const override;

 private:
  void onSetData(UniformBuffer* uniformBuffer) const override;

 protected:
  DEFINE_PROCESSOR_CLASS_ID

  CustomEffectProcessor(const ShaderConfig& config)
      : FragmentProcessor(ClassID()), config(config) {
    std::hash<std::string> hash_fn;
    size_t hash_value = hash_fn(config.funcImpl);
    std::stringstream ss;
    ss << std::setw(8) << std::setfill('0') << std::hex << hash_value;  // 格式化为 8 字符的十六进制
    hash = ss.str();
  }

  bool onIsEqual(const FragmentProcessor& processor) const override;

  ShaderConfig config;
  std::string hash;
};
}  // namespace tgfx
