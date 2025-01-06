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

#include "gpu/processors/FragmentProcessor.h"
#include "tgfx/core/Color.h"

namespace tgfx {
class CustomEffectProcessor : public FragmentProcessor {
 public:
  static std::unique_ptr<CustomEffectProcessor> Make(uint32_t id, const std::string& fragShader, const std::vector<ShaderVar>& params);

  std::string name() const override {
    return "CustomEffectProcessor";
  }

  void onComputeProcessorKey(BytesKey* bytesKey) const override;

  void emitCode(EmitArgs& args) const override;

 private:
  void onSetData(UniformBuffer* uniformBuffer) const override;

 protected:
  DEFINE_PROCESSOR_CLASS_ID

  CustomEffectProcessor(uint32_t id,  const std::string& fragShader, const std::vector<ShaderVar>& params)
      : FragmentProcessor(ClassID()), id(id), fragShader(fragShader), params(params) {
  }

  bool onIsEqual(const FragmentProcessor& processor) const override;

  uint32_t id;
  std::string fragShader;
  std::vector<ShaderVar> params;
};
}  // namespace tgfx
