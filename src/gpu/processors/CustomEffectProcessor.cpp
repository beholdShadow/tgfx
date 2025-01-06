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

#include "gpu/processors/CustomEffectProcessor.h"

namespace tgfx {
std::unique_ptr<CustomEffectProcessor> CustomEffectProcessor::Make(uint32_t id,  const std::string& fragShader, const std::vector<ShaderVar>& params) {
  return std::unique_ptr<CustomEffectProcessor>(new CustomEffectProcessor(id, fragShader, params));
}

void CustomEffectProcessor::onComputeProcessorKey(BytesKey* bytesKey) const {
  bytesKey->write(id);
}

bool CustomEffectProcessor::onIsEqual(const FragmentProcessor&) const {
  return false;
}

void CustomEffectProcessor::emitCode(EmitArgs& args) const {
  auto* fragBuilder = args.fragBuilder;
  fragBuilder->addFunction(this->fragShader);
  // auto colorName = args.uniformHandler->addUniform(ShaderFlags::Fragment, SLType::Float4, "Color");
  fragBuilder->codeAppendf("%s = customFunction(%s);", args.outputColor.c_str(), args.inputColor.c_str());
}

void CustomEffectProcessor::onSetData(UniformBuffer*) const {
  // Color color{1.0, 0.0, 0.0, 1.0};
  // uniformBuffer->setData("Color", color);
}

}  // namespace tgfx
