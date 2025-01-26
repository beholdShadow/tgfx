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

#include "tgfx/core/Shader.h"
#include "core/utils/UniqueID.h"
namespace tgfx {
class CustomShader : public Shader {
 public:
  explicit CustomShader(const ShaderConfig& config):config(config) {
  }
  virtual void setCustomVars(const std::vector<ShaderVar>& vars) override;
 protected:
  virtual Type type() const override {
    return Type::Custom;
  }
  std::unique_ptr<FragmentProcessor> asFragmentProcessor(const FPArgs& args,
                                                         const Matrix* uvMatrix) const override;

 private:
  ShaderConfig config;
};
}  // namespace tgfx
