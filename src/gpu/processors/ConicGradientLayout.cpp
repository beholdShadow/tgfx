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

#include "ConicGradientLayout.h"

namespace tgfx {
ConicGradientLayout::ConicGradientLayout(Matrix matrix, float bias, float scale)
    : FragmentProcessor(ClassID()), coordTransform(matrix), bias(bias), scale(scale) {
  addCoordTransform(&coordTransform);
}

bool ConicGradientLayout::onIsEqual(const FragmentProcessor& processor) const {
  const auto& that = static_cast<const ConicGradientLayout&>(processor);
  return coordTransform.matrix == that.coordTransform.matrix && bias == that.bias &&
         scale == that.scale;
}
}  // namespace tgfx