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

#include "MeasureContext.h"

namespace tgfx {
void MeasureContext::clear() {
  bounds.setEmpty();
}

void MeasureContext::drawRect(const Rect& rect, const MCState& state, const FillStyle&) {
  addLocalBounds(rect, state);
}

void MeasureContext::drawRRect(const RRect& rRect, const MCState& state, const FillStyle&) {
  addLocalBounds(rRect.rect, state);
}

void MeasureContext::drawShape(std::shared_ptr<Shape> shape, const MCState& state,
                               const FillStyle&) {
  auto localBounds = shape->getBounds(state.matrix.getMaxScale());
  addLocalBounds(localBounds, state);
}

void MeasureContext::drawImage(std::shared_ptr<Image> image, const SamplingOptions&,
                               const MCState& state, const FillStyle&) {
  if (image == nullptr) {
    return;
  }
  auto rect = Rect::MakeWH(image->width(), image->height());
  addLocalBounds(rect, state);
}

void MeasureContext::drawImageRect(std::shared_ptr<Image>, const Rect& rect, const SamplingOptions&,
                                   const MCState& state, const FillStyle&) {
  addLocalBounds(rect, state);
}

void MeasureContext::drawGlyphRunList(std::shared_ptr<GlyphRunList> glyphRunList,
                                      const MCState& state, const FillStyle&,
                                      const Stroke* stroke) {
  auto localBounds = glyphRunList->getBounds(state.matrix.getMaxScale());
  if (stroke) {
    stroke->applyToBounds(&localBounds);
  }
  addLocalBounds(localBounds, state);
}

void MeasureContext::drawLayer(std::shared_ptr<Picture> picture, const MCState& state,
                               const FillStyle&, std::shared_ptr<ImageFilter> imageFilter) {
  if (picture == nullptr) {
    return;
  }
  auto deviceBounds = picture->getBounds(&state.matrix);
  if (imageFilter) {
    deviceBounds = imageFilter->filterBounds(deviceBounds);
  }
  addDeviceBounds(deviceBounds, state.clip);
}

void MeasureContext::drawPicture(std::shared_ptr<Picture> picture, const MCState& state) {
  if (picture != nullptr) {
    picture->playback(this, state);
  }
}

void MeasureContext::addLocalBounds(const Rect& localBounds, const MCState& state) {
  auto deviceBounds = state.matrix.mapRect(localBounds);
  addDeviceBounds(deviceBounds, state.clip);
}

void MeasureContext::addDeviceBounds(const Rect& deviceBounds, const Path& clip) {
  if (clip.isEmpty() && clip.isInverseFillType()) {
    bounds.join(deviceBounds);
    return;
  }
  auto intersectBounds = clip.getBounds();
  if (!intersectBounds.intersect(deviceBounds)) {
    return;
  }
  bounds.join(intersectBounds);
}
}  // namespace tgfx
