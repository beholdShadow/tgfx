/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Tencent is pleased to support the open source community by making tgfx available.
//
//  Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "StrokeShape.h"
#include "core/shapes/PathShape.h"
#include "core/utils/UniqueID.h"

namespace tgfx {
Rect StrokeShape::getBounds(float resolutionScale) const {
  auto bounds = shape->getBounds(resolutionScale);
  stroke.applyToBounds(&bounds);
  return bounds;
}

Path StrokeShape::getPath(float resolutionScale) const {
  auto path = shape->getPath(resolutionScale);
  stroke.applyToPath(&path, resolutionScale);
  return path;
}

bool StrokeShape::isRect(Rect* rect) const {
  if (stroke.cap == LineCap::Round) {
    return false;
  }
  Point line[2] = {};
  if (!shape->isLine(line)) {
    return false;
  }
  // check if the line is axis-aligned
  if (line[0].x != line[1].x && line[0].y != line[1].y) {
    return false;
  }
  // use the stroke width and line cap to convert the line to a rect
  auto left = std::min(line[0].x, line[1].x);
  auto top = std::min(line[0].y, line[1].y);
  auto right = std::max(line[0].x, line[1].x);
  auto bottom = std::max(line[0].y, line[1].y);
  auto halfWidth = stroke.width / 2.0f;
  if (stroke.cap == LineCap::Square) {
    if (rect) {
      rect->setLTRB(left - halfWidth, top - halfWidth, right + halfWidth, bottom + halfWidth);
    }
    return true;
  }
  if (rect) {
    if (left == right) {
      rect->setLTRB(left - halfWidth, top, right + halfWidth, bottom);
    } else {
      rect->setLTRB(left, top - halfWidth, right, bottom + halfWidth);
    }
  }
  return true;
}

UniqueKey StrokeShape::getUniqueKey() const {
  static const auto WidthStrokeShapeType = UniqueID::Next();
  static const auto CapJoinStrokeShapeType = UniqueID::Next();
  static const auto FullStrokeShapeType = UniqueID::Next();
  auto hasMiter = stroke.join == LineJoin::Miter && stroke.miterLimit != 4.0f;
  auto hasCapJoin = (hasMiter || stroke.cap != LineCap::Butt || stroke.join != LineJoin::Miter);
  size_t count = 2 + (hasCapJoin ? 1 : 0) + (hasMiter ? 1 : 0);
  auto type =
      hasCapJoin ? (hasMiter ? FullStrokeShapeType : CapJoinStrokeShapeType) : WidthStrokeShapeType;
  BytesKey bytesKey(count);
  bytesKey.write(type);
  bytesKey.write(stroke.width);
  if (hasCapJoin) {
    bytesKey.write(static_cast<uint32_t>(stroke.join) << 16 | static_cast<uint32_t>(stroke.cap));
  }
  if (hasMiter) {
    bytesKey.write(stroke.miterLimit);
  }
  return UniqueKey::Append(shape->getUniqueKey(), bytesKey.data(), bytesKey.size());
}
}  // namespace tgfx