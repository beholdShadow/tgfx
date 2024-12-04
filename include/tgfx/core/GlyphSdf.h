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
#include "tgfx/core/Image.h"
#include "tgfx/core/Path.h"
namespace tgfx {
struct GlyphSdf {
    // Rect bitmapBox = Rect::MakeEmpty();
    float sdfPadding = 0.0f;
    std::shared_ptr<ImageBuffer> buffer = nullptr;
    Path path = {};
    GlyphSdf() = default;
    // 拷贝构造函数
    GlyphSdf(const GlyphSdf& other) {
        this->sdfPadding = other.sdfPadding;
        this->buffer = other.buffer;
        this->path = {};
        this->path.addPath(other.path);
    }
    // 赋值运算符
    GlyphSdf& operator=(const GlyphSdf& other) {
        if (this == &other) { // 防止自我赋值
            return *this;
        }
        this->sdfPadding = other.sdfPadding;
        this->buffer = other.buffer;
        this->path = {};
        this->path.addPath(other.path);
        return *this;
    }

};
}  // namespace tgfx
