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

#include "RuntimeImageFilter.h"
#include "gpu/DrawingManager.h"
#include "gpu/TPArgs.h"
#include "gpu/processors/FragmentProcessor.h"
#include "gpu/proxies/RenderTargetProxy.h"
#include "images/ResourceImage.h"

namespace tgfx {
std::shared_ptr<ImageFilter> ImageFilter::Runtime(std::shared_ptr<RuntimeEffect> effect) {
  if (effect == nullptr) {
    return nullptr;
  }
  return std::make_shared<RuntimeImageFilter>(effect);
}

Rect RuntimeImageFilter::onFilterBounds(const Rect& srcRect) const {
  return effect->filterBounds(srcRect);
}

std::shared_ptr<TextureProxy> RuntimeImageFilter::lockTextureProxy(
    std::shared_ptr<Image> source, const Rect& clipBounds, const TPArgs& args,
    const SamplingOptions& sampling) const {
  auto renderTarget = RenderTargetProxy::MakeFallback(
      args.context, static_cast<int>(clipBounds.width()), static_cast<int>(clipBounds.height()),
      source->isAlphaOnly(), effect->sampleCount(), args.mipmapped);
  if (renderTarget == nullptr) {
    return nullptr;
  }
  // Request a texture proxy from the source image without mipmaps to save memory.
  // It may be ignored if the source image has preset mipmaps.
  TPArgs tpArgs(args.context, args.renderFlags, false);
  auto textureProxy = source->lockTextureProxy(tpArgs, sampling);
  if (textureProxy == nullptr) {
    return nullptr;
  }
  auto offset = Point::Make(-clipBounds.x(), -clipBounds.y());
  auto drawingManager = args.context->drawingManager();
  drawingManager->addRuntimeDrawTask(renderTarget, std::move(textureProxy), effect, offset);
  drawingManager->addTextureResolveTask(renderTarget);
  return renderTarget->getTextureProxy();
}

std::unique_ptr<FragmentProcessor> RuntimeImageFilter::asFragmentProcessor(
    std::shared_ptr<Image> source, const FPArgs& args, const SamplingOptions& sampling,
    const Matrix* uvMatrix) const {
  return makeFPFromTextureProxy(source, args, sampling, uvMatrix);
}
}  // namespace tgfx