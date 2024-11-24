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

#include "RenderPass.h"
#include "core/utils/Log.h"

namespace tgfx {
bool RenderPass::begin(std::shared_ptr<RenderTargetProxy> renderTargetProxy) {
  if (renderTargetProxy == nullptr) {
    return false;
  }
  _renderTarget = renderTargetProxy->getRenderTarget();
  if (_renderTarget == nullptr) {
    return false;
  }
  _renderTargetTexture = renderTargetProxy->getTexture();
  drawPipelineStatus = DrawPipelineStatus::NotConfigured;
  return true;
}

void RenderPass::end() {
  _renderTarget = nullptr;
  _renderTargetTexture = nullptr;
  resetActiveBuffers();
}

void RenderPass::resetActiveBuffers() {
  _program = nullptr;
  _indexBuffer = nullptr;
  _vertexBuffer = nullptr;
}

void RenderPass::bindProgramAndScissorClip(const ProgramInfo* programInfo, const Rect& drawBounds) {
  resetActiveBuffers();
  if (!onBindProgramAndScissorClip(programInfo, drawBounds)) {
    drawPipelineStatus = DrawPipelineStatus::FailedToBind;
    return;
  }
  drawPipelineStatus = DrawPipelineStatus::Ok;
}

void RenderPass::bindBuffers(std::shared_ptr<GpuBuffer> indexBuffer,
                             std::shared_ptr<GpuBuffer> vertexBuffer) {
  if (drawPipelineStatus != DrawPipelineStatus::Ok) {
    return;
  }
  _indexBuffer = std::move(indexBuffer);
  _vertexBuffer = std::move(vertexBuffer);
  _vertexData = nullptr;
}

void RenderPass::bindBuffers(std::shared_ptr<GpuBuffer> indexBuffer,
                             std::shared_ptr<Data> vertexData) {
  if (drawPipelineStatus != DrawPipelineStatus::Ok) {
    return;
  }
  _indexBuffer = std::move(indexBuffer);
  _vertexBuffer = nullptr;
  _vertexData = std::move(vertexData);
  ;
}

void RenderPass::draw(PrimitiveType primitiveType, size_t baseVertex, size_t vertexCount) {
  if (drawPipelineStatus != DrawPipelineStatus::Ok) {
    return;
  }
  size_t maxCount = context->caps()->maxBufferVertices;
  size_t n = vertexCount / maxCount + 1;
  size_t lastCount = vertexCount % maxCount;
  for (size_t i = 0; i < n; i++) {
    // LOGE("RenderPass::draw base : %d, count : %d", baseVertex + i * maxCount, i == n - 1 ? lastCount : maxCount); 
    onDraw(primitiveType, baseVertex + i * maxCount, i == n - 1 ? lastCount : maxCount);
  }
}

void RenderPass::drawIndexed(PrimitiveType primitiveType, size_t baseIndex, size_t indexCount) {
  if (drawPipelineStatus != DrawPipelineStatus::Ok) {
    return;
  }
  size_t maxCount = context->caps()->maxBufferVertices;
  size_t n = indexCount / maxCount + 1;
  size_t lastCount = indexCount % maxCount;
  for (size_t i = 0; i < n; i++) {
    // LOGE("RenderPass::draw base : %d, count : %d", baseVertex + i * maxCount, i == n - 1 ? lastCount : maxCount); 
    onDrawIndexed(primitiveType, baseIndex + i * maxCount, i == n - 1 ? lastCount : maxCount);
  }
}

void RenderPass::clear(const Rect& scissor, Color color) {
  drawPipelineStatus = DrawPipelineStatus::NotConfigured;
  onClear(scissor, color);
}
}  // namespace tgfx
