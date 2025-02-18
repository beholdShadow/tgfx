#include "tgfx/platform/StringUtil.h"
#include <CoreText/CoreText.h>
#include <CoreFoundation/CoreFoundation.h>

namespace tgfx {
std::vector<std::string> StringUtil::SplitToCharArray(const std::string& text) {
    // 调用 JavaScript 的 split 方法
    std::vector<std::string> res;
    return res;
}

bool StringUtil::IsEmoji(const std::string& text) {
    if (text.empty()) return false;

    // 将 UTF-8 字符串转换为 CFStringRef
    CFStringRef cfText = CFStringCreateWithCString(kCFAllocatorDefault, text.c_str(), kCFStringEncodingUTF8);
    if (!cfText) return false;

    // 创建属性字符串
    CFAttributedStringRef attrString = CFAttributedStringCreate(kCFAllocatorDefault, cfText, nullptr);
    if (!attrString) {
        CFRelease(cfText);
        return false;
    }

    // 创建 CTLine
    CTLineRef line = CTLineCreateWithAttributedString(attrString);
    if (!line) {
        CFRelease(attrString);
        CFRelease(cfText);
        return false;
    }

    // 获取 CTRun 数组
    CFArrayRef runs = CTLineGetGlyphRuns(line);
    if (!runs || CFArrayGetCount(runs) == 0) {
        CFRelease(line);
        CFRelease(attrString);
        CFRelease(cfText);
        return false;
    }

    // 检查每个 CTRun 的字形
    bool isEmoji = false;
    for (CFIndex i = 0; i < CFArrayGetCount(runs); i++) {
        CTRunRef run = (CTRunRef)CFArrayGetValueAtIndex(runs, i);
        CFDictionaryRef attributes = CTRunGetAttributes(run);
        CTFontRef font = (CTFontRef)CFDictionaryGetValue(attributes, kCTFontAttributeName);

        // 检查字体是否是 Emoji 字体
        if (font) {
            CFStringRef fontName = CTFontCopyPostScriptName(font);
            if (fontName && CFStringCompare(fontName, CFSTR("AppleColorEmoji"), 0) == kCFCompareEqualTo) {
                isEmoji = true;
                CFRelease(fontName);
                break;
            }
            if (fontName) CFRelease(fontName);
        }
    }

    // 释放资源
    CFRelease(line);
    CFRelease(attrString);
    CFRelease(cfText);

    return isEmoji;
}   

}  // namespace tgfx