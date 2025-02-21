#include "tgfx/platform/StringUtil.h"
#include <CoreText/CoreText.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
namespace tgfx {
std::vector<std::string> StringUtil::SplitFromPlatform(const std::string& text) {
    __block std::vector<std::string> result;
    // 将 std::string (UTF-8) 转换为 NSString
    NSString *str = [NSString stringWithUTF8String:text.c_str()];
    if (!str) {
        // 处理无效的 UTF-8 字符串（可选：返回空向量或抛出异常）
        return result;
    }
    // 使用 NSString 的枚举方法处理字符
    [str enumerateSubstringsInRange:NSMakeRange(0, [str length])
                            options:NSStringEnumerationByComposedCharacterSequences
                         usingBlock:^(NSString *substring, NSRange substringRange, NSRange enclosingRange, BOOL *stop) {
        std::string cStr = [substring UTF8String];
        result.push_back(cStr);
    }];
    return result;
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
